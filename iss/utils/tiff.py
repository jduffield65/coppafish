import tifffile
# import zarr
import numpy as np
import re
from . import errors
import warnings
from tqdm import tqdm


def save(image, im_file, description=None, append=False):
    """
    save image as tiff at path given by im_file
    If image has 3 dimensions, axis at position 2 is moved to position 0.

    :param image: 2d (or 3d) numpy array [ny x nx (x nz or n_channels)]
    :param im_file: string. path to save file
    :param description: string, short description to save to metadata to describe image
        found after saving through tifffile.TiffFile(im_file).pages[0].tags["ImageDescription"].value
        default: None
    :param append: boolean. Whether to add to file if it exists or replace, optional.
        default: False
    """
    # truncate image so don't get aliased values
    image[image > np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max
    image[image < 0] = 0
    image = np.round(image).astype(np.uint16)
    if image.ndim == 3:
        # put dimension that is not y or x as first dimension so easier to load in a single plane later
        # and match MATLAB method of saving
        image = np.moveaxis(image, 2, 0)
    tifffile.imwrite(im_file, image, append=append, description=description)


def save_tile(nbp_file, nbp_basic, nbp_extract_params, image, t, r, c):
    """
    wrapper function to save tiles as tiff files with correct shift and a short description in the metadata

    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param nbp_extract_params: NotebookPage object containing extract parameters
    :param image: numpy float array [ny x nx (x nz)]
    :param t: integer, tiff tile index considering
    :param r: integer, round considering
    :param c: integer, channel considering
    """
    if r == nbp_basic['anchor_round']:
        round = "anchor"
        if c == nbp_basic['anchor_channel']:
            scale = nbp_extract_params['scale_anchor']
            shift = nbp_basic['tile_pixel_value_shift']
            channel = "anchor"
        elif c == nbp_basic['dapi_channel']:
            scale = 1
            shift = 0
            channel = "dapi"
        else:
            scale = 1
            shift = 0
            channel = "not used"
    else:
        round = r
        if c not in nbp_basic['use_channels']:
            scale = 1
            shift = 0
            channel = "not used"
        else:
            scale = nbp_extract_params['scale']
            shift = nbp_basic['tile_pixel_value_shift']
            channel = c
    description = f"Tile = {t}. Round = {round}. Channel = {channel}. Shift = {shift}. Scale = {scale}"
    image = image + shift
    if nbp_basic['3d']:
        expected_shape = (nbp_basic['tile_sz'], nbp_basic['tile_sz'], nbp_basic['nz'])
        if not errors.check_shape(image, expected_shape):
            raise errors.ShapeError("tile to be saved", image.shape, expected_shape)
        save(image, nbp_file['tile'][t][r][c], append=False, description=description)
    else:
        expected_shape = (nbp_basic['tile_sz'], nbp_basic['tile_sz'])
        if not errors.check_shape(image, expected_shape):
            raise errors.ShapeError("tile to be saved", image.shape, expected_shape)
        save(image, nbp_file['tile'][t][r], append=True, description=description)


def load(im_file, planes=None, y_roi=None, x_roi=None):
    """
    load tiff from path given by im_file

    :param im_file: string, path to tiff to load
    :param planes: integer or integer list/numpy array, optional.
        which planes in tiff file to load
        default: None meaning all planes
    :param y_roi: integer or integer list/numpy array, optional.
        y pixels to read in
        default: None meaning all y pixels
    :param x_roi: integer or integer list/numpy array
        x pixels to read in
        default: None meaning all x pixels
    :return: numpy array [len(y_roi) x len(x_roi) (x len(planes))]
    """
    image = tifffile.imread(im_file, key=planes)
    if image.ndim == 3:
        image = np.moveaxis(image, 0, 2)
    if y_roi is not None or x_roi is not None:
        if y_roi is None:
            y_roi = np.arange(image.shape[0])
        if x_roi is None:
            x_roi = np.arange(image.shape[1])
        y_roi = np.array([y_roi]).flatten().reshape(-1, 1)
        x_roi = np.array([x_roi]).flatten().reshape(1, -1).repeat(len(y_roi), 0)
        image = image[y_roi.reshape(-1, 1), x_roi]
    return image


def load_tile(nbp_file, nbp_basic, t, r, c, y=None, x=None, z=None, nbp_extract_params=None):
    """
    load tile t, channel c, round r with pixel value shift subtracted if not DAPI.

    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param t: integer, tiff tile index considering
    :param r: integer, round considering
    :param c: integer, channel considering
    :param y: integer or integer list/numpy array, optional.
        y pixels to read in
        default: None meaning all y pixels
    :param x: integer or integer list/numpy array
        x pixels to read in
        default: None meaning all x pixels
    :param z: integer or integer list/numpy array, optional.
        which z-planes in tiff file to load
        default: None meaning all z-planes
    :param nbp_extract_params: NotebookPage object containing extract parameters, optional.
        provide nbp_extract_params if want to check scale and shift in it match those used to make tiffs
        default: None
    :return:
        numpy (uint16 if dapi otherwise int32) array [ny x nx (x nz)]
    """
    if nbp_extract_params is not None:
        description = load_tile_description(nbp_file, nbp_basic, t, r, c)
        if "Scale = " in description and "Shift = " in description:
            scale_tiff, shift_tiff = get_scale_shift_from_tiff(description)
            scale_nbp, shift_nbp = get_scale_shift_from_nbp(nbp_basic, nbp_extract_params, r, c)
            if scale_tiff != scale_nbp or shift_tiff != shift_nbp:
                raise errors.TiffError(scale_tiff, scale_nbp, shift_tiff, shift_nbp)
        else:
            warnings.warn(f"\nTiff description is: \n{description}"
                          f"\nIt contains no information on Scale or Shift used to make it.")
    if nbp_basic['3d']:
        image = load(nbp_file['tile'][t][r][c], z, y, x)
        # throw error if tile not expected shape
        # only for case where y and x not specified as we know that if they are then load gives correct result
        if y is None and x is None:
            if z is None:
                expected_z_shape = nbp_basic['nz']
            elif len(np.array([z]).flatten()) == 1:
                expected_z_shape = 1
            else:
                expected_z_shape = len(z)
            if expected_z_shape == 1:
                expected_shape = (nbp_basic['tile_sz'], nbp_basic['tile_sz'])
            else:
                expected_shape = (nbp_basic['tile_sz'], nbp_basic['tile_sz'], nbp_basic['nz'])
            if not errors.check_shape(image, expected_shape):
                raise errors.ShapeError("loaded tile", image.shape, expected_shape)
    else:
        image = load(nbp_file['tile'][t][r], c, y, x)
        # throw error if not expected shape
        if y is None and x is None:
            expected_shape = (nbp_basic['tile_sz'], nbp_basic['tile_sz'])
            if not errors.check_shape(image, expected_shape):
                raise errors.ShapeError("loaded tile", image.shape, expected_shape)
    if r == nbp_basic['anchor_round'] and c == nbp_basic['dapi_channel']:
        pass
    else:
        # change from uint16 to int to ensure no info loss when subtract shift
        image = image.astype(int) - nbp_basic['tile_pixel_value_shift']
    return image


def load_description(im_file, plane=0):
    """
    loads in description from tiff file.
    if no description, "N/A" is returned

    :param im_file: string, path to tiff to load
    :param plane: plane to load description from, optional.
        default: 0
    :return: string
    """
    dict = tifffile.TiffFile(im_file).pages[plane].tags
    description = dict.get("ImageDescription")
    if description:
        description = description.value
    else:
        description = "N/A"
    return description


def load_tile_description(nbp_file, nbp_basic, t, r, c):
    """
    load in description saved in tiff for tile t, channel c, round r.

    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param t: integer, tiff tile index considering
    :param r: integer, round considering
    :param c: integer, channel considering
    :return: string
    """
    if nbp_basic['3d']:
        description = load_description(nbp_file['tile'][t][r][c])
    else:
        description = load_description(nbp_file['tile'][t][r], c)
    return description


def get_scale_shift_from_tiff(description):
    """
    Returns scale and shift values detailed in the tiff description

    :param description: string, description saved in tiff file.
    :return:
        scale: float, scale factor applied to tiff
        shift: integer, shift applied to tiff to ensure pixel values positive.
    """
    # scale value is after 'Scale = ' in description
    scale = np.float64(description.split("Scale = ", 1)[1])
    # shift value is between 'Shift = ' and '. Scale = ' in description
    shift = int(re.findall(r'Shift = (.+?). Scale = ', description)[0])
    return scale, shift


def get_scale_shift_from_nbp(nbp_basic, nbp_extract_params, r, c):
    """
    Returns scale and shift values detailed in notebook.

    :param nbp_basic: NotebookPage object containing basic info
    :param nbp_extract_params: NotebookPage object containing extract parameters
    :param r: integer, round considering
    :param c: integer, channel considering
    :return:
        scale: float, scale factor applied to tiff, found from nb['extract_params']['scale']
        shift: integer, shift applied to tiff to ensure pixel values positive.
            Found from nb['basic_info']['tile_pixel_value_shift']
    """
    shift = nbp_basic['tile_pixel_value_shift']
    if r == nbp_basic['anchor_round'] and c == nbp_basic['anchor_channel']:
        scale = nbp_extract_params['scale_anchor']
    elif r != nbp_basic['anchor_round'] and c in nbp_basic['use_channels']:
        scale = nbp_extract_params['scale']
    else:
        scale = 1  # dapi image and un-used channels have no scaling
        shift = 0  # dapi image and un-used channels have no shift
    return scale, shift


def save_stitched(im_file, nbp_file, nbp_basic, tile_origin, r, c):
    """
    stitches together all tiles from round r, channel c and saves the resultant tiff at im_file.

    :param im_file: string. path to save file
    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param tile_origin: numpy float array [n_tiles x 3]. yxz origin of each tile on round r.
    :param r: integer, this will save stitched image of all tiles of round r, channel c.
    :param c: integer, this will save stitched image of all tiles of round r, channel c.
    """
    yx_origin = np.round(tile_origin[:, :2]).astype(int)
    z_origin = np.round(tile_origin[:, 2]).astype(int).flatten()
    yx_size = np.max(yx_origin, axis=0) + nbp_basic['tile_sz']
    if nbp_basic['3d']:
        z_size = z_origin.max() + nbp_basic['nz']
    else:
        z_size = 1
    with tqdm(total=z_size * len(nbp_basic['use_tiles'])) as pbar:
        for z in range(z_size):
            stitched_image = np.zeros(yx_size, dtype=np.uint16)  # any tiles not used will be kept as 0.
            for t in nbp_basic['use_tiles']:
                pbar.set_postfix({'tile': t, 'z': z})
                if nbp_basic['3d']:
                    file_z = z - z_origin[t]
                    if file_z < 0 or file_z >= nbp_basic['nz']:
                        # Set tile to 0 if currently outside its area
                        local_image = np.zeros((nbp_basic['tile_sz'], nbp_basic['tile_sz']))
                    else:
                        local_image = load(nbp_file['tile'][t][r][c], file_z)
                else:
                    local_image = load(nbp_file['tile'][t][r], c)
                stitched_image[yx_origin[t, 0]:yx_origin[t, 0]+nbp_basic['tile_sz'],
                               yx_origin[t, 1]:yx_origin[t, 1]+nbp_basic['tile_sz']] = local_image
                pbar.update(1)
            save(stitched_image, im_file, append=True)
    pbar.close()
