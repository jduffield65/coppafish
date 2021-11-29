import tifffile
# import zarr
import numpy as np
from . import errors


def save(image, im_file, description=None, append=False, move_z_axis=True):
    """
    save image as tiff at path given by im_file

    :param image: 2d (or 3d) numpy array [ny x nx (x nz or n_channels)]
    :param im_file: string. path to save file
    :param description: string, short description to save to metadata to describe image
        found after saving through tifffile.TiffFile(im_file).pages[0].tags["ImageDescription"].value
        default: None
    :param append: boolean. Whether to add to file if it exists or replace, optional.
        default: False
    :param move_z_axis: boolean. Whether to move axes in position 2 to position 0.
    """
    # truncate image so don't get aliased values
    image[image > np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max
    image[image < 0] = 0
    image = np.round(image).astype(np.uint16)
    if image.ndim == 3 and move_z_axis:
        # put dimension that is not y or x as first dimension so easier to load in a single plane later
        # and match MATLAB method of saving
        image = np.moveaxis(image, 2, 0)
    tifffile.imwrite(im_file, image, append=append, description=description)


def save_tile(nbp_file, nbp_basic, nbp_extract_params, image, t, c, r):
    """
    wrapper function to save tiles as tiff files with correct shift and a short description in the metadata

    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param nbp_extract_params: NotebookPage object containing extract parameters
    :param image: numpy float array [ny x nx (x nz)]
    :param t: integer, tiff tile index considering
    :param c: integer, channel considering
    :param r: integer, round considering
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
        errors.wrong_shape('tile image', image, [nbp_basic['tile_sz'], nbp_basic['tile_sz'], nbp_basic['nz']])
        save(image, nbp_file['tile'][t][r][c], append=False, description=description, move_z_axis=True)
    else:
        errors.wrong_shape('tile image', image, [nbp_basic['tile_sz'], nbp_basic['tile_sz']])
        save(image, nbp_file['tile'][t][r], append=True, description=description, move_z_axis=True)


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


def load_tile(nbp_file, nbp_basic, t, c, r, y=None, x=None, z=None, nbp_extract_params=None):
    """
    load tile t, channel c, round r with pixel value shift subtracted if not DAPI.

    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param t: integer, tiff tile index considering
    :param c: integer, channel considering
    :param r: integer, round considering
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
        errors.check_tiff_description(nbp_file, nbp_basic, nbp_extract_params, t, c, r)
    if nbp_basic['3d']:
        image = load(nbp_file['tile'][t][r][c], z, y, x)
        # throw error if tile not expected shape
        # only for case where y and x not specified as we know that if they are then load gives correct result
        if y is None and x is None:
            if z is None:
                exp_z_shape = nbp_basic['nz']
            elif len(np.array([z]).flatten()) == 1:
                exp_z_shape = 1
            else:
                exp_z_shape = len(z)
            if exp_z_shape == 1:
                errors.wrong_shape('loaded tile', image, [nbp_basic['tile_sz'], nbp_basic['tile_sz']])
            else:
                errors.wrong_shape('loaded tile', image, [nbp_basic['tile_sz'], nbp_basic['tile_sz'],
                                                                    nbp_basic['nz']])
    else:
        image = load(nbp_file['tile'][t][r], c, y, x)
        # throw error if not expected shape
        if y is None and x is None:
            errors.wrong_shape('loaded tile', image, [nbp_basic['tile_sz'], nbp_basic['tile_sz']])
    if r == nbp_basic['anchor_round'] and c == nbp_basic['anchor_channel']:
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


def load_tile_description(nbp_file, nbp_basic, t, c, r):
    """
    load in description saved in tiff for tile t, channel c, round r.

    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param t: integer, tiff tile index considering
    :param c: integer, channel considering
    :param r: integer, round considering
    :return: string
    """
    if nbp_basic['3d']:
        description = load_description(nbp_file['tile'][t][r][c])
    else:
        description = load_description(nbp_file['tile'][t][r], c)
    return description
