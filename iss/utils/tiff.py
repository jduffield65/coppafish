import tifffile
# import zarr
import numpy as np
import re
from . import errors
import warnings
from tqdm import tqdm
from typing import Optional, Union, List, Tuple
from ..setup import NotebookPage


def save(image: np.ndarray, im_file: str, description: Optional[str] = None, append: bool = False):
    """
    Save image as tiff at path given by `im_file`.

    !!! note
        If image has 3 dimensions, axis at position 2 is moved to position 0.

        Any pixel values outside range of `uint16` are clamped before saving.
        I.e. `<0` set to `0` and `>np.iinfo(np.uint16).max` set to `np.iinfo(np.uint16).max`.

    Args:
        image: `int [ny x nx (x n_dim3)]`.
            2D or 3D image to be saved.
        im_file: Path to save file.
        description: Short description to save to metadata to describe image.
            Found after saving through `tifffile.TiffFile(im_file).pages[0].tags["ImageDescription"].value`.
        append: Whether to add to file if it exists or replace.
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


def save_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, nbp_extract_debug: NotebookPage, image: np.ndarray,
              t: int, r: int, c: int):
    """
    Wrapper function to save tiles as tiff files with correct shift and a short description in the metadata.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        nbp_extract_debug: `extract_debug` notebook page
        image: `float [ny x nx (x nz)]`.
            Image to save.
        t: tiff tile index considering
        r: Round considering
        c: Channel considering
    """
    if r == nbp_basic.anchor_round:
        round = "anchor"
        if c == nbp_basic.anchor_channel:
            scale = nbp_extract_debug.scale_anchor
            shift = nbp_basic.tile_pixel_value_shift
            channel = "anchor"
        elif c == nbp_basic.dapi_channel:
            scale = 1
            shift = 0
            channel = "dapi"
        else:
            scale = 1
            shift = 0
            channel = "not used"
    else:
        round = r
        if c not in nbp_basic.use_channels:
            scale = 1
            shift = 0
            channel = "not used"
        else:
            scale = nbp_extract_debug.scale
            shift = nbp_basic.tile_pixel_value_shift
            channel = c
    description = f"Tile = {t}. Round = {round}. Channel = {channel}. Shift = {shift}. Scale = {scale}"
    image = image + shift
    if nbp_basic.is_3d:
        expected_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz)
        if not errors.check_shape(image, expected_shape):
            raise errors.ShapeError("tile to be saved", image.shape, expected_shape)
        save(image, nbp_file.tile[t][r][c], append=False, description=description)
    else:
        expected_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz)
        if not errors.check_shape(image, expected_shape):
            raise errors.ShapeError("tile to be saved", image.shape, expected_shape)
        save(image, nbp_file.tile[t][r], append=True, description=description)


def load(im_file: str, planes: Optional[Union[int, List[int], np.ndarray]] = None,
         y_roi: Optional[Union[int, List[int], np.ndarray]] = None,
         x_roi: Optional[Union[int, List[int], np.ndarray]] = None) -> np.ndarray:
    """
    Load tiff from path given by `im_file`.

    !!! note
        If saved image in tiff file has 3 dimensions, axis at position 0 is moved to position 2 before returning.

    Args:
        im_file: Path to tiff to load.
        planes: `int [n_z_planes]`.
            Which planes in tiff file to load. `None` means all planes.
        y_roi: int [n_y_pixels].
            Y pixels to read in. `None` means all y pixels.
        x_roi: `int [n_x_pixels]`.
            X pixels to read in. `None` means all x pixels.

    Returns:
        `int [n_y_pixels x n_x_pixels (x n_z_planes)]`.
            Loaded image. If `n_z_planes = 1` or image in tiff file is 2D, 2D image is returned.
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


def load_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, t: int, r: int, c: int,
              y: Optional[Union[int, List[int], np.ndarray]] = None,
              x: Optional[Union[int, List[int], np.ndarray]] = None,
              z: Optional[Union[int, List[int], np.ndarray]] = None,
              nbp_extract_debug: Optional[NotebookPage] = None) -> np.ndarray:
    """

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        t: tiff tile index considering
        r: Round considering
        c: Channel considering
        y: `int [n_y_pixels]`.
            Y pixels to read in. `None` means all y pixels.
        x: `int [n_x_pixels]`.
            X pixels to read in. `None` means all x pixels.
        z: `int [n_z_planes]`.
            Which z-planes in tiff file to load. `None` means all z-planes.
        nbp_extract_debug: `extract_debug` notebook page.
            Provide `nbp_extract_debug` if want to check `scale` and `shift` in it match those used to make tiffs.

    Returns:
        `int [ny x nx (x nz)]`.
            Loaded image.
    """
    if nbp_extract_debug is not None:
        description = load_tile_description(nbp_file, nbp_basic, t, r, c)
        if "Scale = " in description and "Shift = " in description:
            scale_tiff, shift_tiff = get_scale_shift_from_tiff(description)
            scale_nbp, shift_nbp = get_scale_shift_from_nbp(nbp_basic, nbp_extract_debug, r, c)
            if scale_tiff != scale_nbp or shift_tiff != shift_nbp:
                raise errors.TiffError(scale_tiff, scale_nbp, shift_tiff, shift_nbp)
        else:
            warnings.warn(f"\nTiff description is: \n{description}"
                          f"\nIt contains no information on Scale or Shift used to make it.")
    if nbp_basic.is_3d:
        image = load(nbp_file.tile[t][r][c], z, y, x)
        # throw error if tile not expected shape
        # only for case where y and x not specified as we know that if they are then load gives correct result
        if y is None and x is None:
            if z is None:
                expected_z_shape = nbp_basic.nz
            elif len(np.array([z]).flatten()) == 1:
                expected_z_shape = 1
            else:
                expected_z_shape = len(z)
            if expected_z_shape == 1:
                expected_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz)
            else:
                expected_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz)
            if not errors.check_shape(image, expected_shape):
                raise errors.ShapeError("loaded tile", image.shape, expected_shape)
    else:
        image = load(nbp_file.tile[t][r], c, y, x)
        # throw error if not expected shape
        if y is None and x is None:
            expected_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz)
            if not errors.check_shape(image, expected_shape):
                raise errors.ShapeError("loaded tile", image.shape, expected_shape)
    if r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel:
        pass
    else:
        # change from uint16 to int to ensure no info loss when subtract shift
        image = image.astype(int) - nbp_basic.tile_pixel_value_shift
    return image


def load_description(im_file: str, plane: int = 0) -> str:
    """
    Loads in description from tiff file. If no description, `"N/A"` is returned

    Args:
        im_file: Path to tiff to load
        plane: Plane to load description from

    Returns:
        Description saved in tiff file. `"N/A"` if no description.
    """
    dict = tifffile.TiffFile(im_file).pages[plane].tags
    description = dict.get("ImageDescription")
    if description:
        description = description.value
    else:
        description = "N/A"
    return description


def load_tile_description(nbp_file: NotebookPage, nbp_basic: NotebookPage, t: int, r: int, c: int) -> str:
    """
    Loads in description saved in tiff for tile `t`, channel `c`, round `r`.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        t: tiff tile index considering
        r: Round considering
        c: Channel considering

    Returns:
        Description saved in tiff file. `"N/A"` if no description.
    """
    if nbp_basic.is_3d:
        description = load_description(nbp_file.tile[t][r][c])
    else:
        description = load_description(nbp_file.tile[t][r], c)
    return description


def get_scale_shift_from_tiff(description: str) -> Tuple[float, int]:
    """
    Returns `scale` and `shift` values detailed in the tiff description.

    Args:
        description: `description` saved in tiff file.

    Returns:
        - `scale` - `float`. Scale factor applied to tiff.
        - `shift` - `int`. Shift applied to tiff to ensure pixel values positive.
    """
    # scale value is after 'Scale = ' in description
    scale = float(description.split("Scale = ", 1)[1])
    # shift value is between 'Shift = ' and '. Scale = ' in description
    shift = int(re.findall(r'Shift = (.+?). Scale = ', description)[0])
    return scale, shift


def get_scale_shift_from_nbp(nbp_basic: NotebookPage, nbp_extract_debug: NotebookPage, r: int,
                             c: int) -> Tuple[float, int]:
    """
    Returns `scale` and `shift` values detailed in notebook.

    Args:
        nbp_basic: `basic_info` notebook page
        nbp_extract_debug: `extract_debug` notebook page
        r: Round considering
        c: Channel considering

    Returns:
        - `scale` - `float`. Scale factor applied to tiff. Found from `nb.extract_debug.scale`.
        - `shift` - `int`. Shift applied to tiff to ensure pixel values positive.
            Found from `nb.basic_info.tile_pixel_value_shift`.
    """
    shift = nbp_basic.tile_pixel_value_shift
    if r == nbp_basic.anchor_round and c == nbp_basic.anchor_channel:
        scale = nbp_extract_debug.scale_anchor
    elif r != nbp_basic.anchor_round and c in nbp_basic.use_channels:
        scale = nbp_extract_debug.scale
    else:
        scale = 1  # dapi image and un-used channels have no scaling
        shift = 0  # dapi image and un-used channels have no shift
    return scale, shift


def save_stitched(im_file: str, nbp_file: NotebookPage, nbp_basic: NotebookPage, tile_origin: np.ndarray,
                  r: int, c: int):
    """
    Stitches together all tiles from round `r`, channel `c` and saves the resultant tiff at `im_file`.

    Args:
        im_file: Path to save file.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        tile_origin: `float [n_tiles x 3]`.
            yxz origin of each tile on round `r`.
        r: save_stitched will save stitched image of all tiles of round `r`, channel `c`.
        c: save_stitched will save stitched image of all tiles of round `r`, channel `c`.
    """
    yx_origin = np.round(tile_origin[:, :2]).astype(int)
    z_origin = np.round(tile_origin[:, 2]).astype(int).flatten()
    yx_size = np.max(yx_origin, axis=0) + nbp_basic['tile_sz']
    if nbp_basic.is_3d:
        z_size = z_origin.max() + nbp_basic.nz
    else:
        z_size = 1
    with tqdm(total=z_size * len(nbp_basic.use_tiles)) as pbar:
        for z in range(z_size):
            stitched_image = np.zeros(yx_size, dtype=np.uint16)  # any tiles not used will be kept as 0.
            for t in nbp_basic.use_tiles:
                pbar.set_postfix({'tile': t, 'z': z})
                if nbp_basic.is_3d:
                    file_z = z - z_origin[t]
                    if file_z < 0 or file_z >= nbp_basic.nz:
                        # Set tile to 0 if currently outside its area
                        local_image = np.zeros((nbp_basic.tile_sz, nbp_basic.tile_sz))
                    else:
                        local_image = load(nbp_file.tile[t][r][c], file_z)
                else:
                    local_image = load(nbp_file.tile[t][r], c)
                stitched_image[yx_origin[t, 0]:yx_origin[t, 0]+nbp_basic.tile_sz,
                               yx_origin[t, 1]:yx_origin[t, 1]+nbp_basic.tile_sz] = local_image
                pbar.update(1)
            save(stitched_image, im_file, append=True)
    pbar.close()
