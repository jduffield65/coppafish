import numpy as np
from ..setup import NotebookPage
from .. import utils, extract
try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
import numpy_indexed
import numbers


def save_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, image: np.ndarray,
              t: int, r: int, c: Optional[int] = None):
    """
    Wrapper function to save tiles as npy files with correct shift.
    Moves z-axis to start before saving as it is quicker to load in this order.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        image: `int32 [ny x nx x nz]` or `int32 [n_channels x ny x nx]`.
            Image to save.
        t: npy tile index considering
        r: Round considering
        c: Channel considering
    """
    if nbp_basic.is_3d:
        if c is None:
            raise ValueError('3d image but channel not given.')
        if r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel:
            # If dapi is given then image should already by uint16 so no clipping
            image = image.astype(np.uint16)
        else:
            # need to shift and clip image so fits into uint16 dtype.
            # clip at 1 not 0 because 0 (or -tile_pixel_value_shift)
            # will be used as an invalid value when reading in spot_colors.
            image = np.clip(image + nbp_basic.tile_pixel_value_shift, 1, np.iinfo(np.uint16).max,
                            np.zeros_like(image, dtype=np.uint16), casting="unsafe")
        # In 3D, cannot possibly save any un-used channel hence no exception for this case.
        expected_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz)
        if not utils.errors.check_shape(image, expected_shape):
            raise utils.errors.ShapeError("tile to be saved", image.shape, expected_shape)
        np.save(nbp_file.tile[t][r][c], np.moveaxis(image, 2, 0))
    else:
        if r == nbp_basic.anchor_round:
            if nbp_basic.anchor_channel is not None:
                # If anchor round, only shift and clip anchor channel, leave DAPI and un-used channels alone.
                image[nbp_basic.anchor_channel] = \
                    np.clip(image[nbp_basic.anchor_channel] + nbp_basic.tile_pixel_value_shift, 1,
                            np.iinfo(np.uint16).max, image[nbp_basic.anchor_channel])
            image = image.astype(np.uint16)
            use_channels = [val for val in [nbp_basic.dapi_channel, nbp_basic.anchor_channel] if val is not None]
        else:
            image = np.clip(image + nbp_basic.tile_pixel_value_shift, 1, np.iinfo(np.uint16).max,
                            np.zeros_like(image, dtype=np.uint16), casting="unsafe")
            use_channels = nbp_basic.use_channels
        # set un-used channels to be 0, not clipped to 1.
        image[np.setdiff1d(np.arange(nbp_basic.n_channels), use_channels)] = 0

        expected_shape = (nbp_basic.n_channels, nbp_basic.tile_sz, nbp_basic.tile_sz)
        if not utils.errors.check_shape(image, expected_shape):
            raise utils.errors.ShapeError("tile to be saved", image.shape, expected_shape)
        np.save(nbp_file.tile[t][r], image)


def load_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, t: int, r: int, c: int,
              yxz: Optional[Union[List, Tuple, np.ndarray, jnp.ndarray]] = None,
              apply_shift: bool = True) -> np.ndarray:
    """
    Loads in image corresponding to desired tile, round and channel from the relavent npy file.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        t: npy tile index considering
        r: Round considering
        c: Channel considering
        yxz: If `None`, whole image is loaded otherwise there are two choices:

            - `int [2 or 3]`. List containing y,x,z coordinates of sub image to load in.
                E.g. if `yxz = [np.array([5]), np.array([10,11,12]), np.array([8,9])]`
                returned `image` will have shape `[1 x 3 x 2]`.
                if `yxz = [None, None, z_planes]`, all pixels on given z_planes will be returned
                i.e. shape of image will be `[tile_sz x tile_sz x n_z_planes]`.
            - `int [n_pixels x (2 or 3)]`. Array containing yxz coordinates for which the pixel value is desired.
                E.g. if `yxz = np.ones((10,3))`,
                returned `image` will have shape `[10,]` with all values indicating the pixel value at `[1,1,1]`.
        apply_shift: If `True`, dtype will be `int32` otherwise dtype will be `uint16`
            with the pixels values shifted by `+nbp_basic.tile_pixel_value_shift`.
            May want to disable `apply_shift` to save memory and/or make loading quicker as there will be
            no dtype conversion. If loading in DAPI, dtype always uint16 as is no shift.


    Returns:
        `int32 [ny x nx (x nz)]` or `int32 [n_pixels x (2 or 3)]`
            Loaded image.
    """
    if yxz is not None:
        # Use mmap when only loading in part of image
        if isinstance(yxz, (list, tuple)):
            if nbp_basic.is_3d:
                if len(yxz) != 3:
                    raise ValueError(f'Loading in a 3D tile but dimension of coordinates given is {len(yxz)}.')
                if yxz[0] is None and yxz[1] is None:
                    image = np.load(nbp_file.tile[t][r][c], mmap_mode='r')[yxz[2]]
                    if image.ndim == 3:
                        image = np.moveaxis(image, 0, 2)
                else:
                    coord_index = np.ix_(yxz[0], yxz[1], yxz[2])
                    image = np.moveaxis(np.load(nbp_file.tile[t][r][c], mmap_mode='r'), 0, 2)[coord_index]
            else:
                if len(yxz) != 2:
                    raise ValueError(f'Loading in a 2D tile but dimension of coordinates given is {len(yxz)}.')
                coord_index = np.ix_(np.array([c]), yxz[0], yxz[1])  # add channel as first coordinate in 2D.
                # [0] below is to remove channel index of length 1.
                image = np.load(nbp_file.tile[t][r], mmap_mode='r')[coord_index][0]
        elif isinstance(yxz, (np.ndarray, jnp.ndarray)):
            if nbp_basic.is_3d:
                if yxz.shape[1] != 3:
                    raise ValueError(f'Loading in a 3D tile but dimension of coordinates given is {yxz.shape[1]}.')
                coord_index = tuple(np.asarray(yxz[:, i]) for i in range(3))
                image = np.moveaxis(np.load(nbp_file.tile[t][r][c], mmap_mode='r'), 0, 2)[coord_index]
            else:
                if yxz.shape[1] != 2:
                    raise ValueError(f'Loading in a 2D tile but dimension of coordinates given is {yxz.shape[1]}.')
                coord_index = tuple(np.asarray(yxz[:, i]) for i in range(2))
                coord_index = (np.full(yxz.shape[0], c, int),) + coord_index  # add channel as first coordinate in 2D.
                image = np.load(nbp_file.tile[t][r], mmap_mode='r')[coord_index]
        else:
            raise ValueError(f'yxz should either be an [n_spots x n_dim] array to return an n_spots array indicating '
                             f'the value of the image at these coordinates or \n'
                             f'a list containing {2 + int(nbp_basic.is_3d)} arrays indicating the sub image to load.')
    else:
        if nbp_basic.is_3d:
            # Don't use mmap when loading in whole image
            image = np.moveaxis(np.load(nbp_file.tile[t][r][c]), 0, 2)
        else:
            # Use mmap when only loading in part of image
            image = np.load(nbp_file.tile[t][r], mmap_mode='r')[c]
    if apply_shift and not (r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel):
        image = image.astype(np.int32) - nbp_basic.tile_pixel_value_shift
    return image


def get_npy_tile_ind(tile_ind_nd2: Union[int, List[int]], tile_pos_yx_nd2: np.ndarray,
                     tile_pos_yx_npy: np.ndarray) -> Union[int, List[int]]:
    """
    Gets index of tile in npy file from tile index of nd2 file.

    Args:
        tile_ind_nd2: Index of tile in nd2 file
        tile_pos_yx_nd2: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with nd2 index ```i```.
            Index 0 refers to ```YX = [0, 0]```.
            Index 1 refers to ```YX = [0, 1] if MaxX > 0```.
        tile_pos_yx_npy: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with npy index ```i```.
            Index 0 refers to ```YX = [MaxY, MaxX]```.
            Index 1 refers to ```YX = [MaxY, MaxX - 1] if MaxX > 0```.

    Returns:
        Corresponding indices in npy file
    """
    if isinstance(tile_ind_nd2, numbers.Number):
        tile_ind_nd2 = [tile_ind_nd2]
    npy_index = numpy_indexed.indices(tile_pos_yx_npy, tile_pos_yx_nd2[tile_ind_nd2]).tolist()
    if len(npy_index) == 1:
        return npy_index[0]
    else:
        return npy_index



def save_stitched(im_file: Optional[str], nbp_file: NotebookPage, nbp_basic: NotebookPage, tile_origin: np.ndarray,
                  r: int, c: int, from_raw: bool = False, zero_thresh: int = 0):
    """
    Stitches together all tiles from round `r`, channel `c` and saves the resultant compressed npz at `im_file`.
    Saved image will be uint16 if from nd2 or from DAPI filtered npy files.
    Otherwise, if from filtered npy files, will remove shift and re-scale to fill int16 range.

    Args:
        im_file: Path to save file.
            If `None`, stitched `image` is returned (with z axis last) instead of saved.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        tile_origin: `float [n_tiles x 3]`.
            yxz origin of each tile on round `r`.
        r: save_stitched will save stitched image of all tiles of round `r`, channel `c`.
        c: save_stitched will save stitched image of all tiles of round `r`, channel `c`.
        from_raw: If `False`, will stitch together tiles from saved npy files,
            otherwise will load in raw un-filtered images from nd2/npy file.
        zero_thresh: All pixels with absolute value less than or equal to `zero_thresh` will be set to 0.
            The larger it is, the smaller the compressed file will be.\
        save: If True, saves image as im_file, otherwise returns image
    """
    yx_origin = np.round(tile_origin[:, :2]).astype(int)
    z_origin = np.round(tile_origin[:, 2]).astype(int).flatten()
    yx_size = np.max(yx_origin, axis=0) + nbp_basic.tile_sz
    if nbp_basic.is_3d:
        z_size = z_origin.max() + nbp_basic.nz
        stitched_image = np.zeros(np.append(z_size, yx_size), dtype=np.uint16)
    else:
        z_size = 1
        stitched_image = np.zeros(yx_size, dtype=np.uint16)
    if from_raw:
        round_dask_array = utils.raw.load(nbp_file, nbp_basic, r=r)
        shift = 0  # if from nd2 file, data type is already un-shifted uint16
    else:
        if r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel:
            shift = 0  # if filtered dapi, data type is already un-shifted uint16
        else:
            # if from filtered npy files, data type is shifted uint16, want to save stitched as un-shifted int16.
            shift = nbp_basic.tile_pixel_value_shift
    if shift != 0:
        # change dtype to accommodate negative values and set base value to be zero in the shifted image.
        stitched_image = stitched_image.astype(np.int32) + shift
    with tqdm(total=z_size * len(nbp_basic.use_tiles)) as pbar:
        for t in nbp_basic.use_tiles:
            if from_raw:
                image_t = utils.raw.load(nbp_file, nbp_basic, round_dask_array, r, t, c, nbp_basic.use_z)
                # replicate non-filtering procedure in extract_and_filter
                if not nbp_basic.is_3d:
                    image_t = extract.focus_stack(image_t)
                image_t, bad_columns = extract.strip_hack(image_t)  # find faulty columns
                image_t[:, bad_columns] = 0
                if nbp_basic.is_3d:
                    image_t = np.moveaxis(image_t, 2, 0)  # put z-axis back to the start
            else:
                if nbp_basic.is_3d:
                    image_t = np.load(nbp_file.tile[t][r][c], mmap_mode='r')
                else:
                    image_t = load_tile(nbp_file, nbp_basic, t, r, c, apply_shift=False)
            for z in range(z_size):
                # any tiles not used will be kept as 0.
                pbar.set_postfix({'tile': t, 'z': z})
                if nbp_basic.is_3d:
                    file_z = z - z_origin[t]
                    if file_z < 0 or file_z >= nbp_basic.nz:
                        # Set tile to 0 if currently outside its area
                        local_image = np.zeros((nbp_basic.tile_sz, nbp_basic.tile_sz))
                    else:
                        local_image = image_t[file_z]
                    stitched_image[z, yx_origin[t, 0]:yx_origin[t, 0]+nbp_basic.tile_sz,
                                   yx_origin[t, 1]:yx_origin[t, 1]+nbp_basic.tile_sz] = local_image
                else:
                    stitched_image[yx_origin[t, 0]:yx_origin[t, 0]+nbp_basic.tile_sz,
                                   yx_origin[t, 1]:yx_origin[t, 1]+nbp_basic.tile_sz] = image_t
                pbar.update(1)
    pbar.close()
    if shift != 0:
        # remove shift and re-scale so fits the whole int16 range
        stitched_image = stitched_image - shift
        stitched_image = stitched_image * np.iinfo(np.int16).max / np.abs(stitched_image).max()
        stitched_image = np.rint(stitched_image, np.zeros_like(stitched_image, dtype=np.int16), casting='unsafe')
    if zero_thresh > 0:
        stitched_image[np.abs(stitched_image) <= zero_thresh] = 0

    if im_file is None:
        if z_size > 1:
            stitched_image = np.moveaxis(stitched_image, 0, -1)
        return stitched_image
    else:
        np.savez_compressed(im_file, stitched_image)
