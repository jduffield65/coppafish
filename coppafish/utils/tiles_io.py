import os
import zarr
import numbers
import itertools
import numpy as np
from numcodecs import Blosc
try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp
from tqdm import tqdm
import numpy_indexed
import numpy.typing as npt
from typing import List, Tuple, Union, Optional

from ..setup import NotebookPage
from .. import utils, extract


def tile_exists(file_path: str, file_type: str) -> bool:
    """
    Checks if a tile exists at the given path locations.

    Args:
        file_path (str): tile path.
        file_type (str): file type.

    Returns:
        bool: tile existence.

    Raises:
        ValueError: unsupported file type.
    """
    if file_type.lower() == '.npy':
        return os.path.isfile(file_path)
    elif file_type.lower() == '.zarr':
        # Require a non-empty zarr directory
        return os.path.isdir(file_path) and len(os.listdir(file_path)) > 0
    else:
        raise ValueError(f'Unsupported file_type: {file_type.lower()}')


def _save_image(image: Union[npt.NDArray[np.uint16], jnp.ndarray], file_path: str, file_type: str) -> None:
    """
    Save image in `file_path` location.

    Args:
        image (ndarray[uint16]): image to save.
        file_path (str): file path.
        file_type (str): file type.

    Raises:
        ValueError: unsupported file type.
    """
    if file_type.lower() == '.npy':
        np.save(file_path, image)
    elif file_type.lower() == '.zarr':
        # We chunk each z plane individually, since single z planes are often retrieved. We also chunk x and y so 
        # that each chunk is at least 1MB, as suggested in the zarr documentation.
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
        chunks = (None, 750, 750)
        zarray = zarr.open(
            file_path, mode='w', zarr_version=2, shape=image.shape, chunks=chunks, dtype='|u2', 
            synchronizer=zarr.ThreadSynchronizer(), compressor=compressor)
        zarray[:] = image
    else:
        raise ValueError(f'Unsupported `file_type`: {file_type.lower()}')


def _load_image(file_path: str, file_type: str, mmap_mode: str = None) -> Union[npt.NDArray[np.uint16], zarr.Array]:
    """
    Read in image from file_path location.

    Args:
        file_path (str): image location.
        file_type (str): file type. Either `'.npy'` or `'.zarr'`.
        mmap_mode (str, optional): the mmap_mode for numpy loading only. Default: no mapping.

    Returns `ndarray[uint16]` or `zarr.Array[uint16]`: loaded image.

    Raises:
        ValueError: unsupported file type.
    """
    if file_type.lower() == '.npy':
        return np.load(file_path, mmap_mode=mmap_mode)
    elif file_type.lower() == '.zarr':
        return zarr.open(file_path, mode='r')
    else:
        raise ValueError(f'Unsupported `file_type`: {file_type.lower()}')


def save_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, file_type: str, image: npt.NDArray[np.int32], t: int, 
              r: int, c: Optional[int] = None, num_rotations: int = 0, suffix: str = '') -> None:
    """
    Wrapper function to save tiles as npy files with correct shift. Moves z-axis to first axis before saving as it is 
    quicker to load in this order. Tile `t` is saved to the path `nbp_file.tile[t,r,c]`, the path must contain an 
    extension of `'.npy'`. The tile is saved as a `uint16`, so clipping may occur if the image contains really large 
    values.

    Args:
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        file_type (str): the saving file type. Can be `'.npy'` or `'.zarr'`.
        image (`[ny x nx x nz] ndarray[int32]` or `[n_channels x ny x nx] ndarray[int32]`): image to save.
        t (int): npy tile index considering.
        r (int): round considering.
        c (int, optional): channel considering. Default: not given, raises error when `nbp_basic.is_3d == True`.
        num_rotations (int, optional): number of `90` degree clockwise rotations to apply to image before saving. 
            Applied to the `x` and `y` axes, to 3d `image` data only. Default: `0`.
        suffix (str, optional): suffix to add to file name before the file extension. Default: empty.
    """
    assert image.ndim == 3, '`image` must be 3 dimensional'

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
        expected_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
        if not utils.errors.check_shape(image, expected_shape):
            raise utils.errors.ShapeError("tile to be saved", image.shape, expected_shape)
        # yxz -> zxy
        image = np.swapaxes(image, 2, 0)
        # zxy -> zyx
        image = np.swapaxes(image, 1, 2)
        # Now rotate image
        if num_rotations != 0:
            image = np.rot90(image, k=num_rotations, axes=(1, 2))
        file_path = nbp_file.tile[t][r][c]
        file_path= file_path[:file_path.index(file_type)] + suffix + file_type
        _save_image(image, file_path, file_type)
    else:
        # Don't need to apply rotations here as 2D data obtained from upstairs microscope without this issue
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
        file_path = nbp_file.tile[t][r][c]
        file_path = file_path[file_path.index(file_type):] + suffix + file_type
        _save_image(image, file_path, file_type)


def load_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, file_type: str, t: int, r: int, c: int, 
              yxz: Optional[Union[List, Tuple, np.ndarray, jnp.ndarray]] = None, apply_shift: bool = True, 
              suffix: str = '') -> npt.NDArray[Union[np.int32, np.uint16]]:
    """
    Loads in image corresponding to desired tile, round and channel from the relevant npy file.

    Args:
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        file_type (str): the saved file type. Either `'.npy'` or `'.zarr'`.
        t (int): npy tile index considering.
        r (int): round considering.
        c (int): channel considering.
        yxz (`list` of `int` or `ndarray[int]`, optional): if `None`, whole image is loaded otherwise there are two 
            choices 
            - `list` of `int [2 or 3]`. List containing y,x,z coordinates of sub image to load in.
                E.g. if `yxz = [np.array([5]), np.array([10,11,12]), np.array([8,9])]`
                returned `image` will have shape `[1 x 3 x 2]`.
                if `yxz = [None, None, z_planes]`, all pixels on given z_planes will be returned
                i.e. shape of image will be `[tile_sz x tile_sz x n_z_planes]`.
            - `[n_pixels x (2 or 3)] ndarray[int]`. Array containing yxz coordinates for which the pixel value is 
                desired. E.g. if `yxz = np.ones((10,3))`, returned `image` will have shape `[10,]` with all values 
                indicating the pixel value at `[1,1,1]`.
            Default: `None`. 
            
        apply_shift (bool, optional): if true, dtype will be `int32` otherwise dtype will be `uint16` with the pixels 
            values shifted by `+nbp_basic.tile_pixel_value_shift`. Default: true.
        suffix (str, optional): suffix to add to file name to load from. Default: no suffix.

    Returns:
        `int32 [ny x nx (x nz)]` or `int32 [n_pixels x (2 or 3)]`
            Loaded image.

    Notes:
        - May want to disable `apply_shift` to save memory and/or make loading quicker as there will be no dtype 
        conversion. If loading in DAPI, dtype is always `uint16` as there is no shift.
    """
    if nbp_basic.is_3d:
        file_path = nbp_file.tile[t][r][c]
        file_path = file_path[:file_path.index(file_type)] + suffix + file_type
    if yxz is not None:
        # Use mmap when only loading in part of image
        if isinstance(yxz, (list, tuple)):
            if nbp_basic.is_3d:
                if len(yxz) != 3:
                    raise ValueError(f'Loading in a 3D tile but dimension of coordinates given is {len(yxz)}.')
                if yxz[0] is None and yxz[1] is None:
                    try:
                        image = _load_image(file_path, file_type, mmap_mode='r')[yxz[2]]
                    except ValueError:
                        image = _load_image(file_path, file_type, mmap_mode='r+')[yxz[2]]
                    if image.ndim == 3:
                        image = np.moveaxis(image, 0, 2)
                else:
                    coord_index = np.ix_(yxz[0], yxz[1], yxz[2])
                    image = np.moveaxis(_load_image(file_path, file_type, mmap_mode='r'), 0, 2)[coord_index]
            else:
                if len(yxz) != 2:
                    raise ValueError(f'Loading in a 2D tile but dimension of coordinates given is {len(yxz)}.')
                coord_index = np.ix_(np.array([c]), yxz[0], yxz[1])  # add channel as first coordinate in 2D.
                # [0] below is to remove channel index of length 1.
                image = _load_image(nbp_file.tile[t][r], file_type, mmap_mode='r')[coord_index][0]
        elif isinstance(yxz, (np.ndarray, jnp.ndarray)):
            if nbp_basic.is_3d:
                if yxz.shape[1] != 3:
                    raise ValueError(f'Loading in a 3D tile but dimension of coordinates given is {yxz.shape[1]}.')
                coord_index = tuple(np.asarray(yxz[:, i]) for i in range(3))
                image = np.moveaxis(_load_image(file_path, file_type, mmap_mode='r'), 0, 2)[coord_index]
            else:
                if yxz.shape[1] != 2:
                    raise ValueError(f'Loading in a 2D tile but dimension of coordinates given is {yxz.shape[1]}.')
                coord_index = tuple(np.asarray(yxz[:, i]) for i in range(2))
                coord_index = (np.full(yxz.shape[0], c, int),) + coord_index  # add channel as first coordinate in 2D.
                # image = np.load(nbp_file.tile[t][r], mmap_mode='r')[coord_index]
                image = _load_image(nbp_file.tile[t][r], file_type, mmap_mode='r')[coord_index]
        else:
            raise ValueError(f'yxz should either be an [n_spots x n_dim] array to return an n_spots array indicating '
                             f'the value of the image at these coordinates or \n'
                             f'a list containing {2 + int(nbp_basic.is_3d)} arrays indicating the sub image to load.')
    else:
        if nbp_basic.is_3d :
            # Don't use mmap when loading in whole image
            image = np.moveaxis(_load_image(file_path, file_type), 0, 2)
        else:
            # Use mmap when only loading in part of image
            image = _load_image(file_path, file_type, mmap_mode='r')[c]
    if apply_shift and not (r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel):
        image = image.astype(np.int32) - nbp_basic.tile_pixel_value_shift
    return image


def load_full_tile(
        nbp_file: NotebookPage, nbp_basic: NotebookPage, file_type: str, tile: int, apply_shift: bool = False, 
    ) -> npt.NDArray[Union[np.int32, np.uint16]]:
    """
    Disk load every round and channel pixel values for a particular tile.

    Args:
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.
        file_type (str): saved file type.
        tile (int): tile index.
        apply_shift (bool, optional): Whether to apply shift to image pixels. Default: false.

    Returns:
        `[n_rounds x n_channels x ny x nx (x nz)] ndarray[uint16 or int32]` tile image. If `apply_shift` is true, the 
            ndarray is int32.
    """
    use_rounds, use_channels = nbp_basic.use_rounds, nbp_basic.use_channels
    use_indices = np.zeros(
        (nbp_basic.n_rounds + nbp_basic.use_anchor + nbp_basic.use_preseq, nbp_basic.n_channels), dtype=bool, 
    )
    for r, c in itertools.product(use_rounds + nbp_basic.use_preseq * [nbp_basic.pre_seq_round], use_channels):
        use_indices[r, c] = True
    use_indices[nbp_basic.anchor_round, nbp_basic.anchor_channel] = True
    
    tile_side_length = nbp_basic.tile_sz
    if nbp_basic.is_3d:
        image_shape = (*use_indices.shape, tile_side_length, tile_side_length, nbp_basic.nz)
    else:
        image_shape = (*use_indices.shape, tile_side_length, tile_side_length)
    
    image_tile = np.zeros(image_shape, dtype=np.int32 if apply_shift else np.uint16)
    for r, c in np.argwhere(use_indices):
        image_tile[r, c] = utils.tiles_io.load_tile(
            nbp_file, nbp_basic, file_type, tile, r, c, apply_shift=False, 
            suffix='_raw' if r == nbp_basic.pre_seq_round else ''
        )
    return image_tile
    

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
        Corresponding indices in npy file.
    """
    if isinstance(tile_ind_nd2, numbers.Number):
        tile_ind_nd2 = [tile_ind_nd2]
    npy_index = numpy_indexed.indices(tile_pos_yx_npy, tile_pos_yx_nd2[tile_ind_nd2]).tolist()
    if len(npy_index) == 1:
        return npy_index[0]
    else:
        return npy_index


def save_stitched(im_file: Union[str, None], nbp_file: NotebookPage, nbp_basic: NotebookPage, 
                  nbp_extract: NotebookPage, tile_origin: np.ndarray, r: int, c: int, from_raw: bool = False, 
                  zero_thresh: int = 0, num_rotations: int = 1) -> None:
    """
    Stitches together all tiles from round `r`, channel `c` and saves the resultant compressed npz at `im_file`. Saved 
    image will be uint16 if from nd2 or from DAPI filtered npy files. Otherwise, if from filtered npy files, will 
    remove shift and re-scale to fill int16 range.

    Args:
        im_file (str or none): path to save file. If `None`, stitched `image` is returned (with z axis last) instead of 
            saved.
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_extract (NotebookPage): `extract` notebook page.
        tile_origin (`[n_tiles x 3] ndarray[float]`): yxz origin of each tile on round `r`.
        r (int): save_stitched will save stitched image of all tiles of round `r`, channel `c`.
        c (int): save_stitched will save stitched image of all tiles of round `r`, channel `c`.
        from_raw (bool, optional): if `False`, will stitch together tiles from saved npy files, otherwise will load in 
            raw un-filtered images from nd2/npy file. Default: false.
        zero_thresh (int, optional): all pixels with absolute value less than or equal to `zero_thresh` will be set to 
            0. The larger it is, the smaller the compressed file will be. Default: 0.
        num_rotations (int, optional): the number of rotations to apply to each tile individually. Default: `1`, the 
            same as the notebook default.
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
        round_dask_array = utils.raw.load_dask(nbp_file, nbp_basic, r=r)
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
                image_t = utils.raw.load_image(nbp_file, nbp_basic, t, c, round_dask_array, r, nbp_basic.use_z)
                # replicate non-filtering procedure in extract_and_filter
                if not nbp_basic.is_3d:
                    image_t = extract.focus_stack(image_t)
                image_t, bad_columns = extract.strip_hack(image_t)  # find faulty columns
                image_t[:, bad_columns] = 0
                if nbp_basic.is_3d:
                    image_t = np.moveaxis(image_t, 2, 0)  # put z-axis back to the start
                if num_rotations != 0:
                    image_t = np.rot90(image_t, k=num_rotations, axes=(1, 2))
            else:
                if nbp_basic.is_3d:
                    image_t = load_tile(nbp_file, nbp_basic, nbp_extract.file_type, t, r, c).transpose((2,0,1))
                else:
                    image_t = load_tile(nbp_file, nbp_basic, nbp_extract.file_type, t, r, c, apply_shift=False)
            for z in range(z_size):
                # any tiles not used will be kept as 0.
                pbar.set_postfix({'tile': t, 'z': z})
                if nbp_basic.is_3d:
                    file_z = z - z_origin[t]
                    if file_z < 0 or file_z >= len(nbp_basic.use_z):
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
        # Break things up by z plane so that not everything needs to be stored in ram at once
        im_max = np.abs(stitched_image).max()
        for z in range(stitched_image.shape[0]):
            stitched_image[z] = stitched_image[z] - shift
            stitched_image[z] = stitched_image[z] * np.iinfo(np.int16).max / im_max

        stitched_image = np.rint(stitched_image, np.zeros_like(stitched_image, dtype=np.int16), casting='unsafe')
    if zero_thresh > 0:
        stitched_image[np.abs(stitched_image) <= zero_thresh] = 0

    if im_file is None:
        if z_size > 1:
            stitched_image = np.moveaxis(stitched_image, 0, -1)
        return stitched_image
    else:
        np.savez_compressed(im_file, stitched_image)
