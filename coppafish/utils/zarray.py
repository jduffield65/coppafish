import zarr
import numpy.typing as npt
from numcodecs import Blosc
import numpy as np
from typing import Optional, Union, Tuple, List
try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp

from ..setup import NotebookPage
from .. import utils


def save_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, image: npt.NDArray[np.int32],
              t: int, r: int, c: Optional[int] = None, num_rotations: int = 0, suffix: str = '') -> None:
    """
    Wrapper function to save tiles as zarr files with correct shift. Moves z-axis to first axis before saving as it is 
    quicker to load in this order. Tile `t` is saved to the path `nbp_file.tile[t,r,c]`, the path must contain an 
    extension of `'.zarr'`. The tile is saved as a `uint16`, so clipping may occur if the image contains really large 
    values.

    Args:
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        image (`[ny x nx x nz] ndarray[int32]` or `[n_channels x ny x nx] ndarray[int32]`): image to save.
        t (int): zarr tile index considering.
        r (int): round considering.
        c (int, optional): channel considering. Default: not given, raises error when `nbp_basic.is_3d == True`.
        num_rotations (int, optional): Number of `90` degree clockwise rotations to apply to image before saving. 
            Applied to the `x` and `y` axes, to 3d `image` data only. Default: `0`.
        suffix (str, optional): suffix to add to file name. Default: no suffix.
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
        file_path = file_path[:file_path.index('.zarr')] + suffix + '.zarr'
        # We chunk each z plane individually, since single z planes are often retrieved. We chunk so that each chunk is 
        # at least 1MB, as suggested in the documentation.
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
        chunks = (None, 750, 750)
        zarray = zarr.open(file_path, mode='w', zarr_version=2, shape=image.shape, chunks=chunks, dtype='|u2', 
                           synchronizer=zarr.ThreadSynchronizer(), compressor=compressor)
        zarray[:] = image
    if not nbp_basic.is_3d:
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
        file_path = file_path[file_path.index('.zarr'):] + suffix + '.zarr'
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
        chunks = (750, 750)
        zarray = zarr.open(file_path, mode='w', zarr_version=2, shape=image.shape, chunks=chunks, dtype='|u2', 
                           synchronizer=zarr.ThreadSynchronizer(), compressor=compressor)
        zarray[:] = image


def load_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, t: int, r: int, c: int,
              yxz: Optional[Union[List, Tuple, np.ndarray, jnp.ndarray]] = None,
              apply_shift: bool = True, suffix: str = '') -> npt.NDArray[Union[np.int32, np.uint16]]:
    """
    Loads in image corresponding to desired tile, round and channel from the relevant zarr file.

    Args:
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        t (int): zarr tile index considering.
        r (int): round considering.
        c (int): channel considering.
        yxz (`list` of `int` or `ndarray[int]`, optional): If `None`, whole image is loaded otherwise there are two 
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
        May want to disable `apply_shift` to save memory and/or make loading quicker as there will be no dtype 
        conversion. If loading in DAPI, dtype always `uint16` as there is no shift.
    """
    file_path = nbp_file.tile[t][r][c]
    file_path = file_path[:file_path.index('.zarr')] + suffix + '.zarr'
    if yxz is not None:
        #TODO: Find a quicker way of loading in parts of the image using zarr
        if isinstance(yxz, (list, tuple)):
            if nbp_basic.is_3d:
                if len(yxz) != 3:
                    raise ValueError(f'Loading in a 3D tile but dimension of coordinates given is {len(yxz)}.')
                if yxz[0] is None and yxz[1] is None:
                    # image = zarr.load(file_path)[yxz[2]]
                    zarray = zarr.open(file_path, mode='r')
                    image = zarray[yxz[2]]
                    if image.ndim == 3:
                        image = np.moveaxis(image, 0, 2)
                else:
                    coord_index = np.ix_(yxz[0], yxz[1], yxz[2])
                    image = np.moveaxis(zarr.load(file_path), 0, 2)[coord_index]
            else:
                if len(yxz) != 2:
                    raise ValueError(f'Loading in a 2D tile but dimension of coordinates given is {len(yxz)}.')
                coord_index = np.ix_(np.array([c]), yxz[0], yxz[1])  # add channel as first coordinate in 2D.
                # [0] below is to remove channel index of length 1.
                # image = zarr.load(nbp_file.tile[t][r])[coord_index][0]
                zarray = zarr.open(file_path, mode='r')
                image = zarray[coord_index][0]
        elif isinstance(yxz, (np.ndarray, jnp.ndarray)):
            if nbp_basic.is_3d:
                if yxz.shape[1] != 3:
                    raise ValueError(f'Loading in a 3D tile but dimension of coordinates given is {yxz.shape[1]}.')
                coord_index = tuple(np.asarray(yxz[:, i]) for i in range(3))
                # image = np.moveaxis(zarr.load(file_path), 0, 2)[coord_index]
                zarray = zarr.open(file_path, mode='r')
                image = zarray[:]
                image = np.moveaxis(image, 0, 2)[coord_index]
            else:
                if yxz.shape[1] != 2:
                    raise ValueError(f'Loading in a 2D tile but dimension of coordinates given is {yxz.shape[1]}.')
                coord_index = tuple(np.asarray(yxz[:, i]) for i in range(2))
                coord_index = (np.full(yxz.shape[0], c, int),) + coord_index  # add channel as first coordinate in 2D.
                # image = zarr.load(nbp_file.tile[t][r])[coord_index]
                zarray = zarr.open(file_path, mode='r')
                image = zarray[coord_index]
        else:
            raise ValueError(f'yxz should either be an [n_spots x n_dim] array to return an n_spots array indicating '
                             f'the value of the image at these coordinates or \n'
                             f'a list containing {2 + int(nbp_basic.is_3d)} arrays indicating the sub image to load.')
    else:
        if nbp_basic.is_3d :
            # image = np.moveaxis(zarr.load(file_path), 0, 2)
            zarray = zarr.open(file_path, mode='r')
            image = np.moveaxis(zarray[:], 0, 2)
        else:
            # image = zarr.load(nbp_file.tile[t][r])[c]
            zarray = zarr.open(nbp_file.tile[t][r], mode='r')
            image = zarray[c]
    if apply_shift and not (r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel):
        image = image.astype(np.int32) - nbp_basic.tile_pixel_value_shift
    return image
