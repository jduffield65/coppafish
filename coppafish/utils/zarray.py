import zarr
import numpy as np
from ..setup import NotebookPage
from .. import utils
from typing import Optional, Union, Tuple, List
try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp


def save_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, image: np.ndarray,
              t: int, r: int, c: Optional[int] = None, num_rotations: int = 0, suffix: str = ''):
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
        num_rotations: Number of rotations to apply to image before saving. (Default = 0, done from y to x axis)
        suffix: Suffix to add to file name.
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
        # yxz -> zxy
        image = np.swapaxes(image, 2, 0)
        # zxy -> zyx
        image = np.swapaxes(image, 1, 2)
        # Now rotate image
        if num_rotations != 0:
            image = np.rot90(image, k=num_rotations, axes=(1, 2))
        file_path = nbp_file.tile[t][r][c]
        file_path = file_path[:file_path.index('.zarr')] + suffix + '.zarr'
        # We chunk each z plane individually, since single z planes are often retrieved
        chunks = (None, image.shape[1]//10, image.shape[2]//10)
        zarray = zarr.open(file_path, mode='w', zarr_version=2, shape=image.shape, chunks=chunks, dtype='|u2')
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
        # We chunk each z plane individually, since single z planes are often retrieved
        chunks = (image.shape[0]//10, image.shape[1]//10)
        zarray = zarr.open(file_path, mode='w', zarr_version=2, shape=image.shape, chunks=chunks, dtype='|u2')
        zarray[:] = image


def load_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, t: int, r: int, c: int,
              yxz: Optional[Union[List, Tuple, np.ndarray, jnp.ndarray]] = None,
              apply_shift: bool = True, suffix: str = '') -> np.ndarray:
    """
    Loads in image corresponding to desired tile, round and channel from the relevant zarr file.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        t: npy tile index considering
        r: Round considering
        c: Channel considering
        yxz: If `None`, whole image is loaded otherwise there are two choices:
            - `int [2 or 3]`. `list` containing y,x,z coordinates of sub image to load in.
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
        suffix: Suffix to add to file name to load from.

    Returns:
        `(ny x nx (x nz)) ndarray[int32]` or `(n_pixels x (2 or 3)) ndarray[int32]`
            Loaded image.
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
