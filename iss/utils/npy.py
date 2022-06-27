import numpy as np
from ..setup import NotebookPage
from .. import utils, extract
import jax.numpy as jnp
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
import os


def save_tile(nbp_file: NotebookPage, nbp_basic: NotebookPage, image: np.ndarray,
              t: int, r: int, c: Optional[int] = None):
    """
    Wrapper function to save tiles as npy files with correct shift.
    Moves z-axis to start before saving as it is quicker to load in this order.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        nbp_extract_debug: `extract_debug` notebook page
        image: `int32 [ny x nx x nz]` or `int32 [n_channels x ny x nx]`.
            Image to save.
        t: tiff tile index considering
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
        t: tiff tile index considering
        r: Round considering
        c: Channel considering
        yxz: If None, whole image is loaded otherwise there are two choices:
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


def save_stitched(im_file: str, nbp_file: NotebookPage, nbp_basic: NotebookPage, tile_origin: np.ndarray,
                  r: int, c: int, from_nd2: bool = False):
    """
    Stitches together all tiles from round `r`, channel `c` and saves the resultant compressed uint16 npz at `im_file`.

    Args:
        im_file: Path to save file.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        tile_origin: `float [n_tiles x 3]`.
            yxz origin of each tile on round `r`.
        r: save_stitched will save stitched image of all tiles of round `r`, channel `c`.
        c: save_stitched will save stitched image of all tiles of round `r`, channel `c`.
        from_nd2: If False, will stitch together tiles from saved npy files,
            otherwise will load in raw un-filtered images from nd2 file.
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
    if from_nd2:
        if nbp_basic.use_anchor:
            # always have anchor as first round after imaging rounds
            round_files = nbp_file.round + [nbp_file.anchor]
        else:
            round_files = nbp_file.round
        im_file = os.path.join(nbp_file.input_dir, round_files[r] + nbp_file.raw_extension)
        nd2_all_images = utils.nd2.load(im_file)
    with tqdm(total=z_size * len(nbp_basic.use_tiles)) as pbar:
        for t in nbp_basic.use_tiles:
            if from_nd2:
                image_t = utils.nd2.get_image(nd2_all_images,
                                              extract.get_nd2_tile_ind(t, nbp_basic.tilepos_yx_nd2,
                                                                       nbp_basic.tilepos_yx),
                                              c, nbp_basic.use_z)
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
    np.savez_compressed(im_file, stitched_image)
