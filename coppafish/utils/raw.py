import os
import re
import numpy as np
from typing import List, Optional, Union
import warnings
from .errors import OutOfBoundsError
from ..utils import nd2
from ..setup import NotebookPage
import dask.array
from tqdm import tqdm


def get_tile_indices(folder: str) -> List:
    """
    Returns the number in all `file_names` contained in `folder`.
    It assumes a single number in each `file_name`.

    !!! note
        A `file_name` with no number will be ignored but a `file_name` with two numbers will return both.
    Args:
        folder: Folder which contains files

    Returns:
        `int [n_files_with_single_number]`.
            Numbers contained in file_names
    """
    delimiters = "[a-z]+", "s+", "_", "."   # Any letter, white space, _ or . used as delimiter
    regex_pattern = '|\\'.join(delimiters)
    file_names = os.listdir(folder)
    file_names.sort()  # put in ascending order
    tiles = [int(s) for file_name in file_names for s in re.split(regex_pattern, file_name) if s.isdigit()]
    if len(tiles) != len(file_names):
        warnings.warn(f'Number of files in {folder} is {len(file_names)} but found {len(tiles)} numbers.\n'
                      f'So some files have no numbers / multiple numbers in their name.')
    return tiles


def metadata_sanity_check(metadata: dict, round_folder_path: str) -> List:
    """
    Checks whether information in `metadata` matches what we can determine by reading in raw data.
    This is only called when `nb.file_names.raw_extension == '.npy'`.

    Args:
        metadata: Dictionary containing -
            - `xy_pos` - `List [n_tiles x 2]`. xy position of tiles in pixels.
            - `pixel_microns` - `float`. xy pixel size in microns.
            - `pixel_microns_z` - `float`. z pixel size in microns.
            - `sizes` - dict with fov (`t`), channels (`c`), y, x, z-planes (`z`) dimensions.
        round_folder_path: Path to a folder containing raw .npy files

    Returns: `nd2` tile indices indicated by `.npy` files present in `round_folder_path`.

    """
    tiles = get_tile_indices(round_folder_path)
    if max(tiles) >= metadata['sizes']['t']:
        raise OutOfBoundsError("tiles", max(tiles), 0, metadata['sizes']['t']-1)
    file_names = os.listdir(round_folder_path)
    raw_data_0_path = os.path.join(round_folder_path, file_names[0])
    raw_data_0 = np.load(raw_data_0_path, mmap_mode='r')  # mmap as don't need actual data
    _, n_channels, n_y, n_x, n_z = raw_data_0.shape
    if n_channels != metadata['sizes']['c']:
        raise ValueError(f"Number of channels in the metadata is {metadata['sizes']['c']} "
                         f"but the file\n{raw_data_0_path}\ncontains {n_channels} channels.")
    if n_y != metadata['sizes']['y']:
        raise ValueError(f"y_dimension in the metadata is {metadata['sizes']['y']} "
                         f"but the file\n{raw_data_0_path}\nhas y_dimension = {n_y}.")
    if n_x != metadata['sizes']['x']:
        raise ValueError(f"x_dimension in the metadata is {metadata['sizes']['x']} "
                         f"but the file\n{raw_data_0_path}\nhas x_dimension = {n_x}.")
    if n_z != metadata['sizes']['z']:
        raise ValueError(f"z_dimension in the metadata is {metadata['sizes']['z']} "
                         f"but the file\n{raw_data_0_path}\nhas z_dimension = {n_z}.")
    return tiles


def load_dask(nbp_file: NotebookPage, nbp_basic: NotebookPage, r: int) -> dask.array.Array:
    """
    Loads in the memmap round_dask_array containing images for all tiles/channels in the round.
    This can later be used to load in more quickly.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        r: Round considering (anchor will be assumed to be the last round if using).
    Returns:
        Dask array with indices in order `fov`, `channel`, `y`, `x`, `z`.

    """

    n_tiles, tile_sz, nz = nbp_basic.n_tiles, nbp_basic.tile_sz, nbp_basic.nz
    if nbp_basic.channel_laser is not None:
        channel_laser = list(set(nbp_basic.channel_laser))
    else:
        channel_laser = []
    channel_laser.sort()
    if nbp_basic.channel_camera is not None:
        channel_cam = list(set(nbp_basic.channel_camera))
    else:
        channel_cam = []
    channel_cam.sort()

    # If the camera argument is left blank in the notebook, this just means we only have one camera
    # Not sure how this applies to lasers
    n_lasers, n_cams = max(1, len(np.unique(channel_laser))), max(1, len(np.unique(channel_cam)))

    if not np.isin(nbp_file.raw_extension, ['.nd2', '.npy', 'jobs']):
        raise ValueError(f"nbp_file.raw_extension must be '.nd2', '.npy' or 'jobs' but it is {nbp_file.raw_extension}.")

    if not nbp_file.raw_extension == 'jobs':
        if nbp_basic.use_anchor:
            # always have anchor as first round after imaging rounds
            round_files = nbp_file.round + [nbp_file.anchor]
        else:
            round_files = nbp_file.round
        round_file = os.path.join(nbp_file.input_dir, round_files[r])
        if nbp_file.raw_extension == '.nd2':
            round_dask_array = nd2.load(round_file + nbp_file.raw_extension)
        elif nbp_file.raw_extension == '.npy':
            round_dask_array = dask.array.from_npy_stack(round_file)
    else:
        # Now deal with the case where files are split by laser
        round_laser_dask_array = []
        # Deal with non anchor round first as this follows a different format to anchor round
        if r != nbp_basic.anchor_round:

            #Get all files of a given round
            round_files = nbp_file.round[r*n_tiles*n_lasers:(r+1)*n_tiles*n_lasers]

            for t in tqdm(range(n_tiles), desc='Loading tiles in dask array'):
                #Get all the files of a given tiles (should be 7)
                tile_files = round_files[t*n_lasers: (t+1)*n_lasers]
                tile_dask_array = []

                for f in tile_files:
                    laser_file = os.path.join(nbp_file.input_dir, f + '.nd2')
                    tile_dask_array.append(nd2.load(laser_file))

                tile_da = dask.array.concatenate(tile_dask_array, axis=-1)  # concatenate on the laser axis
                tile_da = dask.array.swapaxes(tile_da, -1, 0)  # we need 'channel', 'z', 'y','x'

                round_laser_dask_array.append(tile_da)

        # If we're dealing with the anchor round, we only need channel 0 (DAPI) and anchor
        # The rest of the array is padded with zeros

        else:

            anchor_laser_index = channel_laser.index(nbp_basic.channel_laser[nbp_basic.anchor_channel])
            dapi_laser_index = channel_laser.index(nbp_basic.channel_laser[nbp_basic.dapi_channel])

            anchor_files = nbp_file.anchor

            for t in tqdm(range(n_tiles), desc='Loading tiles in dask array'):
                # Get all the files of a given tiles (should be 7)
                tile_files = anchor_files[t * n_lasers: (t + 1) * n_lasers]
                tile_dask_array = []

                for f_id, f in enumerate(tile_files):
                    if f_id == anchor_laser_index or f_id == dapi_laser_index:
                        laser_file = os.path.join(nbp_file.input_dir, f + '.nd2')
                        tile_dask_array.append(nd2.load(laser_file))
                    else:
                        tile_dask_array.append(dask.array.zeros((nz+1, tile_sz, tile_sz, n_cams)))
                        # TODO find a better fix for nz. here it is different because of basic_info use_z
                        # Ideally it should have the same shape as the array for dapi

                tile_da = dask.array.concatenate(tile_dask_array, axis=-1)  # concatenate on the laser axis
                tile_da = dask.array.swapaxes(tile_da, -1, 0)  # we need 'channel', 'z', 'y','x'

                round_laser_dask_array.append(tile_da)

            # now concatenate dask arrays
        round_dask_array = dask.array.stack(round_laser_dask_array, axis=0)

    return round_dask_array


def load_image(nbp_file: NotebookPage, nbp_basic: NotebookPage, t: int, c: int,
               round_dask_array: Optional[dask.array.Array] = None, r: Optional[int] = None,
               use_z: Optional[List[int]] = None) -> np.ndarray:
    """
    Loads in raw data either from npy stack or nd2 file for tile and channel specified. If no round dask array specified
    we load one in.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        t: npy tile index considering.
        c: Channel considering.
        round_dask_array: Dask array with indices in order `fov`, `channel`, `y`, `x`, `z`.
            If None, will load it in first.
        r: Round considering (anchor will be assumed to be the last round if using).
            Don't need to provide if give round_dask_array.
        use_z: z-planes to load in.
            If use_z = None, will load in all z-planes.
    Returns:
        numpy array [n_y x n_x x len(use_z)].
    """
    if not np.isin(nbp_file.raw_extension, ['.nd2', '.npy', 'jobs']):
        raise ValueError(f"nbp_file.raw_extension must be '.nd2', '.npy' or 'jobs' but it is {nbp_file.raw_extension}.")

    if round_dask_array is None:
        if nbp_basic.use_anchor:
            # always have anchor as first round after imaging rounds
            round_files = nbp_file.round + [nbp_file.anchor]
        else:
            round_files = nbp_file.round

        if nbp_file.raw_extension == '.nd2':
            round_file = os.path.join(nbp_file.input_dir, round_files[r])
            round_dask_array = nd2.load(round_file + nbp_file.raw_extension)
        elif nbp_file.raw_extension == '.npy':
            round_file = os.path.join(nbp_file.input_dir, round_files[r])
            round_dask_array = dask.array.from_npy_stack(round_file)

    # Return a tile/channel/z-planes from the dask array.
    if use_z is None:
        use_z = nbp_basic.use_z

    t_nd2 = nd2.get_nd2_tile_ind(t, nbp_basic.tilepos_yx_nd2, nbp_basic.tilepos_yx)
    if nbp_file.raw_extension == '.nd2':
        # Only need this if statement because nd2.get_image will be different if use nd2reader not nd2 module
        # which is needed on M1 mac.
        return nd2.get_image(round_dask_array, t_nd2, c, use_z)
    elif nbp_file.raw_extension == '.npy':
        # Need the with below to silence warning
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            return np.asarray(round_dask_array[t_nd2, c, :, :, use_z])
    elif nbp_file.raw_extension == 'jobs':
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            return np.asarray(round_dask_array[t_nd2, c, :, :, use_z])
