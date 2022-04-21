import numpy as np
import os
from typing import Tuple, Optional, List


def get_tilepos(xy_pos: np.ndarray, tile_sz: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Using `xy_pos` from nd2 metadata, this obtains the yx position of each tile.
    I.e. how tiles are arranged with respect to each other.
    Note that this is indexed differently in nd2 file and tiff files in the tile directory.

    Args:
        xy_pos: `float [n_tiles x 2]`.
            xy position of tiles in pixels. Obtained from nd2 metadata.
        tile_sz: xy dimension of tile in pixels.

    Returns:
        - `tilepos_yx_nd2` - `int [n_tiles x 2]`.
            `tilepos_yx_nd2[i, 0]` is y index of tile with fov index `i` in nd2 file.
            `tilepos_yx_nd2[i, 1]` is x index of tile with fov index `i` in nd2 file.
        - `tilepos_yx_tiff` - `int [n_tiles x 2]`.
            `tilepos_yx_tiff[i, 0]` is y index of tile with tile directory (tiff files) index `i`.
            `tilepos_yx_tiff[i, 1]` is x index of tile with tile directory (tiff files) index `i`.
    """
    tilepos_yx_nd2 = np.zeros_like(xy_pos, dtype=int)
    if np.shape(xy_pos)[0] != 1:
        # say y coordinate changes when successive tiles have pixel separation of more than tile_sz/2
        change_y_coord = np.abs(np.ediff1d(xy_pos[:, 1])) > tile_sz / 2
        if False in change_y_coord:
            # sometimes get faulty first xy_pos
            # know that if there are more than one y coordinates, then the first
            # and second tile must have the same y coordinate.
            change_y_coord[0] = False
        ny = sum(change_y_coord) + 1
        nx = np.shape(xy_pos)[0] / ny
        if round(nx) != nx:
            raise ValueError('nx is not an integer')
        tilepos_yx_nd2[:, 0] = np.flip(np.arange(ny).repeat(nx))
        tilepos_yx_nd2[:, 1] = np.tile(np.concatenate((np.flip(np.arange(nx)), np.arange(nx))),
                                          np.ceil(ny / 2).astype(int))[:np.shape(xy_pos)[0]]
    t_tiff_index = get_tile_file_indices(tilepos_yx_nd2)
    tilepos_yx_tiff = tilepos_yx_nd2[np.argsort(t_tiff_index)]
    return tilepos_yx_nd2, tilepos_yx_tiff


def get_tile_file_indices(tilepos_yx_nd2: np.ndarray) -> np.ndarray:
    """
    Tile with nd2 fov index `i` is at yx position `tilepos_yx_nd2[i, :]`
    and tiff file has tile index given by `tile_file_index[i]`.

    Tile file indices are different because want `tile_file_index[0]`
    to refer to tile at yx position `[0, 0]`.

    Args:
        tilepos_yx_nd2: `int [n_tiles x 2]`.
            `tilepos_yx[i, 0]` is y index of tile with fov index `i` in nd2 file.
            `tilepos_yx[i, 1]` is x index of tile with fov index `i` in nd2 file.

    Returns:
        `int [n_tiles]`.
            `tile_file_index` such that tile with nd2 fov index `i` has tiff file index `tile_file_index[i]`.
    """
    ny, nx = tuple(np.max(tilepos_yx_nd2, 0) + 1)
    tile_file_index = np.ravel_multi_index(np.array([tilepos_yx_nd2[:, 0], tilepos_yx_nd2[:, 1]]), (ny, nx))
    return tile_file_index


def get_tile_name(tile_directory: str, file_base: List[str], r: int, t: int, c: Optional[int] = None) -> str:
    """
    Finds the full path to tile, `t`, of particular round, `r`, and channel, `c`, in `tile_directory`.

    Args:
        tile_directory: Path to folder where tiles tiff files saved.
        file_base: `str [n_rounds]`.
            `file_base[r]` is identifier for round `r`.
        r: Round of desired tiff image.
        t: Tile of desired tiff image.
        c: Channel of desired tiff image.

    Returns:
        Full path of tile tiff file.
    """
    if c is None:
        tile_name = os.path.join(tile_directory, '{}_t{}.tif'.format(file_base[r], t))
    else:
        tile_name = os.path.join(tile_directory, '{}_t{}c{}.tif'.format(file_base[r], t, c))
    return tile_name


def get_tile_file_names(tile_directory: str, file_base: List[str], tilepos_yx_nd2: np.ndarray, matlab_tile_names: bool,
                        n_channels: int = 0) -> np.ndarray:
    """
    Gets array of all tile file paths which will be saved in tile directory.

    Args:
        tile_directory: Path to folder where tiles tiff files saved.
        file_base: `str [n_rounds]`.
            `file_base[r]` is identifier for round `r`.
        tilepos_yx_nd2: `int [n_tiles x 2]`.
            `tilepos_yx_nd2[i, 0]` is y index of tile with fov index `i` in nd2 file.
            `tilepos_yx_nd2[i, 1]` is x index of tile with fov index `i` in nd2 file.
        matlab_tile_names: If `True`, tile files will have `t` and `c` index starting at `1` else will start at `0`.
        n_channels: Total number of imaging channels if using 3D.
            `0` if using 2D pipeline as all channels saved in same file.

    Returns:
        `object [n_tiles x n_rounds (x n_channels)]`.
        `tile_files` such that

        - If 2D so `n_channels = 0`, `tile_files[t, r]` is the full path to tiff file containing all channels of
            tile `t`, round `r`.
        - If 3D so `n_channels > 0`, `tile_files[t, r]` is the full path to tiff file containing all z-planes of
        tile `t`, round `r`, channel `c`.
    """
    t_tiff = get_tile_file_indices(tilepos_yx_nd2)
    n_tiles = np.shape(tilepos_yx_nd2)[0]
    n_rounds = len(file_base)
    index_shift = int(matlab_tile_names)
    if n_channels == 0:
        # 2D
        tile_files = np.zeros((n_tiles, n_rounds), dtype=object)
        for r in range(n_rounds):
            for t_nd2 in range(n_tiles):
                tile_files[t_tiff[t_nd2], r] = \
                    get_tile_name(tile_directory, file_base, r, t_tiff[t_nd2] + index_shift)
    else:
        # 3D
        tile_files = np.zeros((n_tiles, n_rounds, n_channels), dtype=object)
        for r in range(n_rounds):
            for t_nd2 in range(n_tiles):
                for c in range(n_channels):
                    tile_files[t_tiff[t_nd2], r, c] = \
                        get_tile_name(tile_directory, file_base, r, t_tiff[t_nd2] + index_shift,
                                      c + index_shift)
    return tile_files

# TODO: Make tile_pos work for non rectangular array of tiles in nd2 file