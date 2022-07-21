import numpy as np
import os
from typing import Tuple, Optional, List


def get_tilepos(xy_pos: np.ndarray, tile_sz: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Using `xy_pos` from nd2 metadata, this obtains the yx position of each tile.
    I.e. how tiles are arranged with respect to each other.
    Note that this is indexed differently in nd2 file and npy files in the tile directory.

    Args:
        xy_pos: `float [n_tiles x 2]`.
            xy position of tiles in pixels. Obtained from nd2 metadata.
        tile_sz: xy dimension of tile in pixels.

    Returns:
        - `tilepos_yx_nd2` - `int [n_tiles x 2]`.
            `tilepos_yx_nd2[i]` is yx index of tile with fov index `i` in nd2 file.
            Index 0 refers to ```YX = [0, 0]```.
            Index 1 refers to ```YX = [0, 1] if MaxX > 0```.
        - `tilepos_yx_npy` - `int [n_tiles x 2]`.
            `tilepos_yx_npy[i, 0]` is yx index of tile with tile directory (npy files) index `i`.
            Index 0 refers to ```YX = [MaxY, MaxX]```.
            Index 1 refers to ```YX = [MaxY, MaxX - 1] if MaxX > 0```.
    """
    tilepos_yx_nd2 = np.zeros_like(xy_pos, dtype=int)
    tilepos_yx_npy = np.zeros_like(xy_pos, dtype=int)
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
        tilepos_yx_nd2[:, 0] = np.arange(ny).repeat(nx)
        tilepos_yx_nd2[:, 1] = np.tile(np.concatenate((np.arange(nx), np.flip(np.arange(nx)))),
                                       np.ceil(ny / 2).astype(int))[:np.shape(xy_pos)[0]]
        tilepos_yx_npy[:, 0] = np.flip(np.arange(ny).repeat(nx))
        tilepos_yx_npy[:, 1] = np.tile(np.concatenate((np.flip(np.arange(nx)), np.flip(np.arange(nx)))),
                                        np.ceil(ny / 2).astype(int))[:np.shape(xy_pos)[0]]
    return tilepos_yx_nd2, tilepos_yx_npy


def get_tile_name(tile_directory: str, file_base: List[str], r: int, t: int, c: Optional[int] = None) -> str:
    """
    Finds the full path to tile, `t`, of particular round, `r`, and channel, `c`, in `tile_directory`.

    Args:
        tile_directory: Path to folder where tiles npy files saved.
        file_base: `str [n_rounds]`.
            `file_base[r]` is identifier for round `r`.
        r: Round of desired npy image.
        t: Tile of desired npy image.
        c: Channel of desired npy image.

    Returns:
        Full path of tile npy file.
    """
    if c is None:
        tile_name = os.path.join(tile_directory, '{}_t{}.npy'.format(file_base[r], t))
    else:
        tile_name = os.path.join(tile_directory, '{}_t{}c{}.npy'.format(file_base[r], t, c))
    return tile_name


def get_tile_file_names(tile_directory: str, file_base: List[str], n_tiles: int, n_channels: int = 0) -> np.ndarray:
    """
    Gets array of all tile file paths which will be saved in tile directory.

    Args:
        tile_directory: Path to folder where tiles npy files saved.
        file_base: `str [n_rounds]`.
            `file_base[r]` is identifier for round `r`.
        n_tiles: Number of tiles in data set.
        n_channels: Total number of imaging channels if using 3D.
            `0` if using 2D pipeline as all channels saved in same file.

    Returns:
        `object [n_tiles x n_rounds (x n_channels)]`.
        `tile_files` such that

        - If 2D so `n_channels = 0`, `tile_files[t, r]` is the full path to npy file containing all channels of
            tile `t`, round `r`.
        - If 3D so `n_channels > 0`, `tile_files[t, r]` is the full path to npy file containing all z-planes of
        tile `t`, round `r`, channel `c`.
    """
    n_rounds = len(file_base)
    if n_channels == 0:
        # 2D
        tile_files = np.zeros((n_tiles, n_rounds), dtype=object)
        for r in range(n_rounds):
            for t in range(n_tiles):
                tile_files[t, r] = \
                    get_tile_name(tile_directory, file_base, r, t)
    else:
        # 3D
        tile_files = np.zeros((n_tiles, n_rounds, n_channels), dtype=object)
        for r in range(n_rounds):
            for t in range(n_tiles):
                for c in range(n_channels):
                    tile_files[t, r, c] = \
                        get_tile_name(tile_directory, file_base, r, t, c)
    return tile_files
# TODO: Make tile_pos work for non rectangular array of tiles in nd2 file
