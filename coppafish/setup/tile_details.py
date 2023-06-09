import numpy as np
import os
from typing import Tuple, Optional, List


def get_tilepos(xy_pos: np.ndarray, tile_sz: int, expected_overlap: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Using `xy_pos` from nd2 metadata, this obtains the yx position of each tile. We label the tiles differently in the
    nd2 and npy formats. In general, there are 2 differences in npy and nd2 tile format:
    1. The same tile in npy and nd2 has a different npy and nd2 index, in nd2 it is the exact same order as the xy pos
    in the metadata, in npy it is ordered from top right to bottom left, moving left along rows and then down rows.
    2. The same tile in npy and nd2 has a different coordinate convention. In npy, a tile with coord [0,0] represents
    bottom left (ie min y and min x) while in nd2, a tile with coord [0,0] represents top right (ie max y and max x).

    Args:
        xy_pos: `float [n_tiles x 2]`.
            xy position of tiles in pixels. Obtained from nd2 metadata. `xy_pos[t,:]` is xy position of tile `t` where
            `t` is the nd2 tile index.
        tile_sz: xy dimension of tile in pixels.
        expected_overlap: expected overlap between tiles as a fraction of tile_sz.

    Returns:

    """
    # since xy_pos is in higher resolution, we need to divide by the expected difference between tiles. May not be
    # exactly the same as tile_sz due to rounding errors, so round to nearest integer
    delta = tile_sz * (1 - expected_overlap)
    xy_pos = np.round(xy_pos / delta).astype(int)

    # get the max x and y values (xy_pos was normalised to make min x and y = 0)
    max_x = np.max(xy_pos[:, 0])
    max_y = np.max(xy_pos[:, 1])

    # Now we need to loop through each tile and assign it a yx index for the nd2 and npy formats
    tilepos_yx_nd2 = np.zeros((xy_pos.shape[0], 2), dtype=int)
    tilepos_yx_npy = np.zeros((xy_pos.shape[0], 2), dtype=int)
    for t in range(xy_pos.shape[0]):
        # nd2 format
        tilepos_yx_nd2[t, 1] = max_x - xy_pos[t, 0]
        tilepos_yx_nd2[t, 0] = max_y - xy_pos[t, 1]

        # npy format
        tilepos_yx_npy[t, 1] = xy_pos[t, 0]
        tilepos_yx_npy[t, 0] = xy_pos[t, 1]

    # Finally, we need to sort tilepos_npy. Want to sort by y first, (ascending), then for each y, sort by x
    # (descending). We need to use merge sort. In practice, need to actually sort by x first then y
    tilepos_yx_npy = tilepos_yx_npy[tilepos_yx_npy[:, 1].argsort()]
    tilepos_yx_npy = tilepos_yx_npy[tilepos_yx_npy[:, 0].argsort(kind='mergesort')]

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


def get_tile_file_names(tile_directory: str, file_base: List[str],
                        n_tiles: int, n_channels: int = 0, jobs: bool = False) -> np.ndarray:
    """
    Gets array of all tile file paths which will be saved in tile directory.

    Args:
        tile_directory: Path to folder where tiles npy files saved.
        file_base: `str [n_rounds]`.
            `file_base[r]` is identifier for round `r`.
        n_tiles: Number of tiles in data set.
        n_channels: Total number of imaging channels if using 3D.
            `0` if using 2D pipeline as all channels saved in same file.
        jobs: Set True if file were acquired using JOBs (i.e. tiles are split by laser)

    Returns:
        `object [n_tiles x n_rounds (x n_channels)]`.
        `tile_files` such that

        - If 2D so `n_channels = 0`, `tile_files[t, r]` is the full path to npy file containing all channels of
            tile `t`, round `r`.
        - If 3D so `n_channels > 0`, `tile_files[t, r]` is the full path to npy file containing all z-planes of
        tile `t`, round `r`, channel `c`.
    """
    if not jobs:
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
    else:
        n_lasers = 7  # TODO: should have the option to pass n_lasers as an argument for better generalisation
        n_rounds = int(len(file_base) / n_tiles / n_lasers)
        tile_files = np.zeros((n_tiles, n_rounds, n_channels), dtype=object)

        for r in range(n_rounds):
            round_files = file_base[r*n_tiles*n_lasers:(r+1)*n_tiles*n_lasers]

            for t in range(n_tiles):
                raw_tile_files = round_files[t * n_lasers: (t + 1) * n_lasers]

                for c in range(n_channels):
                    f_index = int(np.floor(c/4))
                    t_name = os.path.join(tile_directory, '{}_t{}c{}.npy'.format(raw_tile_files[f_index], t, c))
                    tile_files[t, r, c] = t_name

    return tile_files
