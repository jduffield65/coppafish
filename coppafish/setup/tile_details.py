import numpy as np
import os
from typing import Tuple, Optional, List


def get_tilepos(xy_pos: np.ndarray, tile_sz: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Using `xy_pos` from nd2 metadata, this obtains the yx position of each tile.
    I.e. how tiles are arranged with respect to each other. So the output is an n_tiles by 2 matrix where each row
    is the yx index of that tile.
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
    # NOTE: There are 2 differences in npy and nd2 tile format:
    # 1. Tiles in ND2 are ordered according to a snake format where we start at top right, move left across cols, move
    # down one row, move right across all cols, move down one row, etc. Whereas tiles in npy format start at top right
    # move left across all cols, then move to the rightmost col one row down and continue
    # 2. The same tile in npy and nd2 has a different coordinate convention. In nd2, a tile with coord [0,0] represents
    # top right whereas in npy [0,0] represents bottom left.

    n_tiles = xy_pos.shape[0]
    tilepos_yx_nd2 = []
    tilepos_yx_npy = []

    # This should get rid of some rounding errors
    xy_pos = xy_pos.astype(int)

    # Next we will try to split tiles up into rows and columns.
    y_coord = list(np.unique(xy_pos[:, 1]))
    y_coord.sort(reverse=True)
    x_coord = list(np.unique(xy_pos[:, 0]))
    x_coord.sort(reverse=True)

    # Refine these to get rid of any coords that correspond to same row/column. We take 2 x coords to correspond to
    # same col if they are within 10% of a tile width and same with y
    y_coord = [y_coord[i] for i in range(len(y_coord)-1) if y_coord[i] - y_coord[i+1] > 0.1 * tile_sz] + [y_coord[-1]]
    x_coord = [x_coord[i] for i in range(len(x_coord) - 1) if x_coord[i] - x_coord[i + 1] > 0.1 * tile_sz] + [x_coord[-1]]

    # Now update xy_pos and replace the values that we've got rid of representing a tile with the new values
    for t in range(n_tiles):
        if xy_pos[t, 0] not in x_coord:
            # Find the closest thing to this
            for x in x_coord:
                if abs(x - xy_pos[t, 0]) < 0.1 * tile_sz:
                    x_representative = x
                    break
            xy_pos[t, 0] = x_representative
        if xy_pos[t, 1] not in y_coord:
            # Find the closest thing to this
            for y in y_coord:
                if abs(y - xy_pos[t, 1]) < 0.1 * tile_sz:
                    y_representative = y
                    break
            xy_pos[t, 1] = y_representative

    # The next arrays will be useful for indexing the nd2s and npys as these have different grid naming conventions
    # ND2
    y_coord_descend, x_coord_descend = y_coord.copy(), x_coord.copy()
    y_coord_descend.sort(reverse=True)
    x_coord_descend.sort(reverse=True)
    # NPY
    y_coord_ascend, x_coord_ascend = y_coord.copy(), x_coord.copy()
    y_coord_ascend.sort(reverse=False)
    x_coord_ascend.sort(reverse=False)

    # Do ND2 first. With old get metadata function, tilepos[0] referred to nd2 index 0, tilepos[1] to index 1, etc.
    # This is no longer the case.
    # Now need to loop through all indices in the snake order that ND2 does and set these.
    x_reverse = True
    for y in y_coord:
        x_coord.sort(reverse=x_reverse)
        for x in x_coord:
            # check if this xy coord present in the xy coords listed
            if np.any(np.all((xy_pos == np.array([x, y])), axis=1)):
                tilepos_yx_nd2.append([y_coord_descend.index(y), x_coord_descend.index(x)])
        # Alternate the x_reverse
        x_reverse = not x_reverse

    # Convert to ndarray
    tilepos_yx_nd2 = np.array(tilepos_yx_nd2)

    # Now begin for the npy's
    x_reverse = True
    x_coord.sort(reverse=x_reverse)
    for y in y_coord:
        for x in x_coord:
            # Next condition checks if this coord is present in the xy coords
            if np.any(np.all((xy_pos == np.array([x, y])), axis=1)):
                tilepos_yx_npy.append([y_coord_ascend.index(y), x_coord_ascend.index(x)])

    # Convert to ndarray
    tilepos_yx_npy = np.array(tilepos_yx_npy)

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
