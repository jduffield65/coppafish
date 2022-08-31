import numpy as np


def get_tile_origin(v_pairs: np.ndarray, v_shifts: np.ndarray, h_pairs: np.ndarray, h_shifts: np.ndarray,
                    n_tiles: int, home_tile: int) -> np.ndarray:
    """
    This finds the origin of each tile in a global coordinate system based on the shifts between overlapping tiles.

    Args:
        v_pairs: `int [n_v_pairs x 2]`.
            `v_pairs[i,1]` is the tile index of the tile to the south of `v_pairs[i,0]`.
        v_shifts: `int [n_v_pairs x 3]`.
            `v_shifts[i, :]` is the yxz shift from `v_pairs[i,0]` to `v_pairs[i,1]`.
            `v_shifts[:, 0]` should all be negative.
        h_pairs: `int [n_h_pairs x 2]`.
            `h_pairs[i,1]` is the tile index of the tile to the west of `h_pairs[i,0]`.
        h_shifts: `int [n_h_pairs x 3]`.
            `h_shifts[i, :]` is the yxz shift from `h_pairs[i,0]` to `h_pairs[i,1]`.
            `h_shifts[:, 1]` should all be negative.
        n_tiles: Number of tiles (including those not used) in data set.
        home_tile: Index of tile that is anchored to a fixed coordinate when finding tile origins.
            It should be the tile nearest to the centre.

    Returns:
        `float [n_tiles x 3]`. yxz origin of each tile.
    """

    # solve a set of linear equations for each shift,
    # This will be of the form M*x = c, where x and c are both of length n_tiles.
    # The t'th row is the equation for tile t. c has columns for y, x and z coordinates
    pairs = {'v': v_pairs, 'h': h_pairs}
    shifts = {'v': v_shifts, 'h': h_shifts}
    M = np.zeros((n_tiles+1, n_tiles))
    c = np.zeros((n_tiles+1, 3))
    for j in ['v', 'h']:
        for i in range(pairs[j].shape[0]):
            t1 = pairs[j][i, 0]
            t2 = pairs[j][i, 1]
            M[t1, t1] = M[t1, t1] + 1
            M[t1, t2] = M[t1, t2] - 1
            c[t1, :] = c[t1, :] + shifts[j][i, :]   # this is -shifts in MATLAB, but t1, t2 flipped in python
            M[t2, t2] = M[t2, t2] + 1
            M[t2, t1] = M[t2, t1] - 1
            c[t2, :] = c[t2, :] - shifts[j][i, :]   # this is +shifts in MATLAB, but t1, t2 flipped in python

    # now we want to anchor one of the tiles to a fixed coordinate. We do this
    # for a home tile in the middle, because it is going to be connected; and we set
    # its coordinate to a large value, so any non-connected ones can be detected.
    # (BTW this is why spectral clustering works!!)
    huge = 1e6
    M[n_tiles, home_tile] = 1
    c[n_tiles, :] = huge

    tiny = 1e-4  # for regularization
    tile_offset0 = np.linalg.lstsq(M + tiny * np.eye(n_tiles + 1, n_tiles), c, rcond=None)[0]
    # find tiles that are connected to the home tile
    aligned_ok = tile_offset0[:, 0] > huge/2
    tile_offset1 = np.ones((n_tiles, 3)) * np.nan
    tile_offset1[aligned_ok] = tile_offset0[aligned_ok] - huge
    tile_origin = tile_offset1 - np.nanmin(tile_offset1, axis=0)

    return tile_origin
