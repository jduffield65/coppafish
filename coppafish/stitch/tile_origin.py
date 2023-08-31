import numpy as np


def get_tile_origin(v_pairs: np.ndarray, v_shifts: np.ndarray, h_pairs: np.ndarray, h_shifts: np.ndarray,
                    n_tiles: int, home_tile: int) -> np.ndarray:
    """
    This finds the origin of each tile in a global coordinate system based on the shifts between overlapping tiles.
    The problem is over-determined, as there are more shifts than tiles. The solution is found by solving a least
    squares problem.

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

    # M * tile_offset = c
    pairs = {'v': v_pairs, 'h': h_pairs}
    shifts = {'v': v_shifts, 'h': h_shifts}
    M = np.zeros((n_tiles+1, n_tiles))
    expected_shift = np.zeros((n_tiles+1, 3))
    # Loop over vertical and horizontal directions
    for j in ['v', 'h']:
        # For each direction, we loop over all pairs of neighbouring tiles
        for i in range(pairs[j].shape[0]):
            # t1 and t2 are the indices of the two neighbouring tiles
            t1, t2 = pairs[j][i, 0], pairs[j][i, 1]
            # The expected shift of tile t is the sum of all its shifts to its neighbours
            expected_shift[t1, :] = expected_shift[t1, :] + shifts[j][i, :]
            expected_shift[t2, :] = expected_shift[t2, :] - shifts[j][i, :]
            # TODO: Find out what M is
            M[t1, t1] += 1
            M[t1, t2] -= 1
            M[t2, t2] += 1
            M[t2, t1] -= 1

    # Fix the coordinate of the home tile
    huge, tiny = 1e6, 1e-4  # for regularization
    M[n_tiles, home_tile] = 1
    expected_shift[n_tiles, :] = huge
    # Solve the least squares problem, with regularization
    tile_offset0 = np.linalg.lstsq(M + tiny * np.eye(n_tiles + 1, n_tiles), expected_shift, rcond=None)[0]
    # find tiles that are connected to the home tile
    aligned_ok = tile_offset0[:, 0] > huge / 2
    tile_offset1 = np.ones((n_tiles, 3)) * np.nan
    tile_offset1[aligned_ok] = tile_offset0[aligned_ok] - huge
    tile_origin = tile_offset1 - np.nanmin(tile_offset1, axis=0)

    return tile_origin
