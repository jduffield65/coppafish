from scipy.spatial import KDTree
import numpy as np
from typing import List, Tuple

from coppafish import utils


def get_non_duplicate(tile_origin: np.ndarray, use_tiles: List, tile_centre: np.ndarray,
                      spot_local_yxz: np.ndarray, spot_tile: np.ndarray) -> np.ndarray:
    """
    Find duplicate spots as those detected on a tile which is not tile centre they are closest to.

    Args:
        tile_origin: `float [n_tiles x 3]`.
            `tile_origin[t,:]` is the bottom left yxz coordinate of tile `t`.
            yx coordinates in `yx_pixels` and z coordinate in `z_pixels`.
            This is saved in the `stitch` notebook page i.e. `nb.stitch.tile_origin`.
        use_tiles: ```int [n_use_tiles]```.
            Tiles used in the experiment.
        tile_centre: ```float [3]```
            ```tile_centre[:2]``` are yx coordinates in ```yx_pixels``` of the centre of the tile that spots in
            ```yxz``` were found on.
            ```tile_centre[2]``` is the z coordinate in ```z_pixels``` of the centre of the tile.
            E.g. for tile of ```yxz``` dimensions ```[2048, 2048, 51]```, ```tile_centre = [1023.5, 1023.5, 25]```
            Each entry in ```tile_centre``` must be an integer multiple of ```0.5```.
        spot_local_yxz: ```int [n_spots x 3]```.
            Coordinates of a spot s on tile spot_tile[s].
            ```yxz[s, :2]``` are the yx coordinates in ```yx_pixels``` for spot ```s```.
            ```yxz[s, 2]``` is the z coordinate in ```z_pixels``` for spot ```s```.
        spot_tile: ```int [n_spots]```.
            Tile each spot was found on.

    Returns:
        ```bool [n_spots]```.
            Whether spot_tile[s] is the tile that spot_global_yxz[s] is closest to.
    """
    tile_centres = tile_origin[use_tiles] + tile_centre
    # Do not_duplicate search in 2D as overlap is only 2D
    tree_tiles = KDTree(tile_centres[:, :2])
    if np.isnan(tile_origin[np.unique(spot_tile)]).any():
        nan_tiles = np.unique(spot_tile)[np.unique(np.where(np.isnan(tile_origin[np.unique(spot_tile)]))[0])]
        raise ValueError(f"tile_origin for tiles\n{nan_tiles}\ncontains nan values but some spot_tile "
                         f"also contains these tiles. Maybe remove these from use_tiles to continue.\n"
                         f"Also, consider coppafish.plot.n_spots_grid to check if these tiles have few spots.")
    spot_global_yxz = spot_local_yxz + tile_origin[spot_tile]
    all_nearest_tile_ind = tree_tiles.query(spot_global_yxz[:, :2])[1]
    not_duplicate = np.asarray(use_tiles)[all_nearest_tile_ind.flatten()] == spot_tile
    return not_duplicate


def get_bled_codes(gene_codes: np.ndarray, bleed_matrix: np.ndarray, gene_efficiency: np.ndarray) -> np.ndarray:
    """
    This gets ```bled_codes``` such that the spot_color of a gene ```g``` in round ```r``` is expected to be a constant
    multiple of ```bled_codes[g, r]```.
    This function should be run with full bleed_matrix with any rounds/channels/dyes outside those using set to nan.
    Otherwise, will get confusion with dye indices in `gene_codes` being outside size of `bleed_matrix`.

    Args:
        gene_codes: ```int [n_genes x n_rounds]```.
            ```gene_codes[g, r]``` indicates the dye that should be present for gene ```g``` in round ```r```.
        bleed_matrix: ```float [n_channels x n_dyes]```.
            Expected intensity of dye ```d``` is a constant multiple of ```bleed_matrix[:, d]```.
        gene_efficiency: ```float [n_genes, n_rounds]```.
            Efficiency of gene ```g``` in round ```r``` is ```gene_efficiency[g, r]```.

    Returns:
        ```float [n_genes x n_rounds x n_channels]```.
            ```bled_codes``` such that ```spot_color``` of a gene ```g```
            in round ```r``` is expected to be a constant multiple of ```bled_codes[g, r]```. bled_codes[g] will 
            all have a norm of one.
    """
    n_genes, n_rounds = gene_codes.shape[0], gene_codes.shape[1]
    n_channels, n_dyes = bleed_matrix.shape
    if not utils.errors.check_shape(gene_codes, [n_genes, n_rounds]):
        raise utils.errors.ShapeError('gene_codes', gene_codes.shape, (n_genes, n_rounds))
    if gene_codes.max() >= n_dyes:
        ind_1, ind_2 = np.where(gene_codes == gene_codes.max())
        raise ValueError(f"gene_code for gene {ind_1[0]}, round {ind_2[0]} has a dye with index {gene_codes.max()}"
                         f" but there are only {n_dyes} dyes.")
    if gene_codes.min() < 0:
        ind_1, ind_2 = np.where(gene_codes == gene_codes.min())
        raise ValueError(f"gene_code for gene {ind_1[0]}, round {ind_2[0]} has a dye with a negative index:"
                         f" {gene_codes.min()}")

    bled_codes = np.zeros((n_genes, n_rounds, n_channels))
    for g in range(n_genes):
        for r in range(n_rounds):
            for c in range(n_channels):
                bled_codes[g, r, c] = gene_efficiency[g, r] * bleed_matrix[c, gene_codes[g, r]]

    # Give all bled codes an L2 norm of 1
    norm_factor = np.linalg.norm(bled_codes, axis=(1, 2))
    bled_codes = bled_codes / norm_factor[:, None, None]
    return bled_codes


def compute_gene_efficiency(spot_colours: np.ndarray, bled_codes: np.ndarray, gene_no: np.ndarray,
                            gene_score: np.ndarray, gene_codes: np.ndarray, intensity: np.ndarray,
                            spot_number_threshold: int = 25, score_threshold: float = 0.8,
                            intensity_threshold: float = 0) \
        -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Compute gene efficiency and gene coefficients from spot colours and bleed matrix.

    Args:
        spot_colours: `float [n_spots x n_rounds x n_channels]`.
            Spot colours normalised to equalise intensities between channels (and rounds). (BG Removed)
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
        gene_no: `int [n_spots]`. Gene number for each spot.
        gene_score: `float [n_spots]`. Score for each spot.
        gene_codes: `int [n_genes x n_rounds]`.
            Gene codes for each gene.
        intensity: `float [n_spots]`.
            Intensity of each spot.
        spot_number_threshold: `int`.
            Minimum number of spots required to compute gene efficiency.
        score_threshold: `float`.
            Minimum score required to compute gene efficiency.
        intensity_threshold: `float`.
            Minimum intensity required to compute gene efficiency.
    
    #TODO: Add documentation for the returns
    """
    n_spots, n_rounds, n_channels = spot_colours.shape
    n_genes = gene_codes.shape[0]
    gene_efficiency = np.ones([n_genes, n_rounds])
    dye_efficiency = np.ones([n_genes, n_rounds, 0]).tolist()
    use_ge = np.zeros(n_spots, dtype=bool)

    # Compute gene efficiency for each gene and round.
    for g in range(n_genes):
        gene_g_mask = (gene_no == g) * (gene_score > score_threshold) * (intensity > intensity_threshold)
        # Skip gene if not enough spots.
        if np.sum(gene_g_mask) < spot_number_threshold:
            continue
        use_ge += gene_g_mask
        gene_g_spot_colours = spot_colours[gene_g_mask]
        dye_efficiency_g = np.ones([gene_g_spot_colours.shape[0], n_rounds])
        for r in range(n_rounds):
            # Compute gene efficiency for each round. This is just the best scaling factor to match the mean/median
            # spot colour to the expected spot colour.
            expected_spot_colour = bled_codes[g, r]
            observed_spot_colour = gene_g_spot_colours[:, r]
            for s in range(observed_spot_colour.shape[0]):
                a = observed_spot_colour[s]
                b = expected_spot_colour
                # Compute scaling factor.
                dye_efficiency_g[s, r] = np.dot(a, b) / np.dot(b, b)
            # Compute gene efficiency as the median dye efficiency across spots.
            gene_efficiency[g, r] = np.median(dye_efficiency_g[:, r])
            dye_efficiency[g][r] = dye_efficiency_g[:, r]

    # Set negative values to 0.
    gene_efficiency[gene_efficiency < 0] = 0

    return gene_efficiency, use_ge, dye_efficiency


def matrix_match(Y: np.ndarray, X: np.ndarray, u: np.ndarray, v: np.ndarray, alpha: float = 0, beta: float = 0,
                 n_iters: int = 50) -> [np.ndarray, np.ndarray]:
    """
    This function solves the following problem: given two matrices Y and X (n rows and m columns each), find two
    vectors u and v (of length n and m respectively) such that Y_ij ~ u_i * X_ij * v_j for all i and j. This is
    achieved approximately by minimising the squared error between Y and u * X * v.
    Args:
        Y: target matrix (n rows, m columns) to be matched
        X: input matrix (n rows, m columns)
        u: initial guess for u (n entries)
        v: initial guess for v (m entries)
        alpha: weight of the L2 regularisation term on u
        beta: weight of the L2 regularisation term on v
        n_iters: number of iterations to run the algorithm for

    Returns:
        u, v: solutions to the problem. These are vectors of length n and m respectively. u will be L2 normalised,

    """
    u_init, v_init = u.copy(), v.copy()
    n, m = Y.shape
    eta, theta = np.zeros(n), np.zeros(m)
    W = Y * X
    for iter in range(n_iters):
        for i in range(n):
            eta[i] = 1 / (np.sum((X[i] * v) ** 2) + alpha)
        for j in range(m):
            theta[j] = 1 / (np.sum((X[:, j] * u) ** 2) + beta)
        u = (W @ v + alpha * u_init) * eta
        u /= np.linalg.norm(u)
        v = (W.T @ u + beta * v_init) * theta
    return u, v


def matrix_match_exact(Y: np.ndarray, X: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    This function solves the following problem: given two matrices Y and X (n rows and m columns each), find two
    vectors u and v (of length n and m respectively) such that Y_ij ~ u_i * X_ij * v_j for all i and j. We do this by
    solving the problem u_i * Y_ij ~ X_ij * v_j for all i and j, and then inverting the u_i. The problem is constrained
    so that u_i and v_j are nonzero and u has unit L2 norm. When this is the case it can be shown that the solution is
    the eigenvector of W @ W.T with the largest eigenvalue, where W = Y * X.

    Args:
        Y: target matrix (n rows, m columns) to be matched
        X: input matrix (n rows, m columns)

    Returns:
        u, v: solutions to the problem. These are vectors of length n and m respectively. u will be L2 normalised,
        v will not.
    """
    W = Y * X
    u_evals, u_evecs = np.linalg.eig(W @ W.T)
    v_evals, v_evecs = np.linalg.eig(W.T @ W)
    u_evals, u_evecs = np.real(u_evals), np.real(u_evecs)
    v_evals, v_evecs = np.real(v_evals), np.real(v_evecs)
    u = u_evecs[:, np.argmax(u_evals)]
    v = v_evecs[:, np.argmax(v_evals)]
    u /= np.linalg.norm(u)
    B = u @ W @ v
    C = np.sum((X * X) @ (v * v))
    v = v * B / C
    u = 1 / u

    return u, v


def matrix_match_with_prior(Y: np.ndarray, X: np.ndarray, u_init: np.ndarray, v_init: np.ndarray, alpha: float = 0,
                            beta: float = 0) -> [np.ndarray, np.ndarray]:
    """
    This function solves the following problem: given two matrices Y and X (n rows and m columns each), find two
    vectors u and v (of length n and m respectively) such that Y_ij ~ u_i * X_ij * v_j for all i and j. We do this by
    solving the problem u_i * Y_ij ~ X_ij * v_j for all i and j, and then inverting the u_i. The problem is constrained
    so that u_i is close to u_init_i and v_j is close to v_init_j. The problem is also constrained so that u_i has unit
    L2 norm. When this is the case, the solutions can be written in closed form.
    Args:
        Y: target matrix (n rows, m columns) to be matched
        X: input matrix (n rows, m columns)
        u_init: initial guess for u (n entries)
        v_init: initial guess for v (m entries)
        alpha: weight of the L2 regularisation term on u
        beta: weight of the L2 regularisation term on v

    Returns:
        u, v: solutions to the problem. These are vectors of length n and m respectively. u will be L2 normalised,
    """
    n, m = Y.shape
    W = Y * X
    eta = 1 / ((1+alpha)*(1+beta))
    u_operator = np.linalg.inv(np.eye(n) - eta * W @ W.T)
    v_operator = np.linalg.inv(np.eye(m) - eta * W.T @ W)
    u = u_operator @ (alpha * u_init + beta * W @ v_init)
    v = v_operator @ (beta * v_init + alpha * W.T @ u)
    u /= np.linalg.norm(u)
    u = 1 / u
    return u, v
