from typing import Tuple, Optional
import numpy as np
from ...setup import Notebook
from ...call_spots import dot_product_score
from ...call_spots import fit_background


def background_fitting(nb: Notebook, method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes background using parameters in config file. Then removes this from the `spot_colors`.
    Args:
        nb: Notebook containing call_spots page
        method: 'omp' or 'anchor', indicating which `spot_colors` to use.

    Returns:
        `spot_colors` - `float [n_spots x n_rounds_use x n_channels_use]`.
            `spot_color` after normalised by `color_norm_factor` but before background fit.
        `spot_colors_pb` - `float [n_spots x n_rounds_use x n_channels_use]`.
            `spot_color` after background removed.
        `background_var` - `float [n_spots x n_rounds_use x n_channels_use]`.
            inverse of the weighting used for dot product score calculation.
    """
    rc_ind = np.ix_(nb.basic_info.use_rounds, nb.basic_info.use_channels)
    if method.lower() == 'omp':
        spot_colors = np.moveaxis(np.moveaxis(nb.omp.colors, 0, -1)[rc_ind], -1, 0)
        config = nb.get_config()['omp']
    else:
        spot_colors = np.moveaxis(np.moveaxis(nb.ref_spots.colors, 0, -1)[rc_ind], -1, 0)
        config = nb.get_config()['call_spots']
    alpha = config['alpha']
    beta = config['beta']
    spot_colors = spot_colors / nb.call_spots.color_norm_factor[rc_ind]
    spot_colors_pb, background_coef, background_codes = \
        fit_background(spot_colors, 0)
    background_codes = background_codes.reshape(background_codes.shape[0], -1)
    background_var = background_coef ** 2 @ background_codes ** 2 * alpha + beta ** 2
    return spot_colors, spot_colors_pb, background_var


def get_dot_product_score(spot_colors: np.ndarray, bled_codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds dot product score for each `spot_color` given to the gene indicated by `spot_gene_no`.

    Args:
        spot_colors: `float [n_spots x n_rounds_use x n_channels_use]`.
            colors of spots to find score of.
        bled_codes: `float [n_genes x n_rounds_use x n_channels_use]`.
            colors of genes to find dot product with.
        spot_gene_no: `int [n_spots]`.
            Gene that each spot was assigned to. If None, will set `spot_gene_no[s]` to gene for which
            score was largest.

    Returns:
        `spot_score` - `float [n_spots]`.
            Dot product score for each spot.
        `spot_gene_no` - will be same as input if given, otherwise will be the best gene assigned.
    """
    n_spots = spot_colors.shape[0]
    n_genes = bled_codes.shape[0]
    gene_no, score = dot_product_score(spot_colors.reshape((n_spots, -1)), bled_codes.reshape((n_genes, -1)))[:2]

    return score, gene_no
