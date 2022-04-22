import numpy as np
from .. import utils
from typing import Tuple


def fitting_standard_deviation(bled_codes: np.ndarray, coef: np.ndarray, alpha: float, beta: float = 1) -> np.ndarray:
    """
    Based on maximum likelihood estimation, this finds the standard deviation accounting for all genes fit in
    each round/channel. The more genes added, the greater the standard deviation so if the inverse is used as a
    weighting for omp fitting, the rounds/channels which already have genes in will contribute less.

    Args:
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        coef: `float [n_pixels x n_genes]`.
            Coefficient of each `bled_code` for each pixel found on the previous OMP iteration.
        alpha: By how much to increase variance as genes added.
        beta: The variance with no genes added (`coef=0`) is `beta**2`.

    Returns:
        `float [n_pixels x n_rounds x n_channels]`
            Standard deviation of each pixel in each round/channel based on genes fit.
    """
    n_genes, n_rounds, n_channels = bled_codes.shape
    n_pixels = coef.shape[0]

    if not utils.errors.check_shape(coef, [n_pixels, n_genes]):
        raise utils.errors.ShapeError('coef', coef.shape, (n_pixels, n_genes))

    var = np.ones((n_pixels, n_rounds, n_channels)) * beta ** 2
    for g in range(n_genes):
        var = var + alpha * np.expand_dims(coef[:, g]**2, (1, 2)) * np.expand_dims(bled_codes[g]**2, 0)

    sigma = np.sqrt(var)
    return sigma


def fit_coefs(bled_codes: np.ndarray, pixel_colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds x n_channels) x n_pixels]`.
            Flattened then transposed pixel colors which usually has the shape `[n_genes x n_rounds x n_channels]`.

    Returns:
        - residual - `float [(n_rounds x n_channels) x n_pixels]`.
            Residual pixel_colors are removing bled_codes with coefficients specified by coef.
        - coef - `float [n_genes x n_pixels]`.
            coefficient found through least squares fitting for each gene.

    """
    if bled_codes.shape[1] == 1:
        # can do many pixels at once if just one gene and is quicker this way.
        coefs = np.sum(bled_codes * pixel_colors, axis=0) / np.sum(bled_codes ** 2)
        residual = pixel_colors - coefs * bled_codes
        coefs = coefs.reshape(1, -1)
    else:
        # TODO: maybe iterate over all unique combinations of added genes instead of over all spots.
        #  Would not work if do weighted coef fitting though.
        coefs = np.linalg.lstsq(bled_codes, pixel_colors, rcond=None)[0]
        residual = pixel_colors - bled_codes @ coefs
    return residual, coefs
