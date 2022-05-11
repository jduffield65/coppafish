import numpy as np
from .. import utils
from ..call_spots.base import fit_background, dot_product_score
from typing import Tuple, Optional, Union, List
from tqdm import tqdm


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
    n_genes = bled_codes.shape[0]
    n_pixels = coef.shape[0]
    if not utils.errors.check_shape(coef, [n_pixels, n_genes]):
        raise utils.errors.ShapeError('coef', coef.shape, (n_pixels, n_genes))

    var = np.moveaxis(coef**2 @ np.moveaxis(bled_codes**2, 0, 1), 1, 0) * alpha + beta ** 2

    # # Old method - much slower
    # n_genes, n_rounds, n_channels = bled_codes.shape
    # var = np.ones((n_pixels, n_rounds, n_channels)) * beta ** 2
    # for g in range(n_genes):
    #     var = var + alpha * np.expand_dims(coef[:, g] ** 2, (1, 2)) * np.expand_dims(bled_codes[g] ** 2, 0)

    return np.sqrt(var)


def fit_coefs(bled_codes: np.ndarray, pixel_colors: np.ndarray, weight: Optional[np.ndarray] = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    This finds the least squared solution for how the `n_genes` `bled_codes` can best explain each `pixel_color`.
    Can also find weighted least squared solution if `weight` provided.

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds x n_channels) x n_pixels]`.
            Flattened then transposed pixel colors which usually has the shape `[n_genes x n_rounds x n_channels]`.
        weight: `float [(n_rounds x n_channels) x 1]`.
            Weight to be applied to each data value when computing coefficient of each `bled_code` for each pixel.

    Returns:
        - residual - `float [(n_rounds x n_channels) x n_pixels]`.
            Residual pixel_colors are removing bled_codes with coefficients specified by coef.
        - coef - `float [n_pixels x n_genes]`.
            coefficient found through least squares fitting for each gene.

    """
    if weight is not None:
        pixel_colors = pixel_colors * weight
        bled_codes = bled_codes * weight
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
    if weight is not None:
        residual = residual / weight
    return residual, coefs.transpose()


def get_best_gene(residual_pixel_colors: np.ndarray, bled_codes: np.ndarray, coefs: np.ndarray, norm_shift: float,
                  score_thresh: float, alpha: float, beta: float,
                  ignore_genes: Optional[Union[np.ndarray, List]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the `best_gene` to add next to each pixel based on the dot product score with each `bled_code`.


    !!! note
        `best_gene` will be set to -1 if dot product is less than `score_thresh` or if the `best_gene` has already
        been added to the pixel or if best_gene is in ignore_genes.

    Args:
        residual_pixel_colors: `float [n_pixels x n_rounds x n_channels]`.
            Residual pixel colors from previous iteration of omp.
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        coefs: `float [n_pixels x n_genes]`.
            `coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm on previous iteration.
             Most are zero.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added in the current iteration.
        alpha: Used for `fitting_standard_deviation`, by how much to increase variance as genes added.
        beta: Used for `fitting_standard_deviation`, the variance with no genes added (`coef=0`) is `beta**2`.
        ignore_genes: `int [n_genes_ignore]`.
            Genes indices which if they are the best gene, best_gene will be set to -1.

    Returns:
        - best_gene - `int [n_pixels]`.
            `best_gene[s]` is the best gene to add to pixel `s` next. It is -1 if no more genes should be added.
        - sigma - `float [n_pixels x n_rounds x n_channels]`.
            Standard deviation of each pixel in each round/channel based on genes fit on previous iteration.

    """
    if ignore_genes is None:
        ignore_genes = []
    sigma = fitting_standard_deviation(bled_codes, coefs, alpha, beta)
    all_scores = dot_product_score(residual_pixel_colors, bled_codes, norm_shift, 1 / sigma)
    best_gene = np.argmax(np.abs(all_scores), 1)
    all_scores[coefs != 0] = 0  # best gene cannot be a gene which has already been added.
    for g in ignore_genes:
        all_scores[:, g] = 0  # best gene cannot be a gene in ignore_genes.
    best_score = all_scores[np.arange(all_scores.shape[0]), best_gene]
    best_gene[np.abs(best_score) <= score_thresh] = -1
    return best_gene, sigma


def get_all_coefs(pixel_colors: np.ndarray, bled_codes: np.ndarray, background_shift: float,
                  dp_shift: float, dp_thresh: float, alpha: float, beta: float, max_genes: int,
                  weight_coef_fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    This performs omp on every pixel, the stopping criterion is that the dot_product_score
    when selecting the next gene to add exceeds dp_thresh or the number of genes added to the pixel exceeds max_genes.

    !!! note
        Background vectors are fitted first and then not updated again.

    Args:
        pixel_colors: `float [n_pixels x n_rounds x n_channels]`.
            Pixel colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        background_shift: When fitting background,
            this is applied to weighting of each background vector to limit boost of weak pixels.
        dp_shift: When finding `dot_product_score` between residual `pixel_colors` and `bled_codes`,
            this is applied to normalisation of `pixel_colors` to limit boost of weak pixels.
        dp_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added at each iteration.
        alpha: Used for `fitting_standard_deviation`, by how much to increase variance as genes added.
        beta: Used for `fitting_standard_deviation`, the variance with no genes added (`coef=0`) is `beta**2`.
        max_genes: Maximum number of genes that can be added to a pixel i.e. number of iterations of OMP.
        weight_coef_fit: If False, coefs are found through normal least squares fitting.
            If True, coefs are found through weighted least squares fitting using 1/sigma as the weight factor.

    Returns:
        - gene_coefs - `float [n_pixels x n_genes]`.
            `gene_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm. Most are zero.
        - background_coefs - `float [n_pixels x n_channels]`.
            coefficient value for each background vector found for each pixel.
    """
    diff_to_int = np.round(pixel_colors).astype(int) - pixel_colors
    if np.abs(diff_to_int).max() == 0:
        raise ValueError("spot_intensities should be found using normalised spot_colors. "
                         "\nBut all values in spot_colors given are integers indicating they are the raw intensities.")
    del diff_to_int
    n_genes, n_rounds, n_channels = bled_codes.shape
    n_pixels = pixel_colors.shape[0]
    if not utils.errors.check_shape(pixel_colors, [n_pixels, n_rounds, n_channels]):
        raise utils.errors.ShapeError('pixel_colors', pixel_colors.shape, (n_pixels, n_rounds, n_channels))
    no_verbose = n_pixels < 1000  # show progress bar with more than 1000 pixels.

    # Fit background and override initial pixel_colors
    all_coefs = np.zeros((n_pixels, n_genes + n_channels))  # coefs of all genes and background
    pixel_colors, all_coefs[:, -n_channels:], background_codes = fit_background(pixel_colors, background_shift)
    background_genes = np.arange(n_genes, n_genes + n_channels)

    # colors and codes for get_best_gene function
    # Includes background as if background is the best gene, iteration ends.
    # uses residual color as used to find next gene to add.
    all_codes = np.concatenate((bled_codes, background_codes))
    residual_pixel_colors = pixel_colors.copy()

    # colors and codes for fit_coefs function (No background as this is not updated again).
    # always uses post background color as coefficients for all genes re-estimated at each iteration.
    pixel_colors = pixel_colors.reshape((n_pixels, -1)).transpose()
    bled_codes = bled_codes.reshape((n_genes, -1)).transpose()

    added_genes = np.ones((n_pixels, max_genes), dtype=int) * -1
    sigma = np.zeros((n_pixels, n_rounds, n_channels))
    continue_pixels = np.arange(n_pixels)
    for i in range(max_genes):
        # only continue with pixels for which dot product score exceeds threshold
        added_genes[continue_pixels, i], sigma[continue_pixels] = get_best_gene(residual_pixel_colors[continue_pixels],
                                                                                all_codes, all_coefs[continue_pixels],
                                                                                dp_shift, dp_thresh, alpha, beta,
                                                                                background_genes)
        residual_pixel_colors = residual_pixel_colors.reshape((n_pixels, -1)).transpose()
        continue_pixels = added_genes[:, i] >= 0
        n_continue = sum(continue_pixels)
        if n_continue == 0:
            break
        with tqdm(total=n_continue, disable=no_verbose) as pbar:
            pbar.set_postfix({'iter': i, 'n_pixels': n_continue})
            if i == 0:
                # When adding only 1 gene, can do many pixels at once if neglect weighting.
                # Neglecting weighting seems reasonable as small effect with just background.
                for g in range(n_genes):
                    use = added_genes[:, i] == g
                    residual_pixel_colors[:, use], all_coefs[use, g:g + 1] = fit_coefs(bled_codes[:, g:g + 1],
                                                                                       pixel_colors[:, use])
                    pbar.update(np.sum(use))
            else:
                if weight_coef_fit:
                    weight = 1 / sigma.reshape((n_pixels, -1)).transpose()
                for s in np.where(continue_pixels)[0]:
                    # s:s+1 is so shape is correct for fit_coefs function.
                    if weight_coef_fit:
                        residual_pixel_colors[:, s:s + 1], all_coefs[s, added_genes[s, :i + 1]] = \
                            fit_coefs(bled_codes[:, added_genes[s, :i + 1]], pixel_colors[:, s:s + 1],
                                      weight[:, s:s + 1])
                    else:
                        # TODO: maybe do this fitting with all unique combinations of genes so can do
                        #  multiple spots at once.
                        residual_pixel_colors[:, s:s + 1], all_coefs[s, added_genes[s, :i + 1]] = \
                            fit_coefs(bled_codes[:, added_genes[s, :i + 1]], pixel_colors[:, s:s + 1])
                    pbar.update(1)
        pbar.close()
        residual_pixel_colors = residual_pixel_colors.transpose().reshape((n_pixels, n_rounds, n_channels))

    gene_coefs = all_coefs[:, :n_genes]
    background_coefs = all_coefs[:, -n_channels:]
    return gene_coefs, background_coefs
