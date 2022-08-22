import numpy as np
from coppafish.call_spots import fit_background, dot_product_score
from typing import Tuple, Union
from tqdm import tqdm
import warnings


def fit_coefs(bled_codes: np.ndarray, pixel_colors: np.ndarray, genes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Old method before Jax.
    This finds the least squared solution for how the `n_genes` `bled_codes` can best explain each `pixel_color`.
    Can also find weighted least squared solution if `weight` provided.

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds x n_channels) x n_pixels]` if `n_genes==1`
            otherwise  `float [(n_rounds x n_channels)]`.
            Flattened then transposed pixel colors which usually has the shape `[n_pixels x n_rounds x n_channels]`.
        genes: `int [n_pixels x n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain each pixel_color.

    Returns:
        - residual - `float [n_pixels x (n_rounds x n_channels)]`.
            Residual pixel_colors after removing bled_codes with coefficients specified by coef.
        - coefs - `float [n_pixels x n_genes_add]` if n_genes == 1 otherwise `float [n_genes]` if n_pixels == 1.
            coefficient found through least squares fitting for each gene.

    """
    n_pixels = pixel_colors.shape[1]
    residual = np.zeros((n_pixels, pixel_colors.shape[0]))
    coefs = np.zeros_like(genes, dtype=float)
    for s in range(n_pixels):
        coefs[s] = np.linalg.lstsq(bled_codes[:, genes[s]], pixel_colors[:, s], rcond=None)[0]
        residual[s] = pixel_colors[:, s] - bled_codes[:, genes[s]] @ coefs[s]
    return residual, coefs


def fit_coefs_weight(bled_codes: np.ndarray, pixel_colors: np.ndarray, genes: np.ndarray,
                     weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Old method before Jax.
    This finds the least squared solution for how the `n_genes` `bled_codes` can best explain each `pixel_color`.
    Can also find weighted least squared solution if `weight` provided.

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds x n_channels) x n_pixels]` if `n_genes==1`
            otherwise  `float [(n_rounds x n_channels)]`.
            Flattened then transposed pixel colors which usually has the shape `[n_pixels x n_rounds x n_channels]`.
        genes: `int [n_pixels x n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain each pixel_color.
        weight: `float [n_pixels x (n_rounds x n_channels)]`.
            `weight[s, i]` is the weight to be applied to round_channel `i` when computing coefficient of each
            `bled_code` for pixel `s`.

    Returns:
        - residual - `float [n_pixels x (n_rounds x n_channels)]`.
            Residual pixel_colors after removing bled_codes with coefficients specified by coef.
        - coefs - `float [n_pixels x n_genes_add]` if n_genes == 1 otherwise `float [n_genes]` if n_pixels == 1.
            coefficient found through least squares fitting for each gene.

    """
    n_pixels = pixel_colors.shape[1]
    residual = np.zeros((n_pixels, pixel_colors.shape[0]))
    coefs = np.zeros_like(genes, dtype=float)
    pixel_colors = pixel_colors * weight.transpose()
    for s in range(n_pixels):
        bled_codes_s = bled_codes[:, genes[s]] * weight[s][:, np.newaxis]
        coefs[s] = np.linalg.lstsq(bled_codes_s, pixel_colors[:, s], rcond=None)[0]
        residual[s] = pixel_colors[:, s] - bled_codes_s @ coefs[s]
    residual = residual / weight
    return residual, coefs


def get_best_gene_base(residual_pixel_colors: np.ndarray, all_bled_codes: np.ndarray,
                       norm_shift: float, score_thresh: float, inverse_var: np.ndarray,
                       ignore_genes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the `dot_product_score` between `residual_pixel_color` and each code in `all_bled_codes`.
    If `best_score` is less than `score_thresh` or if the corresponding `best_gene` is in `ignore_genes`,
    then `pass_score_thresh` will be False.

    Args:
        residual_pixel_colors: `float [n_pixels x (n_rounds x n_channels)]`.
            Residual pixel color from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
            Includes codes of genes and background.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added in the current iteration.
        inverse_var: `float [(n_rounds x n_channels)]`.
            Inverse of variance in each round/channel based on genes fit on previous iteration.
            Used as `weight_squared` when computing `dot_product_score`.
        ignore_genes: `int [n_pixels x n_genes_ignore]`.
            If `best_gene[s]` is one of these, `pass_score_thresh[s]` will be `False`.

    Returns:
        - best_gene - `int [n_pixels]`.
            `best_gene[s]` is the best gene to add to pixel `s` next.
        - pass_score_thresh - `bool [n_pixels]`.
            `True` if `best_score[s] > score_thresh` and `best_gene[s]` not in `ignore_genes`.
        - best_score - `float [n_pixels]`.
            `dot_product_score` for spot `s` with gene `best_gene[s]`.

    """
    # calculate score including background genes as if best gene is background, then stop iteration.
    all_scores = dot_product_score(residual_pixel_colors, all_bled_codes, norm_shift, inverse_var)
    best_gene = np.argmax(np.abs(all_scores), axis=1)
    # if best_gene is in ignore_gene, set score below score_thresh.
    is_ignore_gene = (best_gene[:, np.newaxis] == ignore_genes).any(axis=1)
    best_score = all_scores[(np.arange(best_gene.shape[0]), best_gene)] * np.invert(is_ignore_gene)
    pass_score_thresh = np.abs(best_score) > score_thresh
    return best_gene, pass_score_thresh, best_score


def get_best_gene_first_iter(residual_pixel_colors: np.ndarray, all_bled_codes: np.ndarray,
                             background_coefs: np.ndarray, norm_shift: float,
                             score_thresh: float, alpha: float, beta: float,
                             background_genes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the `best_gene` to add next based on the dot product score with each `bled_code`.
    If `best_gene` is in `background_genes` or `best_score < score_thresh` then `pass_score_thresh = False`.
    Different for first iteration as no actual non-zero gene coefficients to consider when computing variance
    or genes that can be added which will cause `pass_score_thresh` to be `False`.

    Args:
        residual_pixel_colors: `float [n_pixels x (n_rounds x n_channels)]`.
            Residual pixel color from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
            Includes codes of genes and background.
        background_coefs: `float [n_pixels x n_channels]`.
            `coefs[g]` is the weighting for gene `background_genes[g]` found by the omp algorithm.
             All are non-zero.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added in the current iteration.
        alpha: Used for `fitting_variance`, by how much to increase variance as genes added.
        beta: Used for `fitting_variance`, the variance with no genes added (`coef=0`) is `beta**2`.
        background_genes: `int [n_channels]`.
            Indices of codes in all_bled_codes which correspond to background.
            If the best gene for pixel `s` is set to one of `background_genes`, `pass_score_thresh[s]` will be False.

    Returns:
        - best_gene - `int [n_pixels]`.
            `best_gene[s]` is the best gene to add to pixel `s` next.
        - pass_score_thresh - `bool [n_pixels]`.
            `True` if `best_score > score_thresh`.
        - background_var - `float [n_pixels x (n_rounds x n_channels)]`.
            Variance in each round/channel based on just the background.
        - best_score - `float [n_pixels]`.
            `dot_product_score` for spot `s` with gene `best_gene[s]`.

    """
    background_var = np.square(background_coefs) @ np.square(all_bled_codes[background_genes]) * alpha + beta ** 2
    ignore_genes = np.tile(background_genes, [background_var.shape[0], 1])
    best_gene, pass_score_thresh, best_score = \
        get_best_gene_base(residual_pixel_colors, all_bled_codes, norm_shift, score_thresh, 1 / background_var,
                           ignore_genes)
    return best_gene, pass_score_thresh, background_var, best_score


def get_best_gene(residual_pixel_colors: np.ndarray, all_bled_codes: np.ndarray, coefs: np.ndarray,
                  genes_added: np.array, norm_shift: float, score_thresh: float, alpha: float,
                  background_genes: np.ndarray,
                  background_var: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the `best_gene` to add next to each pixel based on the dot product score with each `bled_code`.
    If `best_gene[s]` is in `background_genes`, already in `genes_added[s]` or `best_score[s] < score_thresh`,
    then `pass_score_thresh[s] = False`.

    !!!note
        The variance computed is based on maximum likelihood estimation - it accounts for all genes and background
        fit in each round/channel. The more genes added, the greater the variance so if the inverse is used as a
        weighting for omp fitting or choosing the next gene,
        the rounds/channels which already have genes in will contribute less.

    Args:
        residual_pixel_colors: `float [n_pixels x (n_rounds x n_channels)]`.
            Residual pixel colors from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
            Includes codes of genes and background.
        coefs: `float [n_pixels x n_genes_added]`.
            `coefs[s, g]` is the weighting of pixel `s` for gene `genes_added[g]` found by the omp algorithm on previous
            iteration. All are non-zero.
        genes_added: `int [n_pixels x n_genes_added]`
            Indices of genes added to each pixel from previous iteration of omp.
            If the best gene for pixel `s` is set to one of `genes_added[s]`, `pass_score_thresh[s]` will be False.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added in the current iteration.
        alpha: Used for `fitting_variance`, by how much to increase variance as genes added.
        background_genes: `int [n_channels]`.
            Indices of codes in all_bled_codes which correspond to background.
            If the best gene for pixel `s` is set to one of `background_genes`, `pass_score_thresh[s]` will be False.
        background_var: `float [n_pixels x (n_rounds x n_channels)]`.
            Contribution of background genes to variance (which does not change throughout omp iterations)  i.e.
            `background_coefs**2 @ all_bled_codes[background_genes]**2 * alpha + beta ** 2`.

    Returns:
        - best_gene - `int [n_pixels]`.
            `best_gene[s]` is the best gene to add to pixel `s` next.
        - pass_score_thresh - `bool [n_pixels]`.
            `True` if `best_score > score_thresh`.
        - inverse_var - `float [n_pixels x (n_rounds x n_channels)]`.
            Inverse of variance of each pixel in each round/channel based on genes fit on previous iteration.
            Includes both background and gene contribution.
        - best_score - `float [n_pixels]`.
            `dot_product_score` for spot `s` with gene `best_gene[s]`.
    """

    n_pixels, n_genes_added = genes_added.shape
    n_genes = all_bled_codes.shape[0]
    coefs_all = np.zeros((n_pixels, n_genes))
    pixel_ind = np.repeat(np.arange(n_pixels), n_genes_added)
    coefs_all[(pixel_ind, genes_added.flatten())] = coefs.flatten()

    inverse_var = 1 / (coefs_all ** 2 @ all_bled_codes ** 2 * alpha + background_var)
    ignore_genes = np.concatenate((genes_added, np.tile(background_genes, [n_pixels, 1])), axis=1)
    best_gene, pass_score_thresh, best_score = \
        get_best_gene_base(residual_pixel_colors, all_bled_codes, norm_shift, score_thresh, inverse_var, ignore_genes)

    return best_gene, pass_score_thresh, inverse_var, best_score


def get_all_coefs(pixel_colors: np.ndarray, bled_codes: np.ndarray, background_shift: float,
                  dp_shift: float, dp_thresh: float, alpha: float, beta: float, max_genes: int,
                  weight_coef_fit: bool = False,
                  track: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, dict]]:
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
        track: If `True` and one pixel, info about genes added at each step returned.

    Returns:
        gene_coefs - `float32 [n_pixels x n_genes]`.
            `gene_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm. Most are zero.
        background_coefs - `float32 [n_pixels x n_channels]`.
            coefficient value for each background vector found for each pixel.
        track_info - dictionary containing info about genes added at each step returned if `track == True` -

            - `background_codes` - `float [n_channels x n_rounds x n_channels]`.
                `background_codes[c]` is the background vector for channel `c` with L2 norm of 1.
            - `background_coefs` - `float [n_channels]`.
                `background_coefs[c]` is the coefficient value for `background_codes[c]`.
            - `gene_added` - `int [n_genes_added + 2]`.
                `gene_added[0]` and `gene_added[1]` are -1.
                `gene_added[2+i]` is the `ith` gene that was added.
            - `residual` - `float [(n_genes_added + 2) x n_rounds x n_channels]`.
                `residual[0]` is the initial `pixel_color`.
                `residual[1]` is the post background `pixel_color`.
                `residual[2+i]` is the `pixel_color` after removing gene `gene_added[2+i]`.
            - `coef` - `float [(n_genes_added + 2) x n_genes]`.
                `coef[0]` and `coef[1]` are all 0.
                `coef[2+i]` are the coefficients for all genes after the ith gene has been added.
            - `dot_product` - `float [n_genes_added + 2]`.
                `dot_product[0]` and `dot_product[1]` are 0.
                `dot_product[2+i]` is the dot product for the gene `gene_added[2+i]`.
            - `inverse_var` - `float [(n_genes_added + 2) x n_rounds x n_channels]`.
                `inverse_var[0]` and `inverse_var[1]` are all 0.
                `inverse_var[2+i]` is the weighting used to compute `dot_product[2+i]`,
                 which down-weights rounds/channels for which a gene has already been fitted.

    """
    n_pixels = pixel_colors.shape[0]
    n_genes, n_rounds, n_channels = bled_codes.shape

    no_verbose = n_pixels < 1000  # show progress bar with more than 1000 pixels.
    if track:
        return_track = True
        if n_pixels == 1:
            track_info = {'residual': np.zeros((max_genes+3, n_rounds, n_channels)),
                          'background_codes': None, 'background_coefs': None,
                          'inverse_var': np.zeros((max_genes+3, n_rounds, n_channels)),
                          'coef': np.zeros((max_genes+3, n_genes+n_channels)), 'dot_product': np.zeros(max_genes+3),
                          'gene_added': np.ones(max_genes+3, dtype=int) * -1}
            track_info['residual'][0] = pixel_colors[0]
        else:
            warnings.warn(f'Can only get track info if running on one pixel, but there are {n_pixels} pixels '
                          f'so not getting track info.')
            track = False
            track_info = None
    else:
        return_track = False

    # Fit background and override initial pixel_colors
    gene_coefs = np.zeros((n_pixels, n_genes), dtype=np.float32)  # coefs of all genes and background
    pixel_colors, background_coefs, background_codes = fit_background(pixel_colors, background_shift)

    if track:
        track_info['residual'][1] = pixel_colors[0]
        track_info['background_codes'] = background_codes
        track_info['background_coefs'] = background_coefs[0]

    background_genes = np.arange(n_genes, n_genes + n_channels)

    # colors and codes for get_best_gene function
    # Includes background as if background is the best gene, iteration ends.
    # uses residual color as used to find next gene to add.
    bled_codes = bled_codes.reshape((n_genes, -1))
    all_codes = np.concatenate((bled_codes, background_codes.reshape(n_channels, -1)))
    bled_codes = bled_codes.transpose()

    # colors and codes for fit_coefs function (No background as this is not updated again).
    # always uses post background color as coefficients for all genes re-estimated at each iteration.
    pixel_colors = pixel_colors.reshape((n_pixels, -1))

    continue_pixels = np.arange(n_pixels)
    with tqdm(total=max_genes, disable=no_verbose) as pbar:
        pbar.set_description('Finding OMP coefficients for each pixel')
        for i in range(max_genes):
            if i == 0:
                # Background coefs don't change, hence contribution to variance won't either.
                added_genes, pass_score_thresh, background_variance, best_score = \
                    get_best_gene_first_iter(pixel_colors, all_codes, background_coefs, dp_shift,
                                             dp_thresh, alpha, beta, background_genes)
                inverse_var = 1 / background_variance
                pixel_colors = pixel_colors.transpose()
            else:
                # only continue with pixels for which dot product score exceeds threshold
                i_added_genes, pass_score_thresh, inverse_var, best_score = \
                    get_best_gene(residual_pixel_colors, all_codes, i_coefs, added_genes, dp_shift,
                                  dp_thresh, alpha, background_genes, background_variance)

                # For pixels with at least one non-zero coef, add to final gene_coefs when fail the thresholding.
                fail_score_thresh = np.invert(pass_score_thresh)
                gene_coefs[np.asarray(continue_pixels[fail_score_thresh])[:, np.newaxis],
                           np.asarray(added_genes[fail_score_thresh])] = i_coefs[fail_score_thresh]

            continue_pixels = continue_pixels[pass_score_thresh]
            n_continue = np.size(continue_pixels)
            pbar.set_postfix({'n_pixels': n_continue})
            if n_continue == 0:
                if track:
                    track_info['inverse_var'][i + 2] = inverse_var.reshape(n_rounds, n_channels)
                    track_info['dot_product'][i + 2] = best_score[0]
                    if i == 0:
                        track_info['gene_added'][i + 2] = added_genes
                    else:
                        track_info['gene_added'][i + 2] = i_added_genes
                        added_genes_fail = np.hstack((added_genes, i_added_genes[:, np.newaxis]))
                        # Need to usee all_codes here to deal with case where the best gene is background
                        if weight_coef_fit:
                            residual_pixel_colors_fail, i_coefs_fail = \
                                fit_coefs_weight(all_codes.T, pixel_colors, added_genes_fail, np.sqrt(inverse_var))
                        else:
                            residual_pixel_colors_fail, i_coefs_fail = fit_coefs(all_codes.T, pixel_colors, added_genes_fail)
                        track_info['residual'][i + 2] = residual_pixel_colors_fail.reshape(n_rounds, n_channels)
                        track_info['coef'][i + 2][added_genes_fail] = i_coefs_fail
                    # Only save info where gene is actually added or for final case where not added.
                    for key in track_info.keys():
                        if 'background' not in key:
                            track_info[key] = track_info[key][:i+3]
                break

            if i == 0:
                added_genes = added_genes[pass_score_thresh, np.newaxis]
            else:
                added_genes = np.hstack((added_genes[pass_score_thresh], i_added_genes[pass_score_thresh, np.newaxis]))
            pixel_colors = pixel_colors[:, pass_score_thresh]
            background_variance = background_variance[pass_score_thresh]
            inverse_var = inverse_var[pass_score_thresh]

            if weight_coef_fit:
                residual_pixel_colors, i_coefs = fit_coefs_weight(bled_codes, pixel_colors, added_genes,
                                                                  np.sqrt(inverse_var))
            else:
                residual_pixel_colors, i_coefs = fit_coefs(bled_codes, pixel_colors, added_genes)

            if i == max_genes-1:
                # Add pixels to final gene_coefs when reach end of iteration.
                gene_coefs[continue_pixels[:, np.newaxis], added_genes] = i_coefs

            if track:
                track_info['residual'][i + 2] = residual_pixel_colors.reshape(n_rounds, n_channels)
                track_info['inverse_var'][i + 2] = inverse_var.reshape(n_rounds, n_channels)
                track_info['coef'][i + 2][added_genes] = i_coefs
                track_info['dot_product'][i + 2] = best_score[0]
                track_info['gene_added'][i + 2] = added_genes[0][-1]

            pbar.update(1)
    pbar.close()
    if track:
        # Only return
        no_gene_add_ind = np.where(track_info['gene_added'] == -1)[0]
        no_gene_add_ind = no_gene_add_ind[no_gene_add_ind >= 2]
        if len(no_gene_add_ind) > 0:
            final_ind = no_gene_add_ind.min()
    if return_track:
        return gene_coefs.astype(np.float32), background_coefs.astype(np.float32), track_info
    else:
        return gene_coefs.astype(np.float32), background_coefs.astype(np.float32)
