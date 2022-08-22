import numpy as np
from .. import utils
from ..call_spots.dot_product_optimised import dot_product_score_single
from ..call_spots import fit_background
from typing import Tuple
from tqdm import tqdm
import jax.numpy as jnp
import jax
from functools import partial


def fit_coefs_single(bled_codes: jnp.ndarray, pixel_color: jnp.ndarray,
                     genes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This finds the least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes`
    can best explain `pixel_color`.

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_color: `float [(n_rounds x n_channels)]`.
            Flattened `pixel_color` which usually has the shape `[n_rounds x n_channels]`.
        genes: `int [n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain pixel_color.

    Returns:
        - residual - `float [(n_rounds x n_channels)]`.
            Residual pixel_color after removing bled_codes with coefficients specified by coefs.
        - coefs - `float [n_genes_add]`.
            Coefficients found through least squares fitting for each gene.
    """
    coefs = jnp.linalg.lstsq(bled_codes[:, genes], pixel_color)[0]
    residual = pixel_color - jnp.matmul(bled_codes[:, genes], coefs)
    return residual, coefs


@jax.jit
def fit_coefs(bled_codes: jnp.ndarray, pixel_colors: jnp.ndarray,
              genes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This finds the least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes[s]`
    can best explain `pixel_colors[:, s]` for each pixel s.

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds x n_channels) x n_pixels]`.
            Flattened then transposed `pixel_colors` which usually has the shape `[n_pixels x n_rounds x n_channels]`.
        genes: `int [n_pixels x n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain each pixel_color.

    Returns:
        - residual - `float [n_pixels x (n_rounds x n_channels)]`.
            Residual pixel_colors after removing bled_codes with coefficients specified by coefs.
        - coefs - `float [n_pixels x n_genes_add]`.
            Coefficients found through least squares fitting for each gene.
    """
    return jax.vmap(fit_coefs_single, in_axes=(None, 1, 0), out_axes=(0, 0))(bled_codes, pixel_colors, genes)


def fit_coefs_weight_single(bled_codes: jnp.ndarray, pixel_color: jnp.ndarray, genes: jnp.ndarray,
                            weight: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This finds the weighted least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes`
    can best explain `pixel_color`. The `weight` indicates which rounds/channels should have more influence when finding
    the coefficients of each gene.

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_color: `float [(n_rounds x n_channels)]`.
            Flattened `pixel_color` which usually has the shape `[n_rounds x n_channels]`.
        genes: `int [n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain pixel_color.
        weight: `float [(n_rounds x n_channels)]`.
            `weight[i]` is the weight to be applied to round_channel `i` when computing coefficient of each
            `bled_code`.

    Returns:
        - residual - `float [(n_rounds x n_channels)]`.
            Residual pixel_color after removing bled_codes with coefficients specified by coefs.
        - coefs - `float [n_genes_add]`.
            Coefficients found through least squares fitting for each gene.
    """
    coefs = jnp.linalg.lstsq(bled_codes[:, genes] * weight[:, jnp.newaxis], pixel_color * weight)[0]
    residual = pixel_color * weight - jnp.matmul(bled_codes[:, genes] * weight[:, jnp.newaxis], coefs)
    return residual / weight, coefs


@jax.jit
def fit_coefs_weight(bled_codes: jnp.ndarray, pixel_colors: jnp.ndarray, genes: jnp.ndarray,
                     weight: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This finds the weighted least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes[s]`
    can best explain `pixel_colors[:, s]` for each pixel s. The `weight` indicates which rounds/channels should
    have more influence when finding the coefficients of each gene.

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds x n_channels) x n_pixels]`.
            Flattened then transposed `pixel_colors` which usually has the shape `[n_pixels x n_rounds x n_channels]`.
        genes: `int [n_pixels x n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain each pixel_color.
        weight: `float [n_pixels x (n_rounds x n_channels)]`.
            `weight[s, i]` is the weight to be applied to round_channel `i` when computing coefficient of each
            `bled_code` for pixel `s`.

    Returns:
        - residual - `float [n_pixels x (n_rounds x n_channels)]`.
            Residual pixel_colors after removing bled_codes with coefficients specified by coefs.
        - coefs - `float [n_pixels x n_genes_add]`.
            Coefficients found through least squares fitting for each gene.
    """
    return jax.vmap(fit_coefs_weight_single, in_axes=(None, 1, 0, 0), out_axes=(0, 0))(bled_codes, pixel_colors, genes,
                                                                                       weight)


def get_best_gene_base(residual_pixel_color: jnp.ndarray, all_bled_codes: jnp.ndarray,
                       norm_shift: float, score_thresh: float, inverse_var: jnp.ndarray,
                       ignore_genes: jnp.ndarray) -> Tuple[int, bool]:
    """
    Computes the `dot_product_score` between `residual_pixel_color` and each code in `all_bled_codes`.
    If `best_score` is less than `score_thresh` or if the corresponding `best_gene` is in `ignore_genes`,
    then `pass_score_thresh` will be False.

    Args:
        residual_pixel_color: `float [(n_rounds x n_channels)]`.
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
        ignore_genes: `int [n_genes_ignore]`.
            If `best_gene` is one of these, `pass_score_thresh` will be `False`.

    Returns:
        - best_gene - The best gene to add next.
        - pass_score_thresh - `True` if `best_score > score_thresh` and `best_gene` not in `ignore_genes`.

    """
    # calculate score including background genes as if best gene is background, then stop iteration.
    all_scores = dot_product_score_single(residual_pixel_color, all_bled_codes, norm_shift, inverse_var)
    best_gene = jnp.argmax(jnp.abs(all_scores))
    # if best_gene is background, set score below score_thresh.
    best_score = all_scores[best_gene] * jnp.isin(best_gene, ignore_genes, invert=True)
    pass_score_thresh = jnp.abs(best_score) > score_thresh
    return best_gene, pass_score_thresh


def get_best_gene_first_iter_single(residual_pixel_color: jnp.ndarray, all_bled_codes: jnp.ndarray,
                                    background_coefs: jnp.ndarray, norm_shift: float,
                                    score_thresh: float, alpha: float, beta: float,
                                    background_genes: jnp.ndarray) -> Tuple[int, bool, jnp.ndarray]:
    """
    Finds the `best_gene` to add next based on the dot product score with each `bled_code`.
    If `best_gene` is in `background_genes` or `best_score < score_thresh` then `pass_score_thresh = False`.
    Different for first iteration as no actual non-zero gene coefficients to consider when computing variance
    or genes that can be added which will cause `pass_score_thresh` to be `False`.

    Args:
        residual_pixel_color: `float [(n_rounds x n_channels)]`.
            Residual pixel color from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
            Includes codes of genes and background.
        background_coefs: `float [n_channels]`.
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
        - best_gene - The best gene to add next.
        - pass_score_thresh - `True` if `best_score > score_thresh`.
        - background_var - `float [(n_rounds x n_channels)]`.
            Variance in each round/channel based on just the background.

    """
    background_var = jnp.square(background_coefs) @ jnp.square(all_bled_codes[background_genes]) * alpha + beta ** 2
    best_gene, pass_score_thresh = get_best_gene_base(residual_pixel_color, all_bled_codes, norm_shift, score_thresh,
                                                      1 / background_var, background_genes)
    return best_gene, pass_score_thresh, background_var


# static_argnums refer to arguments which do not change.
# arrays are not hashable hence all_bled_codes and background_genes not static.
#  https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#caching
@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def get_best_gene_first_iter(residual_pixel_colors: jnp.ndarray, all_bled_codes: jnp.ndarray,
                             background_coefs: jnp.ndarray, norm_shift: float,
                             score_thresh: float, alpha: float, beta: float,
                             background_genes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Finds the `best_gene` to add next to each pixel based on the dot product score with each `bled_code`.
    If `best_gene[s]` is in `background_genes` or `best_score[s] < score_thresh` then `pass_score_thresh[s] = False`.
    Different for first iteration as no actual non-zero gene coefficients to consider when computing variance
    or genes that can be added which will cause `pass_score_thresh` to be `False`.

    Args:
        residual_pixel_colors: `float [n_pixels x (n_rounds x n_channels)]`.
            Residual pixel colors from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
            Includes codes of genes and background.
        background_coefs: `float [n_pixels x n_channels]`.
            `coefs[s, g]` is the weighting of pixel `s` for gene `background_genes[g]` found by the omp algorithm.
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
            Variance of each pixel in each round/channel based on just the background.

    """
    return jax.vmap(get_best_gene_first_iter_single, in_axes=(0, None, 0, None, None, None, None, None),
                    out_axes=(0, 0, 0))(residual_pixel_colors, all_bled_codes, background_coefs, norm_shift,
                                           score_thresh, alpha, beta, background_genes)


def get_best_gene_single(residual_pixel_color: jnp.ndarray, all_bled_codes: jnp.ndarray, coefs: jnp.ndarray,
                         genes_added: jnp.array, norm_shift: float, score_thresh: float, alpha: float,
                         background_genes: jnp.ndarray, background_var: jnp.array) -> Tuple[int, bool, jnp.ndarray]:
    """
    Finds the `best_gene` to add next to each pixel based on the dot product score with each `bled_code`.
    If `best_gene` is in `background_genes`, already in `genes_added` or `best_score < score_thresh`,
    then `pass_score_thresh = False`.

    !!!note
        The variance computed is based on maximum likelihood estimation - it accounts for all genes and background
        fit in each round/channel. The more genes added, the greater the variance so if the inverse is used as a
        weighting for omp fitting or choosing the next gene,
        the rounds/channels which already have genes in will contribute less.

    Args:
        residual_pixel_color: `float [(n_rounds x n_channels)]`.
            Residual pixel color from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
            Includes codes of genes and background.
        coefs: `float [n_genes_added]`.
            `coefs[g]` is the weighting for gene `genes_added[g]` found by the omp algorithm on previous iteration.
             All are non-zero.
        genes_added: `int [n_genes_added]`
            Indices of genes added to each pixel from previous iteration of omp.
            If the best gene for pixel `s` is set to one of `genes_added[s]`, `pass_score_thresh[s]` will be False.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added in the current iteration.
        alpha: Used for `fitting_variance`, by how much to increase variance as genes added.
        background_genes: `int [n_channels]`.
            Indices of codes in all_bled_codes which correspond to background.
            If the best gene is set to one of `background_genes`, `pass_score_thresh` will be False.
        background_var: `float [(n_rounds x n_channels)]`.
            Contribution of background genes to variance (which does not change throughout omp iterations)  i.e.
            `background_coefs**2 @ all_bled_codes[background_genes]**2 * alpha + beta ** 2`.

    Returns:
        - best_gene - The best gene to add next.
        - pass_score_thresh - `True` if `best_score > score_thresh`.
        - inverse_var - `float [(n_rounds x n_channels)]`.
            Inverse of variance in each round/channel based on genes fit on previous iteration.
            Includes both background and gene contribution.
    """
    inverse_var = 1 / (jnp.square(coefs) @ jnp.square(all_bled_codes[genes_added]) * alpha + background_var)
    # calculate score including background genes as if best gene is background, then stop iteration.
    best_gene, pass_score_thresh = get_best_gene_base(residual_pixel_color, all_bled_codes, norm_shift, score_thresh,
                                                      inverse_var, jnp.append(background_genes, genes_added))
    return best_gene, pass_score_thresh, inverse_var


@partial(jax.jit, static_argnums=(4, 5, 6))
def get_best_gene(residual_pixel_colors: jnp.ndarray, all_bled_codes: jnp.ndarray, coefs: jnp.ndarray,
                  genes_added: jnp.array, norm_shift: float, score_thresh: float, alpha: float,
                  background_genes: jnp.ndarray,
                  background_var: jnp.array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    """
    return jax.vmap(get_best_gene_single, in_axes=(0, None, 0, 0, None, None, None, None, 0),
                    out_axes=(0, 0, 0))(residual_pixel_colors, all_bled_codes, coefs, genes_added, norm_shift,
                                        score_thresh, alpha, background_genes, background_var)


def get_all_coefs(pixel_colors: jnp.ndarray, bled_codes: jnp.ndarray, background_shift: float,
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
        - gene_coefs - `float32 [n_pixels x n_genes]`.
            `gene_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm. Most are zero.
        - background_coefs - `float32 [n_pixels x n_channels]`.
            coefficient value for each background vector found for each pixel.
    """
    n_pixels = pixel_colors.shape[0]

    check_spot = np.random.randint(n_pixels)
    diff_to_int = jnp.round(pixel_colors[check_spot]).astype(int) - pixel_colors[check_spot]
    if jnp.abs(diff_to_int).max() == 0:
        raise ValueError(f"pixel_coefs should be found using normalised pixel_colors."
                         f"\nBut for pixel {check_spot}, pixel_colors given are integers indicating they are "
                         f"the raw intensities.")

    n_genes, n_rounds, n_channels = bled_codes.shape
    if not utils.errors.check_shape(pixel_colors, [n_pixels, n_rounds, n_channels]):
        raise utils.errors.ShapeError('pixel_colors', pixel_colors.shape, (n_pixels, n_rounds, n_channels))
    no_verbose = n_pixels < 1000  # show progress bar with more than 1000 pixels.

    # Fit background and override initial pixel_colors
    gene_coefs = np.zeros((n_pixels, n_genes), dtype=np.float32)  # coefs of all genes and background
    pixel_colors, background_coefs, background_codes = fit_background(pixel_colors,
                                                                      background_shift)

    background_genes = jnp.arange(n_genes, n_genes + n_channels)

    # colors and codes for get_best_gene function
    # Includes background as if background is the best gene, iteration ends.
    # uses residual color as used to find next gene to add.
    bled_codes = bled_codes.reshape((n_genes, -1))
    all_codes = jnp.concatenate((bled_codes, background_codes.reshape(n_channels, -1)))
    bled_codes = bled_codes.transpose()

    # colors and codes for fit_coefs function (No background as this is not updated again).
    # always uses post background color as coefficients for all genes re-estimated at each iteration.
    pixel_colors = pixel_colors.reshape((n_pixels, -1))

    continue_pixels = jnp.arange(n_pixels)
    with tqdm(total=max_genes, disable=no_verbose) as pbar:
        pbar.set_description('Finding OMP coefficients for each pixel')
        for i in range(max_genes):
            if i == 0:
                # Background coefs don't change, hence contribution to variance won't either.
                added_genes, pass_score_thresh, background_variance = \
                    get_best_gene_first_iter(pixel_colors, all_codes, background_coefs, dp_shift,
                                             dp_thresh, alpha, beta, background_genes)
                inverse_var = 1 / background_variance
                pixel_colors = pixel_colors.transpose()
            else:
                # only continue with pixels for which dot product score exceeds threshold
                i_added_genes, pass_score_thresh, inverse_var = \
                    get_best_gene(residual_pixel_colors, all_codes, i_coefs, added_genes, dp_shift,
                                  dp_thresh, alpha, background_genes, background_variance)

                # For pixels with at least one non-zero coef, add to final gene_coefs when fail the thresholding.
                fail_score_thresh = jnp.invert(pass_score_thresh)
                # gene_coefs[np.asarray(continue_pixels[fail_score_thresh])] = np.asarray(i_coefs[fail_score_thresh])
                gene_coefs[np.asarray(continue_pixels[fail_score_thresh])[:, np.newaxis],
                           np.asarray(added_genes[fail_score_thresh])] = np.asarray(i_coefs[fail_score_thresh])

            continue_pixels = continue_pixels[pass_score_thresh]
            n_continue = jnp.size(continue_pixels)
            pbar.set_postfix({'n_pixels': n_continue})
            if n_continue == 0:
                break
            if i == 0:
                added_genes = added_genes[pass_score_thresh, np.newaxis]
            else:
                added_genes = jnp.hstack((added_genes[pass_score_thresh], i_added_genes[pass_score_thresh, jnp.newaxis]))
            pixel_colors = pixel_colors[:, pass_score_thresh]
            background_variance = background_variance[pass_score_thresh]
            inverse_var = inverse_var[pass_score_thresh]

            # Maybe add different fit_coefs for i==0 i.e. can do multiple pixels at once for same gene added.
            if weight_coef_fit:
                residual_pixel_colors, i_coefs = fit_coefs_weight(bled_codes, pixel_colors, added_genes,
                                                                  jnp.sqrt(inverse_var))
            else:
                residual_pixel_colors, i_coefs = fit_coefs(bled_codes, pixel_colors, added_genes)

            if i == max_genes-1:
                # Add pixels to final gene_coefs when reach end of iteration.
                gene_coefs[np.asarray(continue_pixels)[:, np.newaxis], np.asarray(added_genes)] = np.asarray(i_coefs)

            pbar.update(1)
    pbar.close()

    return gene_coefs.astype(np.float32), np.asarray(background_coefs).astype(np.float32)
