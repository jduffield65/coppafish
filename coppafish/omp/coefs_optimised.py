from typing import Tuple, Union, List, Any
from tqdm import tqdm
import numpy as np
import scipy
import os
import jax
import jax.numpy as jnp

from . import coefs
from .. import utils
from .. import call_spots
from ..setup import NotebookPage
from ..call_spots import dot_product_optimised

if jax.default_backend() == 'cpu':
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={utils.threads.get_available_threads()}'


def fit_coefs_single(bled_codes: jnp.ndarray, pixel_color: jnp.ndarray,
                     genes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This finds the least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes`
    can best explain `pixel_color`.

    Args:
        bled_codes: `float [(n_rounds * n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_color: `float [(n_rounds * n_channels)]`.
            Flattened `pixel_color` which usually has the shape `[n_rounds x n_channels]`.
        genes: `int [n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain pixel_color.

    Returns:
        - residual - `float [(n_rounds * n_channels)]`.
            Residual pixel_color after removing bled_codes with coefficients specified by coefs.
        - coefs - `float [n_genes_add]`.
            Coefficients found through least squares fitting for each gene.
    """
    coefs = jnp.linalg.lstsq(bled_codes[:, genes], pixel_color)[0]
    residual = pixel_color - jnp.matmul(bled_codes[:, genes], coefs)
    return residual, coefs


def fit_coefs(bled_codes: jnp.ndarray, pixel_colors: jnp.ndarray,
              genes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This finds the least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes[s]`
    can best explain `pixel_colors[:, s]` for each pixel s.

    Args:
        bled_codes (`[(n_rounds * n_channels) x n_genes] ndarray[float]`): flattened then transposed bled codes which 
            usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors (`[(n_rounds * n_channels) x n_pixels] ndarray[float]`): flattened then transposed `pixel_colors` 
            which usually has the shape `[n_pixels x n_rounds x n_channels]`.
        genes (`[n_pixels x n_genes_add] ndarray[int]`): indices of codes in `bled_codes` to find coefficients for 
            which best explain each pixel color.

    Returns:
        - residuals - `[n_pixels x (n_rounds * n_channels)] ndarray[float]`.
            Residual pixel_colors after removing bled_codes with coefficients specified by coefs.
        - coefs - `[n_pixels x n_genes_add] ndarray[float]`.
            Coefficients found through least squares fitting for each gene.
    """
    # Pad pixels to use on all CPU cores
    n_devices = jax.local_device_count()
    n_rounds_channels, n_pixels = pixel_colors.shape
    n_genes_add = genes.shape[1]
    for padding in range(n_devices + 1):
        if (n_pixels + padding) % n_devices == 0:
            break
    pixel_colors = pixel_colors.transpose((1, 0))
    pixel_colors_batched = jnp.append(pixel_colors, jnp.ones((padding, n_rounds_channels)), axis=0)
    genes_batched = jnp.append(genes, jnp.repeat(jnp.arange(n_genes_add)[None], padding, axis=0))
    
    pixel_colors_batched = pixel_colors_batched.reshape((n_devices, -1, n_rounds_channels))
    genes_batched = genes_batched.reshape((n_devices, -1, n_genes_add))
    
    vmap = jax.vmap(fit_coefs_single, in_axes=(None, 0, 0), out_axes=(0, 0))
    residuals, coefs = jax.pmap(vmap, in_axes=(None, 0, 0), out_axes=(0, 0))(
        bled_codes, pixel_colors_batched, genes_batched
    )
    # Combine batches
    return residuals.reshape((-1, n_rounds_channels))[:n_pixels], coefs.reshape((-1, n_genes_add))[:n_pixels]


def fit_coefs_weight_single(bled_codes: jnp.ndarray, pixel_color: jnp.ndarray, genes: jnp.ndarray,
                            weight: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This finds the weighted least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes`
    can best explain `pixel_color`. The `weight` indicates which rounds/channels should have more influence when finding
    the coefficients of each gene.

    Args:
        bled_codes: `float [(n_rounds * n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_color: `float [(n_rounds * n_channels)]`.
            Flattened `pixel_color` which usually has the shape `[n_rounds x n_channels]`.
        genes: `int [n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain pixel_color.
        weight: `float [(n_rounds * n_channels)]`.
            `weight[i]` is the weight to be applied to round_channel `i` when computing coefficient of each
            `bled_code`.

    Returns:
        - residual - `float [(n_rounds * n_channels)]`.
            Residual pixel_color after removing bled_codes with coefficients specified by coefs.
        - coefs - `float [n_genes_add]`.
            Coefficients found through least squares fitting for each gene.
    """
    coefs = jnp.linalg.lstsq(bled_codes[:, genes] * weight[:, jnp.newaxis], pixel_color * weight, rcond=-1)[0]
    residual = pixel_color * weight - jnp.matmul(bled_codes[:, genes] * weight[:, jnp.newaxis], coefs)
    return residual / weight, coefs


def fit_coefs_weight(bled_codes: jnp.ndarray, pixel_colors: jnp.ndarray, genes: jnp.ndarray,
                     weight: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This finds the weighted least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes[s]`
    can best explain `pixel_colors[:, s]` for each pixel s. The `weight` indicates which rounds/channels should
    have more influence when finding the coefficients of each gene.

    Args:
        bled_codes: `float [(n_rounds * n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds * n_channels) x n_pixels]`.
            Flattened then transposed `pixel_colors` which usually has the shape `[n_pixels x n_rounds x n_channels]`.
        genes: `int [n_pixels x n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain each pixel_color.
        weight: `float [n_pixels x (n_rounds * n_channels)]`.
            `weight[s, i]` is the weight to be applied to round_channel `i` when computing coefficient of each
            `bled_code` for pixel `s`.

    Returns:
        - residual - `float [n_pixels x (n_rounds * n_channels)]`.
            Residual pixel_colors after removing bled_codes with coefficients specified by coefs.
        - coefs - `float [n_pixels x n_genes_add]`.
            Coefficients found through least squares fitting for each gene.
    """
    n_devices = jax.local_device_count()
    n_rounds_channels, n_pixels = pixel_colors.shape
    n_genes_add = genes.shape[1]
    for padding in range(n_devices + 1):
        if (n_pixels + padding) % n_devices == 0:
            break
    pixel_colors = pixel_colors.transpose((1, 0))
    pixel_colors_batched = jnp.append(pixel_colors, jnp.ones((padding, n_rounds_channels)), axis=0)
    genes_batched = jnp.append(genes, jnp.repeat(jnp.arange(n_genes_add)[None], padding, axis=0))
    weight_batched = jnp.append(weight, jnp.ones((padding, n_rounds_channels)), axis=0)
    
    pixel_colors_batched = pixel_colors_batched.reshape((n_devices, -1, n_rounds_channels))
    genes_batched = genes_batched.reshape((n_devices, -1, n_genes_add))
    weight_batched = weight_batched.reshape((n_devices, -1, n_rounds_channels))
    
    vmap = jax.vmap(fit_coefs_weight_single, in_axes=(None, 0, 0, 0), out_axes=(0, 0))
    residuals, coefs = jax.pmap(vmap, in_axes=(None, 0, 0, 0), out_axes=(0, 0))(
        bled_codes, pixel_colors_batched, genes_batched, weight_batched
    )
    # Combine batches
    return residuals.reshape((-1, n_rounds_channels))[:n_pixels], coefs.reshape((-1, n_genes_add))[:n_pixels]


def get_best_gene_base(residual_pixel_color: jnp.ndarray, all_bled_codes: jnp.ndarray,
                       norm_shift: float, score_thresh: float, inverse_var: jnp.ndarray,
                       ignore_genes: jnp.ndarray) -> Tuple[int, bool]:
    """
    Computes the `dot_product_score` between `residual_pixel_color` and each code in `all_bled_codes`. If `best_score` 
    is less than `score_thresh` or if the corresponding `best_gene` is in `ignore_genes`, then `pass_score_thresh` will 
    be False.

    Args:
        residual_pixel_color (`[(n_rounds * n_channels)] ndarray[float]`): residual pixel color from previous iteration 
            of omp.
        all_bled_codes (`[n_genes x (n_rounds * n_channels)] ndarray[float]`): `bled_codes` such that `spot_color` of a 
            gene `g` in round `r` is expected to be a constant multiple of `bled_codes[g, r]`. Includes codes of genes 
            and background.
        norm_shift (float): shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh (float): `dot_product_score` of the best gene for a pixel must exceed this for that gene to be 
            added in the current iteration.
        inverse_var (`[(n_rounds * n_channels)] ndarray[float]`): inverse of variance in each round/channel based on 
            genes fit on previous iteration. Used as `weight_squared` when computing `dot_product_score`.
        ignore_genes (`[n_genes_ignore] ndarray[int]`): if `best_gene` is one of these, `pass_score_thresh` will be 
            `False`.

    Returns:
        - best_gene - The best gene to add next.
        - pass_score_thresh - `True` if `best_score > score_thresh` and `best_gene` not in `ignore_genes`.
    """
    # Calculate score including background genes as if best gene is background, then stop iteration.
    all_scores = dot_product_optimised.dot_product_score(residual_pixel_color[None], all_bled_codes, norm_shift, 
                                                         inverse_var[None])[0]
    best_gene = jnp.argmax(jnp.abs(all_scores))
    # if best_gene is background, set score below score_thresh.
    best_score = all_scores[best_gene] * jnp.isin(best_gene, ignore_genes, invert=True)
    pass_score_thresh = jnp.abs(best_score) > score_thresh
    if type(best_gene) == jnp.ndarray:
        best_gene = best_gene[0]
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
        residual_pixel_color: `float [(n_rounds * n_channels)]`.
            Residual pixel color from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds * n_channels)]`.
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
        - background_var - `float [(n_rounds * n_channels)]`.
            Variance in each round/channel based on just the background.

    """
    background_var = jnp.square(background_coefs) @ jnp.square(all_bled_codes[background_genes]) * alpha + beta ** 2
    best_gene, pass_score_thresh = get_best_gene_base(residual_pixel_color, all_bled_codes, norm_shift, score_thresh,
                                                      1 / background_var, background_genes)
    return best_gene, pass_score_thresh, background_var


def get_best_gene_first_iter(residual_pixel_colors: jnp.ndarray, all_bled_codes: jnp.ndarray,
                             background_coefs: jnp.ndarray, norm_shift: float,
                             score_thresh: float, alpha: float, beta: float,
                             background_genes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Finds the `best_gene` to add next based on the dot product score with each `bled_code`.
    If `best_gene` is in `background_genes` or `best_score < score_thresh` then `pass_score_thresh = False`.
    Different for first iteration as no actual non-zero gene coefficients to consider when computing variance
    or genes that can be added which will cause `pass_score_thresh` to be `False`.

    Args:
        residual_pixel_colors (`[n_pixels x (n_rounds * n_channels)] ndarray[float]`): residual pixel color from 
            previous iteration of omp.
        all_bled_codes (`[n_genes x (n_rounds * n_channels)] ndarray[float]`): `bled_codes` such that `spot_color` of a 
            gene `g` in round `r` is expected to be a constant multiple of `bled_codes[g, r]`. Includes codes of genes 
            and background.
        background_coefs (`[n_pixels x n_channels] ndarray[float]`): `coefs[g]` is the weighting for gene 
            `background_genes[g]` found by the omp algorithm. All are non-zero.
        norm_shift (float): shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh (float): `dot_product_score` of the best gene for a pixel must exceed this for that gene to be 
            added in the current iteration.
        alpha (float): Used for `fitting_variance`, by how much to increase variance as genes added.
        beta (float): Used for `fitting_variance`, the variance with no genes added (`coef=0`) is `beta**2`.
        background_genes (`[n_channels] ndarray[int]`): Indices of codes in `all_bled_codes` which correspond to 
            background. If the best gene for pixel `s` is set to one of `background_genes`, `pass_score_thresh[s]` 
            will be `False`.

    Returns:
        - best_gene (`[n_pixels] ndarray[int]`).
            `best_gene[s]` is the best gene to add to pixel `s` next.
        - pass_score_thresh (`[n_pixels] ndarray[bool]`).
            `True` if `best_score > score_thresh`.
        - background_var (`[n_pixels x (n_rounds * n_channels)] ndarray[float]`).
            Variance in each round/channel based on just the background.
    """
    n_pixels, n_rounds_channels = residual_pixel_colors.shape
    n_channels = background_coefs.shape[1]
    n_devices = jax.local_device_count()
    # We pad the number of pixels so n_pixels is a divisor of n_devices and can run on all CPU cores
    for padding in range(n_devices + 1):
        if (n_pixels + padding) % n_devices == 0:
            break
    
    residual_pixel_colors_batched = jnp.append(residual_pixel_colors, jnp.ones((padding, n_rounds_channels)), axis=0)
    background_coefs_batched = jnp.append(background_coefs, jnp.ones((padding, n_channels)), axis=0)
    
    residual_pixel_colors_batched = residual_pixel_colors_batched.reshape((n_devices, -1, n_rounds_channels))
    background_coefs_batched = background_coefs_batched.reshape((n_devices, -1, n_channels))

    vmap = jax.vmap(get_best_gene_first_iter_single, in_axes=(0, None, 0, None, None, None, None, None), 
                    out_axes=(0, 0, 0))
    best_genes, pass_score_thresholds, backgrounds_vars \
        = jax.pmap(vmap, in_axes=(0, None, 0, None, None, None, None, None), out_axes=(0, 0, 0))(
            residual_pixel_colors_batched, all_bled_codes, background_coefs_batched, norm_shift, score_thresh, alpha, 
            beta, background_genes, 
        )
    # Combine batches
    return (best_genes.reshape((n_pixels + padding))[:n_pixels], 
            pass_score_thresholds.reshape((n_pixels + padding))[:n_pixels], 
            backgrounds_vars.reshape((n_pixels + padding, n_rounds_channels))[:n_pixels])


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
        residual_pixel_color: `float [(n_rounds * n_channels)]`.
            Residual pixel color from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds * n_channels)]`.
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
        background_var: `float [(n_rounds * n_channels)]`.
            Contribution of background genes to variance (which does not change throughout omp iterations)  i.e.
            `background_coefs**2 @ all_bled_codes[background_genes]**2 * alpha + beta ** 2`.

    Returns:
        - best_gene - The best gene to add next.
        - pass_score_thresh - `True` if `best_score > score_thresh`.
        - inverse_var - `float [(n_rounds * n_channels)]`.
            Inverse of variance in each round/channel based on genes fit on previous iteration.
            Includes both background and gene contribution.
    """
    inverse_var = 1 / (jnp.square(coefs) @ jnp.square(all_bled_codes[genes_added]) * alpha + background_var)
    # calculate score including background genes as if best gene is background, then stop iteration.
    best_gene, pass_score_thresh = get_best_gene_base(residual_pixel_color, all_bled_codes, norm_shift, score_thresh,
                                                      inverse_var, jnp.append(background_genes, genes_added))
    return best_gene, pass_score_thresh, inverse_var


def get_best_gene(residual_pixel_colors: jnp.ndarray, all_bled_codes: jnp.ndarray, coefs: jnp.ndarray,
                  genes_added: jnp.array, norm_shift: float, score_thresh: float, alpha: float,
                  background_genes: jnp.ndarray,
                  background_var: jnp.array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Finds the `best_gene` to add next to each pixel based on the dot product score with each `bled_code`.
    If `best_gene[s]` is in `background_genes`, already in `genes_added[s]` or `best_score[s] < score_thresh`,
    then `pass_score_thresh[s] = False`.

    Args:
        residual_pixel_colors: `float [n_pixels x (n_rounds * n_channels)]`.
            Residual pixel colors from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds * n_channels)]`.
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
        background_var: `float [n_pixels x (n_rounds * n_channels)]`.
            Contribution of background genes to variance (which does not change throughout omp iterations)  i.e.
            `background_coefs**2 @ all_bled_codes[background_genes]**2 * alpha + beta ** 2`.

    Returns:
        - best_gene - `int [n_pixels]`.
            `best_gene[s]` is the best gene to add to pixel `s` next.
        - pass_score_thresh - `bool [n_pixels]`.
            `True` if `best_score > score_thresh`.
        - inverse_var - `float [n_pixels x (n_rounds * n_channels)]`.
            Inverse of variance of each pixel in each round/channel based on genes fit on previous iteration.
            Includes both background and gene contribution.

    Notes:
        - The variance computed is based on maximum likelihood estimation - it accounts for all genes and background 
            fit in each round/channel. The more genes added, the greater the variance so if the inverse is used as a 
            weighting for omp fitting or choosing the next gene, the rounds/channels which already have genes in will 
            contribute less.
    """
    n_pixels, n_rounds_channels = residual_pixel_colors.shape
    n_genes_add = coefs.shape[1]
    n_devices = jax.local_device_count()
    
    # Pad the data so the data can be split between n_devices
    for padding in range(n_devices + 1):
        if (n_pixels + padding) % n_devices == 0:
            break
    residual_pixel_colors_batched = jnp.append(residual_pixel_colors, jnp.ones((padding, n_rounds_channels)))
    coefs_batched = jnp.append(coefs, jnp.ones((padding, n_genes_add)))
    genes_added_batched = jnp.append(genes_added, jnp.repeat(jnp.arange(n_genes_add)[None], padding, axis=0))
    background_var_batched = jnp.append(background_var, jnp.ones((padding, n_rounds_channels)))
    
    residual_pixel_colors_batched = residual_pixel_colors_batched.reshape((n_devices, -1, n_rounds_channels))
    coefs_batched = coefs_batched.reshape((n_devices, -1, n_genes_add))
    genes_added_batched = genes_added_batched.reshape((n_devices, -1, n_genes_add))
    background_var_batched = background_var_batched.reshape((n_devices, -1, n_rounds_channels))
    
    in_axes = (0, None, 0, 0, None, None, None, None, 0)
    out_axes = (0, 0, 0)
    vmap = jax.vmap(get_best_gene_single, in_axes=in_axes, out_axes=out_axes)
    best_genes, pass_score_thresholds, inverse_vars = jax.pmap(vmap, in_axes=in_axes, out_axes=out_axes)(
        residual_pixel_colors_batched, all_bled_codes, coefs_batched, genes_added_batched, norm_shift, score_thresh, 
        alpha, background_genes, background_var_batched, 
    )
    # Combine batches
    return (best_genes.reshape((n_pixels + padding))[:n_pixels], 
            pass_score_thresholds.reshape((n_pixels + padding))[:n_pixels], 
            inverse_vars.reshape((n_pixels + padding, n_rounds_channels))[:n_pixels])


def get_all_coefs(pixel_colors: jnp.ndarray, bled_codes: jnp.ndarray, background_shift: float,
                  dp_shift: float, dp_thresh: float, alpha: float, beta: float, max_genes: int,
                  weight_coef_fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    This performs omp on every pixel, the stopping criterion is that the dot_product_score when selecting the next gene 
    to add exceeds dp_thresh or the number of genes added to the pixel exceeds max_genes.

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

    Notes:
        - Background vectors are fitted first and then not updated again.
    """
    n_pixels = pixel_colors.shape[0]

    rng = np.random.RandomState(0)
    check_spot = rng.randint(n_pixels)
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
    pixel_colors, background_coefs, background_codes = call_spots.fit_background(pixel_colors, background_shift)

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
                added_genes = added_genes[pass_score_thresh, jnp.newaxis]
            else:
                added_genes = jnp.hstack((added_genes[pass_score_thresh], 
                                          i_added_genes[pass_score_thresh, jnp.newaxis]))
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
                gene_coefs[jnp.asarray(continue_pixels)[:, jnp.newaxis], np.asarray(added_genes)] \
                    = jnp.asarray(i_coefs)

            pbar.update(1)
    pbar.close()

    return np.asarray(gene_coefs, np.float32), np.asarray(background_coefs, dtype=np.float32)


def get_pixel_coefs_yxz(nbp_basic: NotebookPage, nbp_file: NotebookPage, nbp_extract: NotebookPage, config: dict, 
                        tile: int, use_z: List[int], z_chunk_size: int, n_genes: int, 
                        transform: Union[np.ndarray, np.ndarray], color_norm_factor: Union[np.ndarray, np.ndarray], 
                        initial_intensity_thresh: float, bled_codes: Union[np.ndarray, np.ndarray], 
                        dp_norm_shift: Union[int, float]) -> Tuple[np.ndarray, Any]:
    """
    Get each pixel OMP coefficients for a particular tile.
    
    Args:
        nbp_basic (NotebookPage): notebook page for 'basic_info'.
        nbp_file (NotebookPage): notebook page for 'file_names'.
        nbp_extract (NotebookPage): notebook page for 'extract'.
        config (dict): config settings for 'omp'.
        tile (int): tile index.
        use_z (list of int): list of z planes to calculate on.
        z_chunk_size (int): size of each z chunk.
        n_genes (int): the number of genes.
        transform (`[n_tiles x n_rounds x n_channels x 4 x 3] ndarray[float]`): `transform[t, r, c]` is the affine 
            transform to get from tile `t`, `ref_round`, `ref_channel` to tile `t`, round `r`, channel `c`.
        color_norm_factor (`[n_rounds x n_channels] ndarray[float]`): Normalisation factors to divide colours by to 
            equalise channel intensities.
        initial_intensity_thresh (float): pixel intensity threshold, only keep ones above the threshold to save memory 
            and storage space.
        bled_codes (`[n_genes x n_rounds x n_channels] ndarray[float]`): bled codes.
        dp_norm_shift (int or float): when finding `dot_product_score` between residual `pixel_colors` and 
            `bled_codes`, this is applied to normalisation of `pixel_colors` to limit boost of weak pixels.

    Returns:
        - (`[n_pixels x 3] ndarray[int]`): `pixel_yxz_t` is the y, x and z pixel positions of the gene coefficients 
            found.
        - (`[n_pixels x n_genes] scipy.sparse.csr_matrix`): `pixel_coefs_t` contains the gene coefficients for each 
            pixel.
    """
    pixel_yxz_t = np.zeros((0, 3), dtype=np.int16)
    pixel_coefs_t = scipy.sparse.csr_matrix(np.zeros((0, n_genes), dtype=np.float32))
    
    z_chunks = len(use_z) // z_chunk_size + 1
    for z_chunk in range(z_chunks):
        print(f"z_chunk {z_chunk + 1}/{z_chunks}")
        # While iterating through tiles, only save info for rounds/channels using
        # - add all rounds/channels back in later. This returns colors in use_rounds/channels only and no invalid.
        pixel_yxz_tz, pixel_colors_tz = coefs.get_pixel_colours(nbp_basic, nbp_file, nbp_extract, int(tile), z_chunk, 
                                                                z_chunk_size, np.asarray(transform), 
                                                                np.asarray(color_norm_factor))

        pixel_yxz_tz = jnp.asarray(pixel_yxz_tz, dtype=jnp.int16)
        pixel_colors_tz = jnp.asarray(pixel_colors_tz, dtype=jnp.float32)
        # Only keep pixels with significant absolute intensity to save memory.
        # absolute because important to find negative coefficients as well.
        pixel_intensity_tz = call_spots.get_spot_intensity(jnp.abs(pixel_colors_tz))
        keep = pixel_intensity_tz > initial_intensity_thresh
        if not keep.any():
            continue
        pixel_colors_tz = pixel_colors_tz[keep]
        pixel_yxz_tz = pixel_yxz_tz[keep]
        del pixel_intensity_tz, keep

        pixel_coefs_tz = get_all_coefs(pixel_colors_tz, bled_codes, 0, dp_norm_shift, config['dp_thresh'], config['alpha'], 
                                    config['beta'], config['max_genes'], config['weight_coef_fit'])[0]
        pixel_coefs_tz = np.asarray(pixel_coefs_tz, dtype=np.float32)
        del pixel_colors_tz
        # Only keep pixels for which at least one gene has non-zero coefficient.
        keep = (np.abs(pixel_coefs_tz).max(axis=1) > 0).nonzero()[0]  # nonzero as is sparse matrix.
        if len(keep) == 0:
            continue
        
        pixel_yxz_t = np.append(pixel_yxz_t, np.asarray(pixel_yxz_tz[keep]), axis=0)
        pixel_coefs_t = scipy.sparse.vstack((pixel_coefs_t, scipy.sparse.csr_matrix(pixel_coefs_tz[keep])))
        del pixel_yxz_tz, pixel_coefs_tz, keep

    return pixel_yxz_t, pixel_coefs_t
