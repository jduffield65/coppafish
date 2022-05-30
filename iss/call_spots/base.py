import warnings

import numpy as np
from .. import utils
from typing import Union, List, Optional, Tuple
from ..setup.notebook import NotebookPage
from line_profiler_pycharm import profile
from functools import partial
import jax.numpy as jnp
import jax


def color_normalisation(hist_values: np.ndarray, hist_counts: np.ndarray,
                        thresh_intensities: Union[float, List[float], np.ndarray],
                        thresh_probs: Union[float, List[float], np.ndarray], method: str) -> np.ndarray:
    """
    This finds the normalisations for each round, ```r```, and channel, ```c```, such that if ```norm_spot_color[r,c] =
    spot_color[r,c] / norm_factor[r,c]```, the probability of ```norm_spot_color``` being larger than ```
    thresh_intensities[i]``` is less than ```thresh_probs[i]``` for every ```i```.
    Where the probability is based on all pixels from all tiles in that round and channel.

    Args:
        hist_values: ```int [n_pixel_values]```.
            All possible pixel values in saved tiff images i.e. n_pixel_values is approximately
            ```np.iinfo(np.uint16).max``` because tiffs saved as ```uint16``` images.
        hist_counts: ```int [n_pixel_values x n_rounds x n_channels]```.
            ```hist_counts[i, r, c]``` is the number of pixels across all tiles in round ```r```, channel ```c```
            which had the value ```hist_values[i]```.
        thresh_intensities: ```float [n_thresholds]```.
            Thresholds such that the probability of having a normalised spot_color greater than this are quite low.
            Need to be ascending.
            Typical: ```[0.5, 1, 5]``` i.e. we want most of ```normalised spot_colors``` to be less than ```0.5``` so
            high normalised spot color is on the order of ```1```.
        thresh_probs: ```float [n_thresholds]```.
            Probability of normalised spot color being greater than ```thresh_intensities[i]``` must be less than
            ```thresh_probs[i]```. Needs to be same shape as thresh_intensities and descending.
            Typical: ```[0.01, 5e-4, 1e-5]``` i.e. want almost all non spot pixels to have
            normalised intensity less than ```0.5```.
        method: Must be one of the following:

            - ```'single'``` - A single normalisation factor is produced for all rounds of each channel
                i.e. ```norm_factor[r, b]``` for a given ```b``` value, will be the same for all ```r``` values.
            - ```'separate'``` - A different normalisation factor is made for each round and channel.

    Returns:
        ```float [n_rounds x n_channels]```.
            ```norm_factor``` such that ```norm_spot_color[s,r,c] = spot_color[s,r,c] / norm_factor[r,c]```.
    """
    if not utils.errors.check_shape(hist_values, hist_counts.shape[:1]):
        raise utils.errors.ShapeError('hist_values', hist_values.shape, hist_counts.shape[:1])
    # if only one value provided, turn to a list
    if isinstance(thresh_intensities, (float, int)):
        thresh_intensities = [thresh_intensities]
    if isinstance(thresh_probs, (float, int)):
        thresh_probs = [thresh_probs]
    if not utils.errors.check_shape(np.array(thresh_intensities), np.array(thresh_probs).shape):
        raise utils.errors.ShapeError('thresh_intensities', np.array(thresh_intensities).shape,
                                      np.array(thresh_probs).shape)

    # sort thresholds and check that thresh_probs descend as thresh_intensities increase
    ind = np.argsort(thresh_intensities)
    thresh_intensities = np.array(thresh_intensities)[ind]
    thresh_probs = np.array(thresh_probs)[ind]
    if not np.all(np.diff(thresh_probs) <= 0):
        raise ValueError(f"thresh_probs given, {thresh_probs}, do not all descend as thresh_intensities,"
                         f" {thresh_intensities}, increase.")

    n_rounds, n_channels = hist_counts.shape[1:]
    norm_factor = np.zeros((n_rounds, n_channels))
    for r_ind in range(n_rounds):
        if method.lower() == 'single':
            r = np.arange(n_rounds)
        elif method.lower() == 'separate':
            r = r_ind
        else:
            raise ValueError(f"method given was {method} but should be either 'single' or 'separate'")
        for b in range(n_channels):
            hist_counts_rb = np.sum(hist_counts[:, r, b].reshape(hist_values.shape[0], -1), axis=1)
            cum_sum_rb = np.cumsum(hist_counts_rb)
            n_pixels = cum_sum_rb[-1]
            norm_factor_rb = -np.inf
            for thresh_intensity, thresh_prob in zip(thresh_intensities, thresh_probs):
                prob = sum(hist_counts_rb[hist_values >= thresh_intensity * norm_factor_rb]) / n_pixels
                if prob > thresh_prob:
                    norm_factor_rb = hist_values[np.where(cum_sum_rb > (1 - thresh_prob) * n_pixels)[0][1]
                                     ] / thresh_intensity
            norm_factor[r, b] = norm_factor_rb
        if r_ind == 0 and method.lower() == 'single':
            break

    return norm_factor


def get_bled_codes(gene_codes: np.ndarray, bleed_matrix: np.ndarray) -> np.ndarray:
    """
    This gets ```bled_codes``` such that the spot_color of a gene ```g``` in round ```r``` is expected to be a constant
    multiple of ```bled_codes[g, r]```.
    This function should be run with full bleed_matrix with any rounds/channels/dyes outside those using set to nan.
    Otherwise will get confusion with dye indices in `gene_codes` being outside size of `bleed_matrix`.

    !!! note
        All bled_codes returned with an L2 norm of 1 when summed over all rounds and channels
        with any nan values assumed to be 0.

    Args:
        gene_codes: ```int [n_genes x n_rounds]```.
            ```gene_codes[g, r]``` indicates the dye that should be present for gene ```g``` in round ```r```.
        bleed_matrix: ```float [n_rounds x n_channels x n_dyes]```.
            Expected intensity of dye ```d``` in round ```r``` is a constant multiple of ```bleed_matrix[r, :, d]```.

    Returns:
        ```float [n_genes x n_rounds x n_channels]```.
            ```bled_codes``` such that ```spot_color``` of a gene ```g```
            in round ```r``` is expected to be a constant multiple of ```bled_codes[g, r]```.
    """
    n_genes = gene_codes.shape[0]
    n_rounds, n_channels, n_dyes = bleed_matrix.shape
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
                bled_codes[g, r, c] = bleed_matrix[r, c, gene_codes[g, r]]

    # Give all bled codes an L2 norm of 1 assuming any nan values are 0
    norm_factor = np.expand_dims(np.linalg.norm(np.nan_to_num(bled_codes), axis=(1, 2)), (1, 2))
    norm_factor[norm_factor == 0] = 1   # For genes with no dye in any rounds, this avoids blow up on next line
    bled_codes = bled_codes / norm_factor
    return bled_codes


def dot_product_score_jax(spot_colors, bled_codes, norm_shift, weight_squared):
    n_genes, n_round_channels = bled_codes.shape
    spot_colors = spot_colors / (jnp.linalg.norm(spot_colors) + norm_shift)
    bled_codes = bled_codes / jnp.linalg.norm(bled_codes, axis=1, keepdims=True)
    spot_colors = spot_colors * weight_squared
    score = spot_colors @ bled_codes.transpose()
    score = score / jnp.sum(weight_squared) * n_round_channels
    return score


@partial(jax.jit, static_argnums=2)
def dot_product_score_jax_vectorised(spot_colors, bled_codes, norm_shift, weight_squared):
    score = jax.vmap(dot_product_score_jax, in_axes=(0, None, None, 0), out_axes=0)(spot_colors, bled_codes,
                                                                                    norm_shift, weight_squared)
    return score


def dot_product_score(spot_colors: np.ndarray, bled_codes: np.ndarray, norm_shift: float = 0,
                      weight_squared: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Computes `sum(W**2(s * b) / W**2)` where `s` is a `spot_color`, `b` is a `bled_code` and `W**2` is weight_squared
    for a particular `spot_color`. Sum is over all rounds and channels.

    Args:
        spot_colors: `float [n_spots x (n_rounds x n_channels)]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        weight_squared: `float [n_spots x (n_rounds x n_channels)]`.
            squared weight to apply to each round/channel for each spot when computing dot product.
            If `None`, all rounds, channels treated equally.

    Returns:
        `float [n_spots x n_genes]`.
            `score` such that `score[d, c]` gives dot product between `spot_colors` vector `d`
            with `bled_codes` vector `c`.
    """
    # TODO: accept no nan values in spot_colors or bled_codes
    n_spots = spot_colors.shape[0]
    n_genes, n_round_channels = bled_codes.shape
    if not utils.errors.check_shape(spot_colors[0], bled_codes[0].shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      (n_spots, n_round_channels))
    spot_norm_factor = np.linalg.norm(spot_colors, axis=1, keepdims=True)
    spot_norm_factor = spot_norm_factor + norm_shift
    spot_colors = spot_colors / spot_norm_factor

    gene_norm_factor = np.linalg.norm(bled_codes, axis=1, keepdims=True)
    gene_norm_factor[gene_norm_factor == 0] = 1  # so don't blow up if bled_code is all 0 for a gene.
    bled_codes = bled_codes / gene_norm_factor

    if weight_squared is not None:
        if not utils.errors.check_shape(weight_squared, spot_colors.shape):
            raise utils.errors.ShapeError('weight', weight_squared.shape,
                                          spot_colors.shape)
        spot_colors = spot_colors * weight_squared

    # TODO: matmul replace by @
    score = spot_colors @ bled_codes.transpose()

    if weight_squared is not None:
        score = score / np.expand_dims(np.sum(weight_squared, axis=1), 1)
        score = score * n_round_channels  # make maximum score 1 if all weight the same and dot product perfect.

    return score


def get_spot_intensity(spot_colors: np.ndarray) -> np.ndarray:
    """
    Finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.
    Logic is that we expect spots that are genes to have at least one large intensity value in each round
    so high spot intensity is more indicative of a gene.

    Args:
        spot_colors: ```float [n_spots x n_rounds x n_channels]```.
            Spot colors normalised to equalise intensities between channels (and rounds).

    Returns:
        ```float [n_spots]```.
            ```[s]``` is the intensity of spot ```s```.
    """
    diff_to_int = np.round(spot_colors).astype(int) - spot_colors
    if np.abs(diff_to_int).max() == 0:
        raise ValueError("spot_intensities should be found using normalised spot_colors. "
                         "\nBut all values in spot_colors given are integers indicating they are the raw intensities.")
    round_max_color = np.max(spot_colors, axis=2)
    return np.median(round_max_color, axis=1)


@jax.jit
def get_spot_intensity_vectorised(spot_colors: jnp.array) -> float:
    """
    Finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.
    Logic is that we expect spots that are genes to have at least one large intensity value in each round
    so high spot intensity is more indicative of a gene.

    Args:
        spot_colors: ```float [n_spots x n_rounds x n_channels]```.
            Spot colors normalised to equalise intensities between channels (and rounds).

    Returns:
        ```float [n_spots]```.
            ```[s]``` is the intensity of spot ```s```.
    """
    return jax.vmap(lambda x: jnp.median(jnp.max(x, axis=1)), in_axes=0, out_axes=0)(spot_colors)


def fit_background(spot_colors: np.ndarray, weight_shift: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This determines the coefficient of the background vectors for each spot.
    Coefficients determined using a weighted dot product as to avoid overfitting
    and accounting for the fact that background coefficients are not updated after this.

    !!! note
        `background_vectors[i]` is 1 in channel `i` for all rounds and 0 otherwise.
        It is then normalised to have L2 norm of 1 when summed over all rounds and channels.

    Args:
        spot_colors: `float [n_spots x n_rounds x n_channels]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        weight_shift: shift to apply to weighting of each background vector to limit boost of weak spots.

    Returns:
        - residual - `float [n_spots x n_rounds x n_channels]`.
            `spot_colors` after background removed.
        - coef - `float [n_spots, n_channels]`.
            coefficient value for each background vector found for each spot.
        - background_vectors `float [n_channels x n_rounds x n_channels]`.
            background_vectors[c] is the background vector for channel c.

    """
    if weight_shift < 1e-20:
        warnings.warn(f'weight_shift value given, {weight_shift} is below 1e-20.'
                      f'Using weight_shift=1e-20 to stop blow up to infinity.')
    weight_shift = np.clip(weight_shift, 1e-20, np.inf)  # ensure weight_shift > 1e-20 to avoid blow up to infinity.

    n_rounds, n_channels = spot_colors[0].shape
    background_vectors = np.repeat(np.expand_dims(np.eye(n_channels), axis=1), n_rounds, axis=1)
    # give background_vectors an L2 norm of 1 so can compare coefficients with other genes.
    background_vectors = background_vectors / np.linalg.norm(background_vectors, axis=(1, 2), keepdims=True)

    weight_factor = 1 / (np.abs(spot_colors) + weight_shift)
    spot_weight = spot_colors * weight_factor
    background_weight = np.ones((1, n_rounds, n_channels)) * background_vectors[0, 0, 0] * weight_factor
    coef = np.sum(spot_weight * background_weight, axis=1) / np.sum(background_weight ** 2, axis=1)
    residual = spot_colors - np.expand_dims(coef, 1) * np.ones((1, n_rounds, n_channels)) * background_vectors[0, 0, 0]

    # # Old method, about 10x slower
    # n_spots = spot_colors.shape[0]
    # coef = np.zeros([n_spots, n_channels])
    # background_contribution = np.zeros_like(spot_colors)
    # background_vectors = np.zeros([n_channels, n_rounds, n_channels])
    # for c in range(n_channels):
    #     weight_factor = np.zeros([n_spots, n_rounds])
    #     for r in range(n_rounds):
    #         weight_factor[:, r] = 1 / (abs(spot_colors[:, r, c]) + weight_shift)
    #     weight_factor = np.expand_dims(weight_factor, 2)
    #
    #     background_vector = np.zeros([1, n_rounds, n_channels])
    #     background_vector[:, :, c] = 1
    #     # give background_vector an L2 norm of 1 so can compare coefficients with other genes.
    #     background_vector = background_vector / np.expand_dims(np.linalg.norm(background_vector, axis=(1, 2)), (1, 2))
    #     background_vectors[c] = background_vector
    #
    #     background_weight = background_vector * weight_factor
    #     spot_weight = spot_colors * weight_factor
    #
    #     coef[:, c] = np.sum(spot_weight * background_weight, axis=(1, 2)
    #     ) / np.sum(background_weight ** 2, axis=(1, 2))
    #     background_contribution[:, :, c] = np.expand_dims(coef[:, c], 1) * background_vector[0, 0, c]
    #
    # residual = spot_colors - background_contribution
    return residual, coef, background_vectors


def fit_background_jax(spot_color: jnp.ndarray, weight_shift: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    This determines the coefficient of the background vectors.
    Coefficients determined using a weighted dot product as to avoid over-fitting
    and accounting for the fact that background coefficients are not updated after this.

    !!! note
        `background_vectors[i]` is 1 in channel `i` for all rounds and 0 otherwise.
        It is then normalised to have L2 norm of 1 when summed over all rounds and channels.

    Args:
        spot_color: `float [n_rounds x n_channels]`.
            Spot color normalised to equalise intensities between channels (and rounds).
        weight_shift: shift to apply to weighting of each background vector to limit boost of weak spots.

    Returns:
        - residual - `float [n_rounds x n_channels]`.
            `spot_color` after background removed.
        - coefs - `float [n_channels]`.
            coefficient value for each background vector.
        - background_vectors `float [n_channels x n_rounds x n_channels]`.
            background_vectors[c] is the background vector for channel c.
    """
    n_rounds, n_channels = spot_color.shape
    background_vectors = jnp.repeat(jnp.expand_dims(jnp.eye(n_channels), axis=1), n_rounds, axis=1)
    # give background_vectors an L2 norm of 1 so can compare coefficients with other genes.
    background_vectors = background_vectors / jnp.linalg.norm(background_vectors, axis=(1, 2), keepdims=True)
    # array of correct shape containing the non-zero value of background_vectors everywhere.
    background_nz_value = jnp.full((n_rounds, n_channels), background_vectors[0, 0, 0])

    weight_factor = 1 / (jnp.abs(spot_color) + weight_shift)
    spot_weight = spot_color * weight_factor
    background_weight = background_nz_value * weight_factor
    coefs = jnp.sum(spot_weight * background_weight, axis=0) / jnp.sum(background_weight ** 2, axis=0)
    residual = spot_color - coefs * background_nz_value
    return residual, coefs, background_vectors


@partial(jax.jit, static_argnums=1)
def fit_background_jax_vectorised(spot_colors: jnp.ndarray,
                                  weight_shift: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    This determines the coefficient of the background vectors for each spot.
    Coefficients determined using a weighted dot product as to avoid overfitting
    and accounting for the fact that background coefficients are not updated after this.

    !!! note
        `background_vectors[i]` is 1 in channel `i` for all rounds and 0 otherwise.
        It is then normalised to have L2 norm of 1 when summed over all rounds and channels.

    Args:
        spot_colors: `float [n_spots x n_rounds x n_channels]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        weight_shift: shift to apply to weighting of each background vector to limit boost of weak spots.

    Returns:
        - residual - `float [n_spots x n_rounds x n_channels]`.
            `spot_colors` after background removed.
        - coef - `float [n_spots, n_channels]`.
            coefficient value for each background vector found for each spot.
        - background_vectors `float [n_channels x n_rounds x n_channels]`.
            background_vectors[c] is the background vector for channel c.
    """
    return jax.vmap(fit_background_jax, in_axes=(0, None), out_axes=(0, 0, None))(spot_colors, weight_shift)


def get_gene_efficiency(spot_colors: np.ndarray, spot_gene_no: np.ndarray, gene_codes: np.ndarray,
                        bleed_matrix: np.ndarray, min_spots: int = 30) -> np.ndarray:
    """
    `gene_efficiency[g,r]` gives the expected intensity of gene `g` in round `r` compared to that expected
    by the `bleed_matrix`. It is computed based on the average of all `spot_colors` assigned to that gene.

    Args:
        spot_colors: `float [n_spots x n_rounds x n_channels]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        spot_gene_no: `int [n_spots]`.
            Gene each spot was assigned to.
        gene_codes: `int [n_genes, n_rounds]`.
            `gene_codes[g, r]` indicates the dye that should be present for gene `g` in round `r`.
        bleed_matrix: `float [n_rounds x n_channels x n_dyes]`.
            For a spot, `s` matched to gene with dye `d` in round `r`, we expect `spot_colors[s, r]`",
            to be a constant multiple of `bleed_matrix[r, :, d]`"
        min_spots: If number of spots assigned to a gene less than or equal to this, `gene_efficiency[g]=1`
            for all rounds.

    Returns:
        `float [n_genes x n_rounds]`.
            `gene_efficiency[g,r]` gives the expected intensity of gene `g` in round `r` compared to that expected
            by the `bleed_matrix`.
    """
    # Check n_spots, n_rounds, n_channels, n_genes consistent across all variables.
    if not utils.errors.check_shape(spot_colors[0], bleed_matrix[:, :, 0].shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      (spot_colors.shape[0],) + bleed_matrix[:, :, 0].shape)
    if not utils.errors.check_shape(spot_colors[:, 0, 0], spot_gene_no.shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      spot_gene_no.shape + bleed_matrix[:, :, 0].shape)
    n_genes, n_rounds = gene_codes.shape
    if not utils.errors.check_shape(spot_colors[0, :, 0].squeeze(), gene_codes[0].shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      spot_gene_no.shape + (n_rounds,) + (bleed_matrix.shape[1],))

    gene_no_oob = [val for val in spot_gene_no if val < 0 or val >= n_genes]
    if len(gene_no_oob) > 0:
        raise utils.errors.OutOfBoundsError("spot_gene_no", gene_no_oob[0], 0, n_genes - 1)

    gene_efficiency = np.ones([n_genes, n_rounds])
    for g in range(n_genes):
        use = spot_gene_no == g
        if np.sum(use) > min_spots:
            round_strength = np.zeros([np.sum(use), n_rounds])
            for r in range(n_rounds):
                dye_ind = gene_codes[g, r]
                # below is equivalent to MATLAB spot_colors / bleed_matrix.
                round_strength[:, r] = np.linalg.lstsq(bleed_matrix[r, :, dye_ind:dye_ind + 1],
                                                       spot_colors[use, r].transpose(), rcond=None)[0]

            # find a reference round for each gene as that with median strength.
            av_round_strength = np.median(round_strength, 0)
            ref_round = np.abs(av_round_strength - np.median(av_round_strength)).argmin()

            # for each spot, find strength of each round relative to strength in
            # ref_round. Need relative strength not absolute strength
            # because expect spot color to be constant multiple of bled code.
            # So for all genes, gene_efficiency[g, ref_round] = 1 but ref_round is different between genes.

            # Only use spots whose strength in RefRound is positive.
            use = round_strength[:, ref_round] > 0
            if np.sum(use) > min_spots:
                relative_round_strength = round_strength[use] / np.expand_dims(round_strength[use, ref_round], 1)
                gene_efficiency[g] = np.median(relative_round_strength, 0)

    # set negative values to 0
    # TODO: maybe set a maximum value of gene efficiency so no one round can dominate too much.
    gene_efficiency = np.clip(gene_efficiency, 0, np.inf)
    return gene_efficiency


def quality_threshold(nbp: NotebookPage) -> np.ndarray:
    """

    Args:
        nbp:

    Returns:

    """
    if nbp.name == 'ref_spots':
        score = nbp.score
    elif nbp.name == 'omp':
        max_score = nbp.score_multiplier * np.sum(nbp.spot_shape == 1) + np.sum(nbp.spot_shape == -1)
        score = (nbp.score_multiplier * nbp.n_neighbours_pos + nbp.n_neighbours_neg) / max_score
    else:
        raise ValueError(f"Notebook page has name {nbp.name} but needs to be 'ref_spots' or 'omp'.")
    qual_ok = np.array([score > nbp.score_thresh, nbp.intensity > nbp.intensity_thresh]).all(axis=0)
    return qual_ok
