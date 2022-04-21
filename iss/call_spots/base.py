import warnings

import numpy as np
from .. import utils
from typing import Union, List, Optional, Tuple


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
    multiple of ```bled_codes[g, r, :]```.

    Args:
        gene_codes: ```int [n_genes x n_rounds]```.
            ```gene_codes[g, r]``` indicates the dye that should be present for gene ```g``` in round ```r```.
        bleed_matrix: ```float [n_rounds x n_channels x n_dyes]```.
            Expected intensity of dye ```d``` in round ```r``` is a constant multiple of ```bleed_matrix[r, :, d]```.

    Returns:
        ```float [n_genes x n_rounds x n_channels]```. ```bled_codes``` such that ```spot_color``` of a gene ```g```
            in round ```r``` is expected to be a constant multiple of ```bled_codes[g, r]```.
    """
    # ```bled_codes``` such that ```spot_color``` of a gene ```
    #         g``` in round ```r``` is expected to be a constant multiple of ```bled_codes[g, r, :]```.
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

    return bled_codes


def dot_product(data_vectors: np.ndarray, cluster_vectors: np.ndarray,
                norm_axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    """
    Will normalise both ```data_vectors``` and ```cluster_vectors``` and then find the dot product between each vector
    in ```data_vectors``` with each vector in ```cluster_vectors```.

    Args:
        data_vectors: ```float [n_data x ax1_dim x ax2_dim x ... x axN_dim]```.
            Data vectors to find dot product for.
        cluster_vectors: ```float [n_clusters x ax1_dim x ax2_dim x ... x axN_dim]```.
            Cluster vectors to find dot product with.
        norm_axis: Which axis to sum over for normalisation
            e.g. consider example where ```data_vectors``` shape is ```[800 x 5 x 10]```:

            - ```norm_axis = (1,2)```: normalisation will sum over both axis so maximum possible dot product is
                ```1```.
            - ```norm_axis = 1```: normalisation will sum over axis ```1``` so maximum possible dot product is
                ```10```.
            - ```norm_axis = 2```: normalisation will sum over axis ```2``` so maximum possible dot product is
                ```5```.

            If ```norm_axis=None```, summing over all axis i.e. ```(1,...,N)```.

    Returns:
        ```float [n_data x n_clusters]```.
            ```dot_product_score``` such that ```dot_product_score[d, c]``` gives dot product between data vector ```d```
            with cluster vector ```c```.
    """
    if not utils.errors.check_shape(data_vectors[0], cluster_vectors[0].shape):
        raise utils.errors.ShapeError('data_vectors', data_vectors.shape,
                                      data_vectors.shape[:1] + cluster_vectors[0].shape)
    if norm_axis is None:
        norm_axis = tuple(np.arange(data_vectors.ndim))[1:]
    data_vectors_intensity = np.sqrt(np.nansum(data_vectors ** 2, axis=norm_axis))
    norm_data_vectors = data_vectors / np.expand_dims(data_vectors_intensity, norm_axis)
    cluster_vectors_intensity = np.sqrt(np.nansum(cluster_vectors ** 2, axis=norm_axis))
    norm_cluster_vectors = cluster_vectors / np.expand_dims(cluster_vectors_intensity, norm_axis)

    # set nan values to 0.
    norm_data_vectors[np.isnan(norm_data_vectors)] = 0
    norm_cluster_vectors[np.isnan(norm_cluster_vectors)] = 0

    n_data = np.shape(data_vectors)[0]
    n_clusters = np.shape(cluster_vectors)[0]
    # TODO: matmul replace by @
    return np.matmul(np.reshape(norm_data_vectors, (n_data, -1)),
                     np.reshape(norm_cluster_vectors, (n_clusters, -1)).transpose())


def dot_product_score(spot_colors: np.ndarray, bled_codes: np.ndarray, norm_shift: float = 0,
                      weight: Optional[np.ndarray] = None) -> np.ndarray:
    """

    Args:
        spot_colors: `float [n_spots x n_rounds x n_channels]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        weight: `float [n_spots x n_rounds x n_channels]`.
            weight to apply to each round/channel for each spot when computing dot product.
            If `None`, all rounds, channels treated equally.

    Returns:
        `float [n_spots x n_genes]`.
            `score` such that `score[d, c]` gives dot product between `spot_colors` vector `d`
            with `bled_codes` vector `c`.
    """
    # TODO: accept no nan values in spot_colors
    n_spots = spot_colors.shape[0]
    n_genes = bled_codes.shape[0]
    if not utils.errors.check_shape(spot_colors[0], bled_codes[0].shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      (n_spots,) + bled_codes[0].shape)
    spot_norm_factor = np.expand_dims(np.linalg.norm(spot_colors, axis=(1, 2)), (1, 2))
    spot_norm_factor = spot_norm_factor + norm_shift
    spot_colors = spot_colors / spot_norm_factor

    gene_norm_factor = np.expand_dims(np.linalg.norm(bled_codes, axis=(1, 2)), (1, 2))
    bled_codes = bled_codes / gene_norm_factor

    if weight is not None:
        if not utils.errors.check_shape(weight, spot_colors.shape):
            raise utils.errors.ShapeError('weight', weight.shape,
                                          spot_colors.shape)
        spot_colors = spot_colors * weight ** 2

    score = np.reshape(spot_colors, (n_spots, -1)) @ np.reshape(bled_codes, (n_genes, -1)).transpose()

    if weight is not None:
        score = score / np.expand_dims(np.sum(weight ** 2, axis=(1, 2)), 1)
        n_rounds, n_channels = spot_colors[0].shape
        score = score * n_rounds * n_channels  # make maximum score 1 if all weight the same and dot product perfect.

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
    diff_to_int = np.round(spot_colors[~np.isnan(spot_colors)]).astype(int) - spot_colors[~np.isnan(spot_colors)]
    if np.abs(diff_to_int).max() == 0:
        raise ValueError("spot_intensities should be found using normalised spot_colors. "
                         "\nBut all values in spot_colors given are integers indicating they are the raw intensities.")
    round_max_color = np.nanmax(spot_colors, axis=2)
    return np.nanmedian(round_max_color, axis=1)


def fit_background(spot_colors: np.ndarray, weight_shift: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    This determines the coefficient of the background vectors for each spot.
    Coefficients determined using a weighted dot product as to avoid overfitting
    and accounting for the fact that background coefficients are not updated after this.

    !!! note
        background vector `i` is 1 in channel `i` for all rounds and 0 otherwise.
        It is then normalised to have L2 norm of 1 when summed over all rounds and channels.

    Args:
        spot_colors: `float [n_spots x n_rounds x n_channels]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        weight_shift: shift to apply to weighting of each background vector to limit boost of weak spots.

    Returns:
        - residual - `float [n_spots x n_rounds x n_channels]`.
            `spot_colors` after background removed.
        - coef - ```float [n_spots, n_channels]```.
            coefficient value for each background vector found for each spot.
    """
    if weight_shift < 1e-20:
        warnings.warn(f'weight_shift value given, {weight_shift} is below 1e-20.'
                      f'Using weight_shift=1e-20 to stop blow up to infinity.')
    weight_shift = np.clip(weight_shift, 1e-20, np.inf)  # ensure weight_shift > 1e-20 to avoid blow up to infinity.

    n_spots, n_rounds, n_channels = spot_colors.shape
    coef = np.zeros([n_spots, n_channels])
    background_contribution = np.zeros_like(spot_colors)
    for c in range(n_channels):
        weight_factor = np.zeros([n_spots, n_rounds])
        for r in range(n_rounds):
            weight_factor[:, r] = 1 / (abs(spot_colors[:, r, c]) + weight_shift)
        weight_factor = np.expand_dims(weight_factor, 2)

        background_vector = np.zeros([1, n_rounds, n_channels])
        background_vector[:, :, c] = 1
        # give background_vector an L2 norm of 1 so can compare coefficients with other genes.
        background_vector = background_vector / np.expand_dims(np.linalg.norm(background_vector, axis=(1, 2)), (1, 2))

        background_weight = background_vector * weight_factor
        spot_weight = spot_colors * weight_factor

        coef[:, c] = np.sum(spot_weight * background_weight, axis=(1, 2)) / np.sum(background_weight ** 2, axis=(1, 2))
        background_contribution[:, :, c] = np.expand_dims(coef[:, c], 1) * background_vector[0, 0, c]

    residual = spot_colors - background_contribution
    return residual, coef


def get_gene_efficiency(spot_colors: np.ndarray, spot_gene_no: np.ndarray, gene_codes: np.ndarray,
                        bleed_matrix: np.ndarray, min_spots: int = 10) -> np.ndarray:
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
    gene_efficiency = np.clip(gene_efficiency, 0, np.inf)
    return gene_efficiency
