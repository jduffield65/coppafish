import numpy as np
from .. import utils


def color_normalisation(hist_values, hist_counts, thresh_intensities, thresh_probs, method):
    """
    This finds the normalisations for each round, r,  and channel, c, such that if
    norm_spot_color[r,c] = spot_color[r,c] / norm_factor[r,c], the probability of norm_spot_color being larger than
    thresh_intensities[i] is less than thresh_probs[i] for every i.
    Where the probability is based on all pixels from all tiles in that round and channel.

    :param hist_values: numpy integer array [n_pixels]
        all possible pixel values in saved tiff images i.e. n_pixels is approximately np.iinfo(np.uint16).max
        because tiffs saved as uint16 images.
    :param hist_counts: numpy integer array [n_pixels x n_rounds x n_channels]
        hist_counts[i, r, c] is the number of pixels across all tiles in round r, channel c
        which had the value hist_values[i].
    :param thresh_intensities: float, list or numpy array [n_thresholds]
        thresholds such that the probability of having a normalised spot_color greater than this are quite low.
        Need to be ascending.
        Typical: [0.5, 1, 5] i.e. we want most of normalised spot_colors to be less than 0.5 so high
        normalised spot color is on the order of 1.
    :param thresh_probs: float, list or numpy array [n_thresholds]
        probability of normalised spot color being greater than thresh_intensities[i] must be less than thresh_probs[i].
        Needs to be same shape as thresh_intensities and descending.
        Typical: [0.01, 5e-4, 1e-5] i.e. want almost all non spot pixels to have normalised intensity less than 0.5.
    :param method: string, 'single' or 'separate'
        'single': a single normalisation factor is produced for all rounds of each channel
            i.e. norm_factor[r, b] for a given b value, will be the same for all r values.
        'separate': a different normalisation factor is made for each round and channel.
    :return:
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
                prob = sum(hist_counts_rb[hist_values >= thresh_intensity*norm_factor_rb]) / n_pixels
                if prob > thresh_prob:
                    norm_factor_rb = hist_values[np.where(cum_sum_rb > (1 - thresh_prob) * n_pixels)[0][1]
                                     ] / thresh_intensity
            norm_factor[r, b] = norm_factor_rb
        if r_ind == 0 and method.lower() == 'single':
            break

    return norm_factor


def get_bled_codes(gene_codes, bleed_matrix):
    """
    this gets bled_codes such that the spot_color of a gene g in round r is expected to be a constant
    multiple of bled_codes[g, r, :].

    :param gene_codes: numpy integer array [n_genes x n_rounds]
        gene_codes[g, r] indicates the dye that should be present for gene g in round r.
    :param bleed_matrix: numpy float array [n_rounds x n_channels x n_dyes]
        expected intensity of dye d in round r is a constant multiple of bleed_matrix[r, :, d].
    :return: numpy float array [n_genes x n_rounds x n_channels]
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
                bled_codes[g, r, c] = bleed_matrix[r, c, gene_codes[r]]

    return bled_codes

# TODO: when doing dot product scores, have absolute dot product and then difference to second best but no standard dev.
