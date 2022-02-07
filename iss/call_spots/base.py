import numpy as np
from .. import utils


def color_normalisation(hist_values, hist_counts, thresh_intensities, thresh_probs, method):
    """
    This finds the normalisations for each round, r,  and channel, c, such that if
    norm_spot_color[r,c] = spot_color[r,c] / norm_factor[r,c], the probability of norm_spot_color being larger than
    thresh_intensities[i] is less than thresh_probs[i] for every i.
    Where the probability is based on all pixels from all tiles in that round and channel.

    :param hist_values: numpy integer array [n_pixel_values]
        all possible pixel values in saved tiff images i.e. n_pixel_values is approximately np.iinfo(np.uint16).max
        because tiffs saved as uint16 images.
    :param hist_counts: numpy integer array [n_pixel_values x n_rounds x n_channels]
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
                bled_codes[g, r, c] = bleed_matrix[r, c, gene_codes[g, r]]

    return bled_codes


def dot_product(data_vectors, cluster_vectors, norm_axis=None):
    """
    Will normalise both data_vectors and cluster_vectors and then find the dot product between each vector in
    data_vectors with each vector in cluster_vectors.

    :param data_vectors: numpy float array [n_data x ax1_dim x ax2_dim x ... x axN_dim]
    :param cluster_vectors: numpy float array [n_clusters x ax1_dim x ax2_dim x ... x axN_dim]
    :param norm_axis: integer or tuple of integers, optional
        which axis to sum over for normalisation
        e.g. consider example where data_vectors shape is [800 x 5 x 10]
            norm_axis = (1,2): normalisation will sum over both axis so maximum possible dot product is 1.
            norm_axis = 1: normalisation will sum over axis 1 so maximum possible dot product is 10.
            norm_axis = 2: normalisation will sum over axis 2 so maximum possible dot product is 5.
        default is summing over all axis i.e. (1,...,N).
    :return:
        numpy float array [n_data x n_clusters] giving dot product for each data vector with each cluster vector
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
    return np.matmul(np.reshape(norm_data_vectors, (n_data, -1)),
                     np.reshape(norm_cluster_vectors, (n_clusters, -1)).transpose())


def get_spot_intensity(spot_colors):
    """
    finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.
    Logic is that we expect spots that are gene to have at least one large intensity value on each round
    so high spot intensity is more indicative of a gene.

    :param spot_colors: numpy float array [n_spots x n_rounds x n_channels]
    :return: numpy float array [n_spots]
    """
    diff_to_int = np.round(spot_colors[~np.isnan(spot_colors)]).astype(int)-spot_colors[~np.isnan(spot_colors)]
    if np.abs(diff_to_int).max() == 0:
        raise ValueError("spot_intensities should be found using normalised spot_colors. "
                         "\nBut all values in spot_colors given are integers indicating they are the raw intensities.")
    round_max_color = np.nanmax(spot_colors, axis=2)
    return np.nanmedian(round_max_color, axis=1)
