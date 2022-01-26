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
