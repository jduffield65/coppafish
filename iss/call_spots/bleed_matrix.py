import numpy as np
from .. import utils


def scaled_k_means(x, initial_cluster_mean, score_thresh=0, min_cluster_size=10, n_iter=100):
    """
    does a clustering that minimizes the norm of x[i] - g[i] * cluster_mean{[cluster_ind[i]]}
    for each data point i in x, where g is the gain which is not explicitly computed

    :param x: data set of vectors to build cluster means from [n_points x n_dims]
    :param initial_cluster_mean: numpy float array [n_clusters x n_dims], starting point of mean cluster vectors
    :param score_thresh: float between 0 and 1, optional
        points in x with dot product to a cluster mean vector greater than this
        contribute to new estimate of mean vector.
        default: 0
    :param min_cluster_size: integer, optional
        if less than this many points assigned to a cluster, that cluster mean vector will be set to 0.
        default: 10
    :param n_iter: integer, optional
        maximum number of iterations performed.
        default: 100
    :return:
        norm_cluster_mean: numpy float array [n_clusters x n_dims], final normalised mean cluster vectors
        cluster_ind: numpy integer array [n_points]
            index of cluster each point was assigned to. -1 means fell below score_thresh and not assigned.
        cluster_eig_value: numpy float array [n_clusters], first eigenvalue of outer product matrix for each cluster.
    """
    # normalise starting points and original data
    norm_cluster_mean = initial_cluster_mean / np.linalg.norm(initial_cluster_mean, axis=1).reshape(-1, 1)
    x_norm = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
    n_clusters = initial_cluster_mean.shape[0]
    n_points, n_dims = x.shape
    cluster_ind = np.ones(x.shape[0], dtype=int) * -2  # set all to -2 so won't end on first iteration
    cluster_eig_val = np.zeros(n_clusters)

    if not utils.errors.check_shape(initial_cluster_mean, [n_clusters, n_dims]):
        raise utils.errors.ShapeError('initial_cluster_mean', initial_cluster_mean.shape, (n_clusters, n_dims))

    for i in range(n_iter):
        cluster_ind_old = cluster_ind.copy()

        # project each point onto each cluster. Use normalized so we can interpret score
        score = np.matmul(x_norm, norm_cluster_mean.transpose())
        cluster_ind = np.argmax(score, axis=1)  # find best cluster for each point
        top_score = score[np.arange(n_points), cluster_ind]
        top_score[np.where(np.isnan(top_score))[0]] = score_thresh-1  # don't include nan values
        cluster_ind[top_score < score_thresh] = -1  # unclusterable points

        if (cluster_ind == cluster_ind_old).all():
            break

        for c in range(n_clusters):
            my_points = x[cluster_ind == c]  # don't use normalized, to avoid overweighting weak points
            n_my_points = my_points.shape[0]
            if n_my_points < min_cluster_size:
                norm_cluster_mean[c] = 0
                continue
            eig_vals, eigs = np.linalg.eig(np.matmul(my_points.transpose(), my_points)/n_my_points)
            best_eig_ind = np.argmax(eig_vals)
            norm_cluster_mean[c] = eigs[:, best_eig_ind] * np.sign(eigs[:, best_eig_ind].mean())  # make them positive
            cluster_eig_val[c] = eig_vals[best_eig_ind]

    return norm_cluster_mean, cluster_ind, cluster_eig_val


def get_bleed_matrix(spot_colors, dye_channel_intensity, method, score_thresh=0, min_cluster_size=10, n_iter=100):
    """
    this returns a bleed matrix such that bleed_matrix[r, c, d] is the expected intensity for
    dye d in round r, channel c.

    :param spot_colors: numpy float array [n_spots x n_rounds x n_channels]
        intensity found for each spot in each round and channel, normalized in some way to equalize channel intensities
    :param dye_channel_intensity: numpy float array [n_dyes x n_channels]
        initial guess for intensity we expect each dye to produce in each channel. Should be normalized in same way as
        spot_colors.
    :param method: string, 'single' or 'separate'
        'single': a single bleed matrix is produced for all rounds
        'separate': a different bleed matrix is made for each round
    :param score_thresh: float between 0 and 1, optional
        threshold used for scaled_k_means affecting which spots contribute to bleed matrix estimate
        default: 0
    :param min_cluster_size: integer, optional
        if less than this many points assigned to a dye, that dye mean vector will be set to 0.
        default: 10
    :param n_iter: integer, optional
        maximum number of iterations performed in scaled_k_means
        default: 100
    :return: numpy float array [n_rounds x n_channels x n_dyes]
    """
    n_rounds, n_channels = spot_colors.shape[1:]
    n_dyes = dye_channel_intensity.shape[0]
    if not utils.errors.check_shape(dye_channel_intensity, [n_dyes, n_channels]):
        raise utils.errors.ShapeError('initial_cluster_mean', dye_channel_intensity.shape, (n_dyes, n_channels))

    bleed_matrix = np.zeros((n_rounds, n_channels, n_dyes))  # Round, Measured, Real
    if method.lower() == 'separate':
        for r in range(n_rounds):
            spot_channel_intensity = spot_colors[:, r, :]
            # get rid of any nan codes
            spot_channel_intensity = spot_channel_intensity[~np.isnan(spot_channel_intensity).any(axis=1)]
            dye_codes, _, dye_eig_vals = scaled_k_means(spot_channel_intensity, dye_channel_intensity, score_thresh,
                                                        min_cluster_size, n_iter)
            for d in range(n_dyes):
                bleed_matrix[r, :, d] = dye_codes[d] * np.sqrt(dye_eig_vals[d])
    elif method.lower() == 'single':
        spot_channel_intensity = spot_colors.reshape(-1, n_channels)
        # get rid of any nan codes
        spot_channel_intensity = spot_channel_intensity[~np.isnan(spot_channel_intensity).any(axis=1)]
        dye_codes, _, dye_eig_vals = scaled_k_means(spot_channel_intensity, dye_channel_intensity, score_thresh,
                                                    min_cluster_size, n_iter)
        for r in range(n_rounds):
            for d in range(n_dyes):
                bleed_matrix[r, :, d] = dye_codes[d] * np.sqrt(dye_eig_vals[d])
    else:
        raise ValueError(f"method given was {method} but should be either 'single' or 'separate'")
    return bleed_matrix
