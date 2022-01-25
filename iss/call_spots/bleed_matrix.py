import numpy as np
from .. import utils


def scaled_k_means(x, initial_cluster_mean, score_thresh=0, min_cluster_size=10, n_iter=100):
    """
    does a clustering that minimizes the norm of x[i] - g[i] * cluster_mean{[cluster_ind[i]]}
    for each data point i in x, where g is the gain which is not explicitly computed

    :param x: data set of vectors to build cluster means from [n_points x n_dims]
    :param initial_cluster_mean: starting point of mean cluster vectors [n_clusters x n_dims]
    :param score_thresh: float between 0 and 1, optional
        points in x with dot product to a cluster mean vector greater than this
        contribute to new estimate of mean vector.
    :param min_cluster_size: integer, optional
        if less than this many points assigned to a cluster, that cluster mean vector will be set to 0.
        default: 10
    :param n_iter: integer, maximum number of iterations performed.
    :return:
    """
    # TODO: set cluster_ind to -1 if below score_thresh
    # normalise starting points and original data
    norm_cluster_mean = initial_cluster_mean / np.linalg.norm(initial_cluster_mean, axis=1).reshape(-1,1)
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
            norm_cluster_mean[c] = eigs[:, 0] * np.sign(eigs[:,0].mean())  # make them positive
            cluster_eig_val[c] = eig_vals[0]

    return norm_cluster_mean, cluster_ind, cluster_eig_val
