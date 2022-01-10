import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_transform(yxz_base, transform_old, yxz_target, dist_thresh, yxz_target_tree=None,
                  reg_constant_rot=30000, reg_constant_shift=9, reg_transform=None):
    """
    This finds the affine transform that transforms yxz_base such that the distances between the neighbours
    with yxz_target are minimised.

    :param yxz_base: numpy float array [n_base_spots x 3]
        coordinates of spots you want to transform.
    :param transform_old: numpy float array [4 x 3]
        affine transform found for previous iteration of PCR algorithm.
    :param yxz_target: numpy float array [n_target_spots x 3]
        coordinates of spots in image that you want to transform yxz_base to.
    :param dist_thresh: float
         if neighbours closer than this, they are used to compute the new transform.
         typical = 3
    :param yxz_target_tree: sklearn NearestNeighbours object, optional.
        KDTree produced from yxz_target. If None, it will be computed.
        default: None.
    :param reg_constant_rot: float, optional
        constant used for scaling and rotation when doing regularised least squares.
        default: 30000
    :param reg_constant_shift: float
        constant used for shift when doing regularised least squares.
        default: 9
    :param reg_transform: numpy float array [4 x 3], optional
        affine transform which we want final transform to be near when doing regularised least squares.
        If None, then no regularisation is performed.
        default: None
    :return:
        transform: numpy float array [4 x 3]. Updated affine transform
        neighbour: numpy integer array [n_base_spots,]
            neighbour[i] is index of coordinate in yxz_target to which transformation of yxz_base[i] is closest.
        n_matches: integer, number of neighbours which have distance less than dist_thresh.
        error: float, average distance between neighbours below dist_thresh.
    """
    if yxz_target_tree is None:
        yxz_target_tree = NearestNeighbors(n_neighbors=1).fit(yxz_target)
    yxz_base_pad = np.pad(yxz_base, [(0, 0), (0, 1)], constant_values=1)
    yxz_transform = np.matmul(yxz_base_pad, transform_old)
    distances, neighbour = yxz_target_tree.kneighbors(yxz_transform)
    neighbour = neighbour.flatten()
    distances = distances.flatten()
    use = distances < dist_thresh
    n_matches = sum(use)
    error = np.sqrt(np.mean(distances[use]**2))
    if reg_transform is None:
        transform = np.linalg.lstsq(yxz_base_pad[use, :], yxz_target[neighbour[use], :], rcond=None)[0]
    else:
        scale = np.array([reg_constant_rot, reg_constant_rot, reg_constant_rot, reg_constant_shift]).reshape(4, 1)
        yxz_base_regularised = np.concatenate((yxz_base_pad[use, :], np.eye(4)*scale), axis=0)
        yxz_target_regularised = np.concatenate((yxz_target[neighbour[use], :], reg_transform * scale), axis=0)
        transform = np.linalg.lstsq(yxz_base_regularised, yxz_target_regularised, rcond=None)[0]
    if sum(transform[2, :] == 0) == 3:
        transform[2, 2] = 1  # if 2d transform, set scaling of z to 1 still
    return transform, neighbour, n_matches, error
