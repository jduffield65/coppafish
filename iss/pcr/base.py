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


def transform_from_scale_shift(scale, shift):
    """

    :param scale: numpy float array [n_channels x n_dims]
        scale[c, d] is the scaling to account for chromatic aberration from reference channel
        to channel c for dimension d.
        typically as an initial guess all values in scale will be 1.
    :param shift: numpy float array [n_tiles x n_rounds x n_dims]
        shift[t, r, d] is the shift to account for the shift between the reference round for tile t and
        round r for tile t in dimension d.
    :return:
        numpy float array [dim+1 x dim x n_tiles x n_rounds x n_channels]
        gives affine transform from given scales and shifts
    """
    n_channels = scale.shape[0]
    n_tiles, n_rounds, dim = shift.shape
    transforms = np.zeros((dim+1, dim, n_tiles, n_rounds, n_channels))
    for t in range(n_tiles):
        for r in range(n_rounds):
            for c in range(n_channels):
                transforms[:dim, :, t, r, c] = np.eye(dim) * scale[c]
                transforms[dim, :, t, r, c] = shift[t, r]
    return transforms


def mod_median(array, ignore, axis=0):
    """
    this computes the median ignoring values indicated by ignore.

    :param array: numpy array
    :param ignore: numpy array of same size as array.
    :param axis: integer or integer list of what axis to average over
    :return: float or numpy array.
    """
    mod_array = array.copy()
    mod_array[ignore] = np.nan
    return np.nanmedian(mod_array, axis=axis)


def get_average_transform(transforms, n_matches, matches_thresh, scale_thresh, shift_thresh):
    """
    This finds all transforms which pass some thresholds and computes the average transform using them.
    average transform for tile t, round r, channel c, av_transforms[:, :, t, r, c] has:
        zero rotation
        scaling given by median for channel c over all tiles and rounds
            i.e. median(av_transforms[0, 0, :, :, c]) for y scaling.
        shift given by median for tile t, round r over all channels.
            i.e. median(av_transforms[4, 0, t, r, :]) for y shift if dim=3.

    :param transforms: numpy float array [dim+1 x dim x n_tiles x n_rounds x n_channels]
        transforms[:, :, t, r, c] is the affine transform for tile t from the reference image to round r, channel c.
    :param n_matches: numpy integer array [n_tiles x n_rounds x n_channels]
        number of matches found by point cloud registration
    :param matches_thresh: integer or numpy integer array [n_tiles x n_rounds x n_channels]
        matches for a particular transform must exceed this to be used when computing average transform
        can specify a single threshold for all transforms or a different threshold for each
        e.g. you may give a lower threshold if that tile/round/channel had less spots.
        typical = 200.
    :param scale_thresh: numpy float array [dim,]
        specifies by how much it is acceptable for the scaling to differ from the average scaling in each dimension.
        typically this threshold will be the same in all dimensions as expect chromatic aberration to be same in each.
        threshold should be fairly large, it is just to get rid of crazy scalings which sometimes get a lot of matches.
        typical = 0.01
    :param shift_thresh: numpy float array [dim,]
        specifies by how much it is acceptable for the shift to differ from the average shift in each dimension.
        typically this threshold will be the same in y and x but different in z.
        typical = 10 xy pixels in xy direction, 2 z pixels in z direction (normalised to have same units as xy pixels).
    :return:
        av_transforms: numpy float array [dim+1 x dim x n_tiles x n_rounds x n_channels]
        av_scaling: numpy float array [n_channels x dim]
        av_shifts: numpy float array [n_tiles x n_rounds x dim]
        failed: numpy boolean array [n_tiles x n_rounds x n_channels]
            indicates tiles/rounds/channels to which transform had too few matches or transform was anomalous compared
            to median. These were not included when calculating av_transforms.
        failed_non_matches: numpy boolean array [n_tiles x n_rounds x n_channels]
            indicates tiles/rounds/channels to which transform was anomalous compared to median either due to shift or
            scaling in one or more directions.
    """
    dim = transforms.shape[1]
    failed_matches = n_matches < matches_thresh
    failed = failed_matches.copy()

    scaling = transforms[np.arange(dim), np.arange(dim), :, :, :]
    av_scaling = mod_median(scaling,  np.expand_dims(failed, 0).repeat(dim, 0), axis=[1, 2])
    diff_to_av_scaling = np.abs(scaling - np.expand_dims(av_scaling, [1, 2]))
    failed_scale = np.max(diff_to_av_scaling - scale_thresh.reshape(dim, 1, 1, 1) > 0, axis=0)
    failed = np.logical_or(failed, failed_scale)

    shifts = transforms[3]
    av_shifts = mod_median(shifts, np.expand_dims(failed, 0).repeat(dim, 0), axis=3)
    diff_to_av_shift = np.abs(shifts - np.expand_dims(av_shifts, 3))
    failed_shift = np.max(diff_to_av_shift - shift_thresh.reshape(dim, 1, 1, 1), axis=0) > 0
    failed = np.logical_or(failed, failed_shift)

    # find average shifts and scaling again using final failed array
    av_scaling = mod_median(scaling, np.expand_dims(failed, 0).repeat(dim, 0), axis=[1, 2])
    av_shifts = mod_median(shifts, np.expand_dims(failed, 0).repeat(dim, 0), axis=3)
    all_failed_scale_c = np.unique(np.argwhere(np.isnan(av_scaling))[:, 1:], axis=0)
    n_failed = len(all_failed_scale_c)
    if n_failed > 0:
        # to compute median scale to particular channel, at least one good tile/round.
        raise ValueError(f"\nNo suitable scales found for the following channels across all tiles/rounds\n"
                         f"{[all_failed_scale_c[i][0] for i in range(n_failed)]}")
    all_failed_shifts_tr = np.unique(np.argwhere(np.isnan(av_shifts))[:, 1:], axis=0)
    n_failed = len(all_failed_shifts_tr[:, 0])
    if n_failed > 0:
        # to compute median shift to particular tile/round, at least one good channel is required.
        raise ValueError(f"\nNo suitable shifts found for the following tile/round combinations"
                         f" across all colour channels\n"
                         f"t: {[all_failed_shifts_tr[i, 0] for i in range(n_failed)]}\n"
                         f"r: {[all_failed_shifts_tr[i, 1] for i in range(n_failed)]}")

    av_scaling = np.moveaxis(av_scaling, 0, -1)  # so get in order channel,dim
    av_shifts = np.moveaxis(av_shifts, 0, -1)  # so get in order tile,round,dim
    av_transforms = transform_from_scale_shift(av_scaling, av_shifts)
    # indicates tiles/rounds/channels which have anomalous transform compared to median independent of number of matches
    failed_non_matches = np.logical_or(failed_scale, failed_shift)
    return av_transforms, av_scaling, av_shifts, failed, failed_non_matches



#    return av_transforms, av_scaling, failed, failed_transform