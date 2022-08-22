import numpy as np
from scipy.spatial import KDTree
from .. import utils
from tqdm import tqdm
from typing import Optional, Tuple, Union, List
import warnings


def get_transform(yxz_base: np.ndarray, transform_old: np.ndarray, yxz_target: np.ndarray, dist_thresh: float,
                  yxz_target_tree: Optional[KDTree] = None, reg_constant_scale: float = 30000,
                  reg_constant_shift: float = 9,
                  reg_transform: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    This finds the affine transform that transforms ```yxz_base``` such that the distances between the neighbours
    with ```yxz_target``` are minimised.

    Args:
        yxz_base: ```float [n_base_spots x 3]```.
            Coordinates of spots you want to transform.
        transform_old: ```float [4 x 3]```.
            Affine transform found for previous iteration of PCR algorithm.
        yxz_target: ```float [n_target_spots x 3]```.
            Coordinates of spots in image that you want to transform ```yxz_base``` to.
        dist_thresh: If neighbours closer than this, they are used to compute the new transform.
            Typical: ```3```.
        yxz_target_tree: KDTree produced from ```yxz_target```.
            If ```None```, it will be computed.
        reg_constant_scale: Constant used for scaling and rotation when doing regularized least squares.
        reg_constant_shift: Constant used for shift when doing regularized least squares.
        reg_transform: ```float [4 x 3]```.
            Affine transform which we want final transform to be near when doing regularized least squares.
            If ```None```, then no regularization is performed.

    Returns:
        - ```transform``` - ```float [4 x 3]```.
            Updated affine transform.
        - ```neighbour``` - ```int [n_base_spots]```.
            ```neighbour[i]``` is index of coordinate in ```yxz_target``` to which transformation of
            ```yxz_base[i]``` is closest.
        - ```n_matches``` - ```int```.
            Number of neighbours which have distance less than ```dist_thresh```.
        - ```error``` - ```float```.
            Average distance between ```neighbours``` below ```dist_thresh```.
    """
    if yxz_target_tree is None:
        yxz_target_tree = KDTree(yxz_target)
    yxz_base_pad = np.pad(yxz_base, [(0, 0), (0, 1)], constant_values=1)
    yxz_transform = yxz_base_pad @ transform_old
    distances, neighbour = yxz_target_tree.query(yxz_transform, distance_upper_bound=dist_thresh)
    neighbour = neighbour.flatten()
    distances = distances.flatten()
    use = distances < dist_thresh
    n_matches = np.sum(use)
    error = np.sqrt(np.mean(distances[use] ** 2))
    if reg_transform is None:
        transform = np.linalg.lstsq(yxz_base_pad[use, :], yxz_target[neighbour[use], :], rcond=None)[0]
    else:
        scale = np.array([reg_constant_scale, reg_constant_scale, reg_constant_scale, reg_constant_shift]).reshape(4, 1)
        yxz_base_regularised = np.concatenate((yxz_base_pad[use, :], np.eye(4) * scale), axis=0)
        yxz_target_regularised = np.concatenate((yxz_target[neighbour[use], :], reg_transform * scale), axis=0)
        transform = np.linalg.lstsq(yxz_base_regularised, yxz_target_regularised, rcond=None)[0]
    if np.sum(transform[2, :] == 0) == 3:
        transform[2, 2] = 1  # if 2d transform, set scaling of z to 1 still
    return transform, neighbour, n_matches, error


def transform_from_scale_shift(scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Gets ```[dim+1 x dim]``` affine transform from scale for each channel and shift for each tile/round.

    Args:
        scale: ```float [n_channels x n_dims]```.
            ```scale[c, d]``` is the scaling to account for chromatic aberration from reference channel
            to channel ```c``` for dimension ```d```.
            Typically, as an initial guess all values in scale will be ```1```.
        shift: ```float [n_tiles x n_rounds x n_dims]```.
            ```shift[t, r, d]``` is the shift to account for the shift between the reference round for tile ```t``` and
            round ```r``` for tile ```t``` in dimension ```d```.

    Returns:
        ```float [n_tiles x n_rounds x n_channels x dim+1 x dim]```.
            ```[t, r, c]``` is the affine transform for tile ```t```, round ```r```, channel ```c``` computed from
            ```scale[c]``` and ```shift[t, r]```.
    """
    n_channels = scale.shape[0]
    n_tiles, n_rounds, dim = shift.shape
    transforms = np.zeros((n_tiles, n_rounds, n_channels, dim + 1, dim))
    for t in range(n_tiles):
        for r in range(n_rounds):
            for c in range(n_channels):
                transforms[t, r, c, :dim, :, ] = np.eye(dim) * scale[c]
                transforms[t, r, c, dim, :] = shift[t, r]
    return transforms


def mod_median(array: np.ndarray, ignore: np.ndarray, axis: Union[int, List[int]] = 0) -> Union[float, np.ndarray]:
    """
    This computes the median ignoring values indicated by ```ignore```.

    Args:
        array: ```float [n_dim_1 x n_dim_2 x ... x n_dim_N]```.
            array to compute median from.
        ignore: ```bool [n_dim_1 x n_dim_2 x ... x n_dim_N]```.
            True for values in array that should not be used to compute median.
        axis: ```int [n_axis_av]```.
            Which axis to average over.

    Returns:
        Median value without using those values indicated by ```ignore```.
    """
    mod_array = array.copy()
    mod_array[ignore] = np.nan
    return np.nanmedian(mod_array, axis=axis)


def get_average_transform(transforms: np.ndarray, n_matches: np.ndarray, matches_thresh: Union[int, np.ndarray],
                          scale_thresh: np.ndarray,
                          shift_thresh: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray]:
    """
    This finds all transforms which pass some thresholds and computes the average transform using them.
    `av_transforms[t, r, c]` is the average transform for tile `t`, round `r`, channel `c` and has:

    - Zero rotation.
    - Scaling given by median for channel `c` over all tiles and rounds.
        I.e. `median(av_transforms[:, :, c, 0, 0])` for y scaling.
    - shift given by median for tile `t`, round `r` over all channels.
        I.e. `median(av_transforms[t, r, _, 4, 0])` for y shift if `dim=3`.

    Args:
        transforms: ```float [n_tiles x n_rounds x n_channels x dim+1 x dim]```.
            ```transforms[t, r, c]``` is the affine transform for tile ```t``` from the reference image to
             round ```r```, channel ```c```.
        n_matches: ```int [n_tiles x n_rounds x n_channels]```.
            Number of matches found by point cloud registration.
        matches_thresh: ```int [n_tiles x n_rounds x n_channels]``` or single ```int```.
            ```n_matches``` for a particular transform must exceed this to be used when computing ```av_transforms```.
            Can specify a single threshold for all transforms or a different threshold for each.
            E.g. you may give a lower threshold if that tile/round/channel has fewer spots.
            Typical: ```200```.
        scale_thresh: ```float [n_dim]```.
            Specifies by how much it is acceptable for the scaling to differ from the average scaling in each dimension.
            Typically, this threshold will be the same in all dimensions as expect
            chromatic aberration to be same in each.
            Threshold should be fairly large, it is just to get rid of crazy scalings which sometimes
            get a lot of matches.
            Typical: `0.01`.
        shift_thresh: `float [n_dim]`.
            Specifies by how much it is acceptable for the shift to differ from the average shift in each dimension.
            Typically, this threshold will be the same in y and x but different in z.
            Typical: `10` xy pixels in xy direction, `2` z pixels in z direction
            (normalised to have same units as `yx_pixels`).

    Returns:
        - `av_transforms` - `float [n_tiles x n_rounds x n_channels x dim+1 x dim]`.
            `av_transforms[t, r, c]` is the average transform for tile `t`, round `r`, channel `c`.
        - `av_scaling` - `float [n_channels x dim]`.
            `av_scaling[c, d]` is the median scaling for channel `c`, dimension `d`, over all tiles and rounds.
        - `av_shifts` - `float [n_tiles x n_rounds x dim]`.
            `av_shifts[t, r, d]` is the median scaling for tile `t`, round `r`, dimension `d`, over all channels.
        - `failed` - `bool [n_tiles x n_rounds x n_channels]`.
            Indicates the tiles/rounds/channels to which transform had too few matches or transform was anomalous
            compared to median. These were not included when calculating `av_transforms`.
        - `failed_non_matches` - `bool [n_tiles x n_rounds x n_channels]`.
            Indicates the tiles/rounds/channels to which transform was anomalous compared to median either due to shift
            or scaling in one or more directions.
    """
    dim = transforms.shape[-1]
    failed_matches = n_matches < matches_thresh
    failed = failed_matches.copy()

    # Assume scaling is the same for particular channel across all tile and rounds
    scaling = transforms[:, :, :, np.arange(dim), np.arange(dim)]
    scaling = np.moveaxis(scaling, -1, 0)
    av_scaling = mod_median(scaling, np.expand_dims(failed, 0).repeat(dim, 0), axis=[1, 2])
    diff_to_av_scaling = np.abs(scaling - np.expand_dims(av_scaling, [1, 2]))
    failed_scale = np.max(diff_to_av_scaling - np.array(scale_thresh).reshape(dim, 1, 1, 1) > 0, axis=0)
    failed = np.logical_or(failed, failed_scale)

    # Assume shift the same for particular tile and round across all channels
    shifts = np.moveaxis(transforms[:, :, :, 3], -1, 0)
    av_shifts = mod_median(shifts, np.expand_dims(failed, 0).repeat(dim, 0), axis=3)
    diff_to_av_shift = np.abs(shifts - np.expand_dims(av_shifts, 3))
    failed_shift = np.max(diff_to_av_shift - np.array(shift_thresh).reshape(dim, 1, 1, 1), axis=0) > 0
    failed = np.logical_or(failed, failed_shift)

    # find average shifts and scaling again using final failed array
    av_scaling = mod_median(scaling, np.expand_dims(failed, 0).repeat(dim, 0), axis=[1, 2])
    av_shifts = mod_median(shifts, np.expand_dims(failed, 0).repeat(dim, 0), axis=3)
    all_failed_scale_c = np.unique(np.argwhere(np.isnan(av_scaling))[:, 1:], axis=0)
    n_failed = len(all_failed_scale_c)
    if n_failed > 0:
        # to compute median scale to particular channel, at least one good tile/round.
        raise ValueError(f"\nNo suitable scales found for the following channels across all tiles/rounds\n"
                         f"{[all_failed_scale_c[i][0] for i in range(n_failed)]}\n"
                         f"Consider removing these from use_channels.")
    all_failed_shifts_tr = np.unique(np.argwhere(np.isnan(av_shifts))[:, 1:], axis=0)
    n_failed = len(all_failed_shifts_tr[:, 0])
    if n_failed > 0:
        # to compute median shift to particular tile/round, at least one good channel is required.
        raise ValueError(f"\nNo suitable shifts found for the following tile/round combinations"
                         f" across all colour channels\n"
                         f"t: {[all_failed_shifts_tr[i, 0] for i in range(n_failed)]}\n"
                         f"r: {[all_failed_shifts_tr[i, 1] for i in range(n_failed)]}\n"
                         f"Look at the following diagnostics to see why registration has few matches for these:\n"
                         f"coppafish.plot.view_register_shift_info\ncoppafish.plot.view_register_search\n"
                         f"coppafish.plot.view_icp\n"
                         f"If it seems to be a single tile/round that is the problem, maybe remove from "
                         f"use_tiles/use_rounds and re-run.")

    av_scaling = np.moveaxis(av_scaling, 0, -1)  # so get in order channel,dim
    av_shifts = np.moveaxis(av_shifts, 0, -1)  # so get in order tile,round,dim
    av_transforms = transform_from_scale_shift(av_scaling, av_shifts)
    # indicates tiles/rounds/channels which have anomalous transform compared to median independent of number of matches
    failed_non_matches = np.logical_or(failed_scale, failed_shift)
    return av_transforms, av_scaling, av_shifts, failed, failed_non_matches


def icp(yxz_base: np.ndarray, yxz_target: np.ndarray, transforms_initial: np.ndarray, n_iter: int,
        dist_thresh: float, matches_thresh: Union[int, np.ndarray], scale_dev_thresh: np.ndarray,
        shift_dev_thresh: np.ndarray, reg_constant_scale: Optional[float] = None,
        reg_constant_shift: Optional[float] = None) -> Tuple[np.ndarray, dict]:
    """
    This gets the affine `transforms` from `yxz_base` to `yxz_target` using iterative closest point until
    all iterations used or convergence.
    For `transforms` that have matches below `matches_thresh` or are anomalous compared to `av_transform`,
    the `transforms` are recomputed using regularized least squares to ensure they are close to the `av_transform`.
    If either `reg_constant_rot = None` or `reg_constant_shift = None` then this is not done.

    Args:
        yxz_base: `object [n_tiles]`.
            `yxz_base[t]` is a numpy `float` array `[n_base_spots x dim]`.
            coordinates of spots on reference round of tile `t`.
        yxz_target: `object [n_tiles x n_rounds x n_channels]`.
            `yxz_target[t, r, c]` is a numpy `float` array `[n_target_spots x 3]`.
            coordinates of spots in tile `t`, round `r`, channel `c`.
        transforms_initial: `float [n_tiles x n_rounds x n_channels x dim+1 x dim]`.
            `transforms_initial[t, r, c]` is the starting affine transform for tile `t`
            from the reference image to round `r`, channel `c`.
            `transforms_initial[t, r, c, :dim, :dim]` is probably going to be the identity matrix.
            `transforms_initial[t, r, c, dim, :]` is the shift which needs to be pre-computed somehow to get a
            good result.
        n_iter: Max number of iterations to perform of ICP.
        dist_thresh: If neighbours closer than this, they are used to compute the new transform.
            Typical: `3`.
        matches_thresh: `int [n_tiles x n_rounds x n_channels]` or single `int`.
            `n_matches` for a particular transform must exceed this to be used when computing `av_transform`.
            Can specify a single threshold for all transforms or a different threshold for each.
            E.g. you may give a lower threshold if that tile/round/channel has fewer spots.
            Typical: `200`.
        scale_dev_thresh: `float [n_dim]`.
            Specifies by how much it is acceptable for the scaling to differ from the average scaling in each dimension.
            Typically, this threshold will be the same in all dimensions as expect chromatic aberration to be
            same in each.
            Threshold should be fairly large, it is just to get rid of crazy scalings which sometimes get
            a lot of matches.
            Typical: `0.01`.
        shift_dev_thresh: `float [n_dim]`.
            Specifies by how much it is acceptable for the shift to differ from the average shift in each dimension.
            Typically, this threshold will be the same in y and x but different in z.
            Typical: `10` xy pixels in xy direction, `2` z pixels in z direction
            (normalised to have same units as `yx_pixels`).
        reg_constant_scale: Constant used for scaling and rotation when doing regularized least squares.
            `None` means no regularized least squares performed.
            Typical = `5e8`.
        reg_constant_shift: Constant used for shift when doing regularized least squares.
            `None` means no regularized least squares performed.
            Typical = `500`

    Returns:
        - `transforms` - `float [n_tiles x n_rounds x n_channels x dim+1 x dim]`.
            `transforms[t, r, c]` is the final affine transform found for tile `t`, round `r`, channel `c`.
        - `debug_info` - `dict` containing 7 `np.ndarray` -

            - `n_matches` - `int [n_tiles x n_rounds x n_channels]`.
                Number of matches found for each transform.
            - `error` - `float [n_tiles x n_rounds x n_channels]`.
                Average distance between neighbours below `dist_thresh`.
            - `failed` - `bool [n_tiles x n_rounds x n_channels]`.
                Indicates tiles/rounds/channels to which transform had too few matches or transform was
                anomalous compared to median. These were not included when calculating `av_scalings` or `av_shifts`.
            - `is_converged` - `bool [n_tiles x n_rounds x n_channels]`.
                `False` if max iterations reached before transform converged.
            - `av_scaling` - `float [n_channels x n_dim]`.
                Chromatic aberration scaling factor to each channel from reference channel.
                Made using all rounds and tiles.
            - `av_shift` - `float [n_tiles x n_rounds x dim]`.
                `av_shift[t, r]` is the average shift from reference round to round `r` for tile `t` across all
                colour channels.
            - `transforms_outlier` - `float [n_tiles x n_rounds x n_channels x dim+1 x dim]`.
                `transforms_outlier[t, r, c]` is the final affine transform found for tile `t`, round `r`, channel `c`
                without regularization for `t`, `r`, `c` indicated by failed otherwise it is `0`.
    """
    n_tiles, n_rounds, n_channels = yxz_target.shape
    if not utils.errors.check_shape(yxz_base, [n_tiles]):
        raise utils.errors.ShapeError("yxz_base", yxz_base.shape, (n_tiles,))
    tree_target = np.zeros_like(yxz_target)
    for t in range(n_tiles):
        for r in range(n_rounds):
            for c in range(n_channels):
                tree_target[t, r, c] = KDTree(yxz_target[t, r, c])

    n_matches = np.zeros_like(yxz_target, dtype=int)
    error = np.zeros_like(yxz_target, dtype=float)
    neighbour = np.zeros_like(yxz_target)
    is_converged = np.zeros_like(yxz_target, dtype=bool)
    transforms = transforms_initial.copy().astype(float)
    transforms_outlier = np.zeros_like(transforms)
    finished_good_images = False
    av_transforms = None
    i_finished_good = 0
    i = 0
    with tqdm(total=n_tiles * n_rounds * n_channels) as pbar:
        pbar.set_description(f"Iterative Closest Point to find affine transforms")
        while i < n_iter:
            pbar.set_postfix({'iter': f'{i + 1}/{n_iter}', 'regularized': str(finished_good_images)})
            neighbour_last = neighbour.copy()
            for t in range(n_tiles):
                for r in range(n_rounds):
                    for c in range(n_channels):
                        if is_converged[t, r, c]:
                            continue
                        if finished_good_images:
                            reg_transform = av_transforms[t, r, c]
                        else:
                            reg_transform = None
                        transforms[t, r, c], neighbour[t, r, c], n_matches[t, r, c], error[t, r, c] = \
                            get_transform(yxz_base[t], transforms[t, r, c], yxz_target[t, r, c], dist_thresh,
                                          tree_target[t, r, c], reg_constant_scale, reg_constant_shift, reg_transform)
                        if i > i_finished_good:
                            is_converged[t, r, c] = np.abs(neighbour[t, r, c] - neighbour_last[t, r, c]).max() == 0
                            if is_converged[t, r, c]:
                                pbar.update(1)
            if (is_converged.all() and not finished_good_images) or (i == n_iter - 1 and not finished_good_images):
                av_transforms, av_scaling, av_shifts, failed, failed_non_matches = \
                    get_average_transform(transforms, n_matches, matches_thresh, scale_dev_thresh, shift_dev_thresh)
                if reg_constant_scale is not None and reg_constant_shift is not None:
                    # reset transforms of those that failed to average transform as starting point for
                    # regularised fitting
                    transforms_outlier[failed, :, :] = transforms[failed, :, :].copy()
                    transforms[failed, :, :] = av_transforms[failed, :, :]
                    is_converged[failed] = False
                    i = -1  # Allow n_iter to find regularized best transforms as well.
                    finished_good_images = True
                    pbar.update(-np.sum(failed.flatten()))
            i += 1
            if is_converged.all():
                break
    pbar.close()
    if i == n_iter:
        warnings.warn(f"Max number of iterations, {n_iter} reached but only "
                      f"{np.sum(is_converged)}/{np.sum(is_converged>=0)} transforms converged")

    debug_info = {'n_matches': n_matches, 'error': error, 'failed': failed, 'is_converged': is_converged,
                  'av_scaling': av_scaling, 'av_shifts': av_shifts, 'transforms_outlier': transforms_outlier}
    return transforms, debug_info


def get_single_affine_transform(spot_yxz_base: np.ndarray, spot_yxz_transform: np.ndarray, z_scale_base: float,
                                z_scale_transform: float, start_transform: np.ndarray,
                                neighb_dist_thresh: float, tile_centre: np.ndarray, n_iter: int = 100,
                                reg_constant_scale: Optional[float] = None, reg_constant_shift: Optional[float] = None,
                                reg_transform: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, float, bool]:
    """
    Finds the affine transform taking `spot_yxz_base` to `spot_yxz_transform`.

    Args:
        spot_yxz_base: Point cloud want to find the shift from.
            spot_yxz_base[:, 2] is the z coordinate in units of z-pixels.
        spot_yxz_transform: Point cloud want to find the shift to.
            spot_yxz_transform[:, 2] is the z coordinate in units of z-pixels.
        z_scale_base: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        z_scale_transform: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        start_transform: `float [4 x 3]`.
            Start affine transform for iterative closest point.
            Typically, `start_transform[:3, :]` is identity matrix and
            `start_transform[3]` is approx yxz shift (z shift in units of xy pixels).
        neighb_dist_thresh: Distance between 2 points must be less than this to be constituted a match.
        tile_centre: int [3].
            yxz coordinates of centre of image where spot_yxz found on.
        n_iter: Max number of iterations to perform of ICP.
        reg_constant_scale: Constant used for scaling and rotation when doing regularized least squares.
            `None` means no regularized least squares performed.
            Typical = `5e8`.
        reg_constant_shift: Constant used for shift when doing regularized least squares.
            `None` means no regularized least squares performed.
            Typical = `500`
        reg_transform: `float [4 x 3]`.
            Transform to regularize to when doing regularized least squares.
            `None` means no regularized least squares performed.

    Returns:
        - `transform` - `float [4 x 3]`.
            `transform` is the final affine transform found.
        - `n_matches` - Number of matches found for each transform.
        - `error` - Average distance between neighbours below `neighb_dist_thresh`.
        - `is_converged` - `False` if max iterations reached before transform converged.
    """

    spot_yxz_base = (spot_yxz_base - tile_centre) * [1, 1, z_scale_base]
    spot_yxz_transform = (spot_yxz_transform - tile_centre) * [1, 1, z_scale_transform]
    tree_transform = KDTree(spot_yxz_transform)
    neighbour = np.zeros(spot_yxz_base.shape[0], dtype=int)
    transform = start_transform.copy()
    for i in range(n_iter):
        neighbour_last = neighbour.copy()
        transform, neighbour, n_matches, error = \
            get_transform(spot_yxz_base, transform, spot_yxz_transform, neighb_dist_thresh,
                          tree_transform, reg_constant_scale, reg_constant_shift, reg_transform)

        is_converged = np.abs(neighbour - neighbour_last).max() == 0
        if is_converged:
            break
    return transform, n_matches, error, is_converged
