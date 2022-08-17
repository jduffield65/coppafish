import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import pairwise_distances
from ..utils.base import setdiff2d
from typing import Tuple, Optional, List, Union
import warnings
import numpy_indexed


def shift_score(distances: np.ndarray, thresh: float) -> float:
    """
    Computes a score to quantify how good a shift is based on the distances between the neighbours found.
    the value of this score is approximately the number of close neighbours found.

    Args:
        distances: `float [n_neighbours]`.
            Distances between each pair of neighbours.
        thresh: Basically the distance in pixels below which neighbours are a good match.
            Typical = `2`.

    Returns:
        Score to quantify how good a shift is based on the distances between the neighbours found.
    """
    return np.sum(np.exp(-distances ** 2 / (2 * thresh ** 2)))


def extend_array(array: np.ndarray, extend_scale: int, direction: str = 'both') -> np.ndarray:
    """
    Extrapolates array using its mean spacing in the direction specified by `extend_sz` values.

    Args:
        array: `float [n_values]`.
            Array probably produced using `np.arange`. It is expected to be in ascending order with constant step.
        extend_scale: By how many values to extend the array.
        direction: One of the following, specifying how to extend the `array` -

            - `'below'` - `array` extended below the min value.
            - `'above'` - `array` extended above the max value
            - `'both'` - `array` extended in both directions (by `extend_sz` in each direction).

    Returns:
        `float [n_values + extend_scale * (direction == 'both' + 1)]`.
            `array` extrapolated in `direction` specified.
    """
    if extend_scale == 0:
        ext_array = array
    else:
        step = np.mean(np.ediff1d(array))
        ext_below = np.arange(array.min() - extend_scale * step, array.min(), step)
        ext_above = np.arange(array.max() + step, array.max() + extend_scale * step + step / 2, step)
        if direction == 'below':
            ext_array = np.concatenate((ext_below, array))
        elif direction == 'above':
            ext_array = np.concatenate((array, ext_above))
        elif direction == 'both':
            ext_array = np.concatenate((ext_below, array, ext_above))
        else:
            raise ValueError(f"direction specified was {direction}, whereas it should be 'below', 'above' or 'both'")
    return ext_array


def refined_shifts(shifts: np.ndarray, best_shift: float, refined_scale: float = 0.5,
                   extend_scale: float = 2) -> np.ndarray:
    """
    If `shifts` is an array with mean spacing `step` then this builds array
    that covers from

    `best_shift - extend_scale * step` to `best_shift + extend_scale * step`
    with a spacing of `step*refined_scale`.

    The new step, `step*refined_scale`, is forced to be an integer.

    If only one `shift` provided, doesn't do anything.

    Args:
        shifts: `float [n_shifts]`.
            Array probably produced using `np.arange`. It is expected to be in ascending order with constant step.
        best_shift: Value in `shifts` to build new shifts around.
        refined_scale: Scaling to apply to find new shift spacing.
        extend_scale: By how many steps to build new shifts.

    Returns:
        `float [n_new_shifts]`. Array covering from

        `best_shift - extend_scale * step` to `best_shift + extend_scale * step` with a spacing of `step*refined_scale`.
    """
    if np.size(shifts) == 1:
        refined_shifts = shifts
    else:
        step = np.mean(np.ediff1d(shifts))
        refined_step = np.ceil(refined_scale * step).astype(int)
        refined_shifts = np.arange(best_shift - extend_scale * step,
                                   best_shift + extend_scale * step + refined_step / 2, refined_step)
    return refined_shifts


def update_shifts(search_shifts: np.ndarray, prev_found_shifts: np.ndarray) -> np.ndarray:
    """
    Returns a new array of `search_shifts` around the mean of `prev_found_shifts` if new array has fewer entries or if
    mean of `prev_found_shifts` is outside initial range of `search_shifts`.
    If more than one `prev_found_shifts` is outside the `search_shifts` in the same way i.e. too high or too low,
    `search_shifts` will be updated too.

    Args:
        search_shifts: `int [n_shifts]`.
            Indicates all shifts currently searched over.
        prev_found_shifts: `int [n_shifts_found]`.
            Indicate shifts found on all previous runs of `compute_shift`.

    Returns:
        `int [n_new_shifts]`.

        New set of shifts around mean of previously found shifts.
        Will only return updated shifts if new array has fewer entries than before or mean of `prev_found_shifts`
        is outside range of `search_shifts`.
    """
    n_shifts = len(search_shifts)
    n_prev_shifts = len(prev_found_shifts)
    if n_shifts > 1 and n_prev_shifts > 0:
        step = np.mean(np.ediff1d(search_shifts))
        mean_shift = np.mean(prev_found_shifts, dtype=int)
        n_shifts_new = 2 * np.ceil((np.max(prev_found_shifts) - mean_shift) / step + 1).astype(int) + 1
        if n_shifts_new < n_shifts or mean_shift <= search_shifts.min() or mean_shift >= search_shifts.max():
            # only update shifts if results in less to search over.
            search_shifts = refined_shifts(search_shifts, mean_shift, 1, ((n_shifts_new - 1) / 2).astype(int))
        if np.sum(prev_found_shifts > search_shifts.max()) > 1:
            search_shifts = np.arange(search_shifts.min(), prev_found_shifts.max() + step, step)
        if np.sum(prev_found_shifts < search_shifts.min()) > 1:
            search_shifts = np.arange(prev_found_shifts.min(), search_shifts.max() + step, step)
    return search_shifts


def get_best_shift_3d(yxz_base: np.ndarray, yxz_transform_tree: KDTree, neighb_dist_thresh: float, y_shifts: np.ndarray,
                      x_shifts: np.ndarray, z_shifts: np.ndarray,
                      ignore_shifts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Finds the shift from those given that is best applied to `yx_base` to match `yx_transform`.

    Args:
        yxz_base: `float [n_spots_base x 3]`.
            Coordinates of spots on base image (yxz units must be same).
        yxz_transform_tree: KDTree built from coordinates of spots on transformed image
            (`float [n_spots_transform x 3]`, yxz units must be same).
        neighb_dist_thresh: Basically the distance below which neighbours are a good match.
            Typical = `2`.
        y_shifts: `float [n_y_shifts]`.
            All possible shifts to test in y direction, probably made with `np.arange`.
        x_shifts: `float [n_x_shifts]`.
            All possible shifts to test in x direction, probably made with `np.arange`.
        z_shifts: `float [n_z_shifts]`.
            All possible shifts to test in z direction, probably made with `np.arange`.
        ignore_shifts: `float [n_ignore x 3]`.
            Contains yxz shifts to not search over.
            If `None`, all permutations of `y_shifts`, `x_shifts`, `z_shifts` used.

    Returns:
        - `best_shift` - `float [shift_y, shift_x, shift_z]`.
            Best shift found.
        - `best_score` - `float`.
            Score of best shift.
        - `all_shifts` - `float [n_shifts x 3]`.
            yxz shifts searched over.
        - `score` - `float [n_shifts]`.
            Score of all shifts.
    """
    all_shifts = np.array(np.meshgrid(y_shifts, x_shifts, z_shifts)).T.reshape(-1, 3)
    if ignore_shifts is not None:
        all_shifts = setdiff2d(all_shifts, ignore_shifts)
    score = np.zeros(all_shifts.shape[0])
    dist_upper_bound = 3 * neighb_dist_thresh  # beyond this, score < exp(-4.5) and quicker to use this.
    for i in range(all_shifts.shape[0]):
        yxz_shifted = yxz_base + all_shifts[i]
        distances = yxz_transform_tree.query(yxz_shifted, distance_upper_bound=dist_upper_bound)[0]
        score[i] = shift_score(distances, neighb_dist_thresh)
    best_shift_ind = score.argmax()
    return all_shifts[best_shift_ind], score[best_shift_ind], all_shifts, score


def get_best_shift_2d(yx_base_slices: List[np.ndarray], yx_transform_trees: List[KDTree], neighb_dist_thresh: float,
                      y_shifts: np.ndarray, x_shifts: np.ndarray,
                      ignore_shifts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """

    Args:
        yx_base_slices: List of n_slices arrays indicating yx_base coordinates of spots in that slice.
        yx_transform_trees: List of n_slices KDTrees, each built from the yx_transform coordinates of spots in
            that slice.
        neighb_dist_thresh: Basically the distance below which neighbours are a good match.
            Typical = `2`.
        y_shifts: `float [n_y_shifts]`.
            All possible shifts to test in y direction, probably made with `np.arange`.
        x_shifts: `float [n_x_shifts]`.
            All possible shifts to test in x direction, probably made with `np.arange`.
        ignore_shifts: `float [n_ignore x 2]`.
            Contains yx shifts to not search over.
            If `None`, all permutations of `y_shifts`, `x_shifts` used.

    Returns:
        - `best_shift` - `float [shift_y, shift_x]`.
            Best shift found.
        - `best_score` - `float`.
            Score of best shift.
        - `all_shifts` - `float [n_shifts x 2]`.
            yx shifts searched over.
        - `score` - `float [n_shifts]`.
            Score of all shifts.
    """
    all_shifts = np.array(np.meshgrid(y_shifts, x_shifts)).T.reshape(-1, 2)
    if ignore_shifts is not None:
        all_shifts = setdiff2d(all_shifts, ignore_shifts)
    score = np.zeros(all_shifts.shape[0])
    n_trees = len(yx_transform_trees)
    dist_upper_bound = 3 * neighb_dist_thresh  # beyond this, score < exp(-4.5) and quicker to use this.
    for i in range(all_shifts.shape[0]):
        for j in range(n_trees):
            yx_shifted = yx_base_slices[j] + all_shifts[i]
            distances = yx_transform_trees[j].query(yx_shifted, distance_upper_bound=dist_upper_bound)[0]
            score[i] += shift_score(distances, neighb_dist_thresh)
    best_shift_ind = score.argmax()
    return all_shifts[best_shift_ind], score[best_shift_ind], all_shifts, score


def get_score_thresh(all_shifts: np.ndarray, all_scores: np.ndarray, best_shift: Union[np.ndarray, List], min_dist: float,
                     max_dist: float, thresh_multiplier: float) -> Tuple[float, Optional[np.ndarray]]:
    """
    Score thresh is the max of all scores from transforms between a `distance=min_dist` and `distance=max_dist`
    from the `best_shift`.
    I.e. we expect just for actual shift, there will be sharp gradient in score near it,
    so threshold is multiple of nearby score.
    If not the actual shift, then expect scores in this annulus will also be quite large.

    Args:
        all_shifts: `float [n_shifts x 2]`.
            yx shifts searched over.
        all_scores: `float [n_shifts]`.
            `all_scores[s]` is the score corresponding to `all_shifts[s]`.
        best_shift: `float [2]`.
            yx shift with the best score.
        min_dist: `score_thresh` computed from `all_shifts` a distance between `min_shift` and `max_shift`
            from `best_shifts`.
        max_dist: `score_thresh` computed from `all_shifts` a distance between `min_shift` and `max_shift`
            from `best_shifts`.
        thresh_multiplier: `score_thresh` is `thresh_multiplier` * mean of scores of shifts the correct distance
            from `best_shift`.

    Returns:
        score_thresh - Threshold used to determine if `best_shift` found is legitimate.
        shift_thresh - `float [2]`
            shift corresponding to `score_thresh`. Will be None if there were no shifts
            in the range set by `min_dist` and `max_dist`.
    """
    dist_to_best = pairwise_distances(np.array(all_shifts), np.array(best_shift)[np.newaxis]).squeeze()
    use = np.where(np.logical_and(dist_to_best <= max_dist, dist_to_best >= min_dist))[0]
    if len(use) > 0:
        thresh_ind = use[np.argmax(all_scores[use])]
        score_thresh = thresh_multiplier * all_scores[thresh_ind]
        shift_thresh = all_shifts[thresh_ind]
    else:
        score_thresh = thresh_multiplier * np.median(all_scores)
        shift_thresh = None
    return score_thresh, shift_thresh


def get_2d_slices(yxz_base: np.ndarray, yxz_transform: np.ndarray,
                  nz_collapse: Optional[int]) -> Tuple[List[np.ndarray], List[KDTree], int]:
    """
    This splits `yxz_base` and `yxz_transform` into `n_slices = nz / nz_collapse` 2D slices.
    Then can do a 2D exhaustive search over multiple 2D slices instead of 3D exhaustive search.

    Args:
        yxz_base: `float [n_spots_base x 3]`.
            Coordinates of spots on base image (yx in yx pixel units, z in z pixel units).
        yxz_transform: `float [n_spots_transform x 3]`.
            Coordinates of spots on transformed image (yx in yx pixel units, z in z pixel units).
        nz_collapse: Maximum number of z-planes allowed to be flattened into a 2D slice.
            If `None`, `n_slices`=1.

    Returns:
        - `yx_base_slices` - List of n_slices arrays indicating yx_base coordinates of spots in that slice.
        - `yx_transform_trees` - List of n_slices KDTrees, each built from the yx_transform coordinates of spots in
            that slice.
        - transform_min_z - Guess of z shift from `yxz_base` to `yxz_transform` in units of `z_pixels`.
    """
    if nz_collapse is not None:
        nz = int(np.ceil(yxz_base[:, 2].max() + 1))
        n_slices = int(np.ceil(nz / nz_collapse))
        base_z_slices = np.array_split(np.arange(nz), n_slices)
        slice_max_z_base = 0  # min z for slice 0
        transform_max_z = int(np.ceil(yxz_transform[:, 2].max() + 1))
        # transform_min_z provides an approx guess to the z shift.
        transform_min_z = np.min([int(np.floor(yxz_transform[:, 2].min())), transform_max_z - nz])
        slice_max_z_transform = transform_min_z  # min z for slice 0
        yx_base_slices = []
        yx_transform_trees = []
        for i in range(n_slices):
            slice_min_z_base = slice_max_z_base  # set min z to the max z of the last slice
            slice_max_z_base = base_z_slices[i][-1] + 1
            in_slice_base = np.array([yxz_base[:, 2] >= slice_min_z_base,
                                      yxz_base[:, 2] < slice_max_z_base]).all(axis=0)
            yx_base_slices.append(yxz_base[in_slice_base, :2])
            # transform z coords may have systematic z shift so start from min_z not 0.
            slice_min_z_transform = slice_max_z_transform  # set min z to the max z of the last slice
            if i == n_slices-1:
                # For final slice, ensure all z planes in yxz_transform included.
                slice_max_z_transform = transform_max_z + 1
            else:
                slice_max_z_transform = base_z_slices[i][-1] + 1 + transform_min_z
            in_slice_transform = np.array([yxz_transform[:, 2] >= slice_min_z_transform,
                                           yxz_transform[:, 2] < slice_max_z_transform]).all(axis=0)
            yx_transform_trees.append(KDTree(yxz_transform[in_slice_transform, :2]))
    else:
        transform_min_z = 0
        yx_base_slices = [yxz_base[:, :2]]
        yx_transform_trees = [KDTree(yxz_transform[:, :2])]
    return yx_base_slices, yx_transform_trees, transform_min_z


def compute_shift(yxz_base: np.ndarray, yxz_transform: np.ndarray, min_score_2d: Optional[float],
                  min_score_multiplier: Optional[float], min_score_min_dist: Optional[float],
                  min_score_max_dist: Optional[float],
                  neighb_dist_thresh: float, y_shifts: np.ndarray, x_shifts: np.ndarray,
                  z_shifts: Optional[np.ndarray] = None, widen: Optional[List[int]] = None,
                  max_range: Optional[List[int]] = None, z_scale: Union[float, List] = 1,
                  nz_collapse: Optional[int] = None, z_step: int = 3) -> Tuple[np.ndarray, float, float, dict]:
    """
    This finds the shift from those given that is best applied to `yxz_base` to match `yxz_transform`.

    If the `score` of this is below `min_score_2d`, a widened search is performed.
    If the `score` is above `min_score_2d`, a refined search is done about the best shift so as to find the absolute
    best shift, not the best shift among those given.

    Args:
        yxz_base: `int [n_spots_base x 3]`.
            Coordinates of spots on base image (yx in yx pixel units, z in z pixel units).
        yxz_transform: `int [n_spots_transform x 3]`.
            Coordinates of spots on transformed image (yx in yx pixel units, z in z pixel units).
        min_score_2d: If score of best shift is below this, will search among the widened shifts.
            If `None`, `min_score_2d` will be computed using `get_score_thresh`.
        min_score_multiplier: Parameter used to find `min_score_2d` and `min_score_3d` if not given.
            Typical = `1.5` (definitely more than `1`).
        min_score_min_dist: `min_score_2d` is set to max score of those scores for shifts a distance between `min_dist`
            and `max_dist` from the best_shift.
        min_score_max_dist: `min_score_2d` is set to max score of those scores for shifts a distance between `min_dist`
            and `max_dist` from the best_shift.
        neighb_dist_thresh: Basically the distance below which neighbours are a good match.
            Typical = `2`.
        y_shifts: `float [n_y_shifts]`.
            All possible shifts to test in y direction, probably made with `np.arange`.
        x_shifts: `float [n_x_shifts]`.
            All possible shifts to test in x direction, probably made with `np.arange`.
        z_shifts: `float [n_z_shifts]`.
            All possible shifts to test in z direction, probably made with `np.arange`.
            If not given, will compute automatically from initial guess when making slices and `z_step`.
        widen: `int [3]`.
            By how many shifts to extend search in `[y, x, z]` direction if score below `min_score`.
            This many are added above and below current range.
            If all widen parameters are `0`, widened search is never performed.
            If `None`, set to `[0, 0, 0]`.
        max_range: `int [3]`.
            The range of shifts searched over will continue to be increased according to `widen` until
            the `max_range` is reached in each dimension.
            If a good shift is still not found, the best shift will still be returned without error.
            If None and widen supplied, range will only be widened once.
        z_scale: By what scale factor to multiply z coordinates to make them same units as xy.
            I.e. `z_pixel_size / xy_pixel_size`.
            If one value, given same scale used for yxz_base and yxz_transform.
            Otherwise, first value used for yxz_base and second for yxz_transform.
        nz_collapse: Maximum number of z-planes allowed to be flattened into a 2D slice.
            If `None`, `n_slices`=1. Should be `None` for 2D data.
        z_step: `int`.
            Step of shift search in z direction in uints of `z_pixels`.
            `z_shifts` are computed automatically as 1 shift either side of an initial guess.

    Returns:
        - `best_shift` - `float [shift_y, shift_x, shift_z]`.
            Best shift found.
        - `best_score` - `float`.
            Score of best shift.
        - `min_score_3d` - `float`.
            Same as `min_score_2d`, unless input was `None` in which case this is the calculated value.
        - `debug_info` - dict containing debugging information:
            - `shifts_2d`: `int [n_shifts_2d x 2]`
                All yx shifts searched to get best `yx_shift`.
            - `scores_2d`: `float [n_shifts_2d]`
                Score corresponding to each 2d shift.
            - `shifts_3d`: `int [n_shifts_3d x 3]`
                All yxz shifts searched to get best `yxz_shift`. `None` if `nz_collapse is None` i.e. 2D point cloud.
            - `scores_3d`: `float [n_shifts_3d]`
                Score corresponding to each 3d shift. `None` if `nz_collapse is None` i.e. 2D point cloud.
            - `shift_2d_initial`: `float [2]`
                Best shift found after first 2D search. I.e. annulus around this shift was used
                to compute `min_score_2d` and `shift_thresh`.
            - `shift_thresh`: `int [3]`
                yxz shift corresponding to `min_score_3d`. Will be `None` if `min_score_2d` provided in advance.
                `shift_thresh[:2]` is the yx shift corresponding to `min_score_2d`
            - `min_score_2d`: Same as input `min_score_2d`, unless was `None` in
                which case this is the calculated value.
    """
    if widen is None:
        widen = [0, 0, 0]
    if np.asarray(z_scale).size == 1:
        z_scale = [z_scale, z_scale]
    if len(z_scale) > 2:
        raise ValueError(f'Only 2 z_scale values should be provided but z_scale given was {z_scale}.')
    yx_base_slices, yx_transform_trees, z_shift_guess = get_2d_slices(yxz_base, yxz_transform, nz_collapse)
    if nz_collapse is not None:
        # Only do z-scaling in 3D case
        yxz_base = yxz_base * [1, 1, z_scale[0]]
        yxz_transform = yxz_transform * [1, 1, z_scale[1]]
    yxz_transform_tree = KDTree(yxz_transform)
    shift_2d, score_2d, all_shifts_2d, all_scores_2d = get_best_shift_2d(yx_base_slices, yx_transform_trees,
                                                                         neighb_dist_thresh, y_shifts, x_shifts)

    # Only look at 3 shifts in z to start with about guess from getting the 2d slices.
    if z_shifts is None:
        z_shifts = np.arange(z_shift_guess - z_step, z_shift_guess + z_step + 1, z_step)

    # save initial_shifts so don't look over same shifts twice
    # initial_shifts = np.array(np.meshgrid(y_shifts, x_shifts)).T.reshape(-1, 2)
    shift_2d_initial = shift_2d.copy()
    if min_score_2d is None:
        min_score_2d, shift_thresh = get_score_thresh(all_shifts_2d, all_scores_2d, shift_2d, min_score_min_dist,
                                                      min_score_max_dist, min_score_multiplier)
        shift_thresh = np.pad(shift_thresh, (0, 1))  # add z shift = 0
    else:
        shift_thresh = None
    if score_2d <= min_score_2d and np.max(widen[:2]) > 0:
        shift_ranges = np.array([np.ptp(i) for i in [y_shifts, x_shifts]])
        if max_range is None:
            # If don't specify max_range, only widen once.
            max_range = np.array([np.ptp(i) for i in [y_shifts, x_shifts, z_shifts]]) * (np.array(widen[:2]) > 0)
            max_range[max_range > 0] += 1
            max_range_2d = max_range[:2]
        else:
            max_range_2d = np.asarray(max_range[:2])
        # keep extending range of shifts in yx until good score reached or hit max shift_range.
        while score_2d <= min_score_2d:
            if np.all(shift_ranges >= max_range_2d):
                warnings.warn(f"Shift search range exceeds max_range = {max_range_2d} in yxz directions but \n"
                              f"best score is only {round(score_2d, 2)} which is below "
                              f"min_score = {round(min_score_2d, 2)}."
                              f"\nBest shift found was {shift_2d}.")
                break
            else:
                warnings.warn(f"Best shift found ({shift_2d}) has score of {round(score_2d, 2)} which is below "
                              f"min_score = {round(min_score_2d, 2)}."
                              f"\nRunning again with extended shift search range in yx.")
            if shift_ranges[0] < max_range_2d[0]:
                y_shifts = extend_array(y_shifts, widen[0])
            if shift_ranges[1] < max_range_2d[1]:
                x_shifts = extend_array(x_shifts, widen[1])
            shift_2d_new, score_2d_new, all_shifts_new, all_scores_new = \
                get_best_shift_2d(yx_base_slices, yx_transform_trees, neighb_dist_thresh, y_shifts, x_shifts,
                                  all_shifts_2d)
            if score_2d_new > score_2d:
                score_2d = score_2d_new
                shift_2d = shift_2d_new
            # update initial_shifts so don't look over same shifts twice
            all_shifts_2d = np.append(all_shifts_2d, all_shifts_new, axis=0)
            all_scores_2d = np.append(all_scores_2d, all_scores_new, axis=0)
            # initial_shifts = np.array(np.meshgrid(y_shifts, x_shifts)).T.reshape(-1, 2)
            shift_ranges = np.array([np.ptp(i) for i in [y_shifts, x_shifts]])

    if nz_collapse is None:
        # nz_collapse not provided for 2D data.
        shift = np.append(shift_2d, 0)
        score = score_2d
        all_shifts_3d = None
        all_scores_3d = None
        min_score_3d = min_score_2d
    else:
        ignore_shifts = None
        if shift_thresh is None:
            y_shift_2d = np.array(shift_2d[0])
            x_shift_2d = np.array(shift_2d[1])
        else:
            y_shift_2d = np.array([shift_2d[0], shift_thresh[0]])
            x_shift_2d = np.array([shift_2d[1], shift_thresh[1]])
            if len(np.unique(y_shift_2d)) == 2 and len(np.unique(x_shift_2d)) == 2:
                # Only find shifts for the shift_2d and shift_thresh, get rid of cross terms.
                ignore_shifts = np.array([[shift_2d[0], shift_thresh[1]], [shift_thresh[0], shift_2d[1]]])
                ignore_shifts = np.tile(np.pad(ignore_shifts, [(0, 0), (0, 1)]), [len(z_shifts), 1])
                ignore_shifts[:, 2] = np.repeat(z_shifts * z_scale[0], len(y_shift_2d))
        # z_scale for yxz_base used from now on as we are finding the shift from yxz_base to yxz_transform.
        shift, score, all_shifts_3d, all_scores_3d = get_best_shift_3d(yxz_base, yxz_transform_tree, neighb_dist_thresh,
                                                                       y_shift_2d, x_shift_2d, z_shifts * z_scale[0],
                                                                       ignore_shifts=ignore_shifts)
        if shift_thresh is not None:
            # Set min_score_3d to max score at shift used to find min_score_2d across all z planes
            # multiplied by min_score_multiplier
            shift_thresh_ind = np.where(numpy_indexed.indices(shift_thresh[np.newaxis, :2].astype(int),
                                                              all_shifts_3d[:, :2].astype(int), missing=-10) == 0)[0]
            shift_thresh_best_ind = shift_thresh_ind[np.argmax(all_scores_3d[shift_thresh_ind])]
            min_score_3d = all_scores_3d[shift_thresh_best_ind] * min_score_multiplier
            shift_thresh = (all_shifts_3d[shift_thresh_best_ind] / [1, 1, z_scale[0]]).astype(int)
        else:
            min_score_3d = min_score_2d

        if score < min_score_2d and widen[2] > 0:
            # keep extending range of shifts in z until good score reached or hit max shift_range.
            # yx shift is kept as 2d shift found when using slices.
            max_range_z = np.asarray(max_range[2])
            z_shift_range = np.ptp(z_shifts)
            while score < min_score_3d:
                if z_shift_range > max_range_z:
                    warnings.warn(f"Shift search range exceeds max_range = {max_range_z} in z directions but \n"
                                  f"best score is only {np.around(score, 2)} which is below "
                                  f"min_score = {np.around(min_score_3d, 2)}."
                                  f"\nBest shift found was {shift}.")
                    break
                else:
                    warnings.warn(f"Best shift found ({shift}) has score of {round(score, 2)} which is below "
                                  f"min_score = {np.around(min_score_3d, 2)}."
                                  f"\nRunning again with extended shift search range in z.")
                z_shifts = extend_array(z_shifts, widen[2])
                shift_new, score_new, all_shifts_new, all_scores_new = \
                    get_best_shift_3d(yxz_base, yxz_transform_tree, neighb_dist_thresh, y_shift_2d,
                                      x_shift_2d, z_shifts * z_scale[0], all_shifts_3d)
                if score_new > score:
                    score = score_new
                    shift = shift_new
                # update initial_shifts so don't look over same shifts twice
                all_shifts_3d = np.append(all_shifts_3d, all_shifts_new, axis=0)
                all_scores_3d = np.append(all_scores_3d, all_scores_new, axis=0)
                z_shift_range = np.ptp(z_shifts)

    # refined search near maxima with half the step
    y_shifts = refined_shifts(y_shifts, shift[0])
    x_shifts = refined_shifts(x_shifts, shift[1])
    z_shifts = refined_shifts(z_shifts, shift[2] / z_scale[0])
    shift2, score2, all_shifts_new, all_scores_new = \
        get_best_shift_3d(yxz_base, yxz_transform_tree, neighb_dist_thresh, y_shifts, x_shifts,
                          z_shifts * z_scale[0], all_shifts_3d)
    if score2 > score:
        shift = shift2
    if nz_collapse is None:
        all_shifts_2d = np.append(all_shifts_2d, all_shifts_new[:, :2], axis=0)
        all_scores_2d = np.append(all_scores_2d, all_scores_new, axis=0)
    else:
        all_shifts_3d = np.append(all_shifts_3d, all_shifts_new, axis=0)
        all_scores_3d = np.append(all_scores_3d, all_scores_new, axis=0)
    # final search with a step of 1
    y_shifts = refined_shifts(y_shifts, shift[0], refined_scale=1e-50, extend_scale=1)
    x_shifts = refined_shifts(x_shifts, shift[1], refined_scale=1e-50, extend_scale=1)
    z_shifts = refined_shifts(z_shifts, shift[2] / z_scale[0], refined_scale=1e-50, extend_scale=1)
    shift, score, all_shifts_new, all_scores_new = \
        get_best_shift_3d(yxz_base, yxz_transform_tree, neighb_dist_thresh, y_shifts, x_shifts,
                          z_shifts * z_scale[0], all_shifts_3d)
    if nz_collapse is None:
        all_shifts_2d = np.append(all_shifts_2d, all_shifts_new[:, :2], axis=0)
        all_scores_2d = np.append(all_scores_2d, all_scores_new, axis=0)
    else:
        all_shifts_3d = np.append(all_shifts_3d, all_shifts_new, axis=0)
        all_scores_3d = np.append(all_scores_3d, all_scores_new, axis=0)
        all_shifts_3d[:, 2] = all_shifts_3d[:, 2] / z_scale[0]
        all_shifts_3d = all_shifts_3d.astype(np.int16)
    shift[2] = shift[2] / z_scale[0]
    return shift.astype(int), score, min_score_3d, {'shifts_2d': all_shifts_2d, 'scores_2d': all_scores_2d,
                                                    'shifts_3d': all_shifts_3d, 'scores_3d': all_scores_3d,
                                                    'shift_2d_initial': shift_2d_initial, 'shift_thresh': shift_thresh,
                                                    'min_score_2d': min_score_2d}
