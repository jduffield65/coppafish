import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import iqr
from ..utils.base import setdiff2d
from typing import Tuple, Optional, List


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
    return np.sum(np.exp(-distances**2 / (2*thresh**2)))


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
        ext_below = np.arange(array.min() - extend_scale*step, array.min(), step)
        ext_above = np.arange(array.max() + step, array.max() + extend_scale*step + step / 2, step)
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
                                   best_shift + extend_scale * step + refined_step/2, refined_step)
    return refined_shifts


def update_shifts(search_shifts: np.ndarray, prev_found_shifts: np.ndarray) -> np.ndarray:
    """
    Returns a new array of `search_shifts` around the mean of `prev_found_shifts` if new array has fewer entries or if
    mean of `prev_found_shifts` is outside initial range of `search_shifts`.

    Args:
        search_shifts: `int [n_shifts]`.
            Indicates all shifts currently searched over.
        prev_found_shifts: `int [n_shifts_found]`.
            Indicates shifts found on all previous runs of `compute_shift`.

    Returns:
        `int [n_new_shifts]`.

        New set of shifts around mean of previously found shifts.
        Will only return updated shifts if new array has fewer entries than before or mean of `prev_found_shifts`
        is outside range of `search_shifts`.
    """
    n_shifts = len(search_shifts)
    if n_shifts > 1:
        step = np.mean(np.ediff1d(search_shifts))
        mean_shift = np.mean(prev_found_shifts, dtype=int)
        n_shifts_new = 2*np.ceil((max(prev_found_shifts) - mean_shift)/step + 1).astype(int)+1
        if n_shifts_new < n_shifts or mean_shift <= search_shifts.min() or mean_shift >= search_shifts.max():
            # only update shifts if results in less to search over.
            search_shifts = refined_shifts(search_shifts, mean_shift, 1, ((n_shifts_new - 1) / 2).astype(int))
    return search_shifts


def get_best_shift(yxz_base: np.ndarray, yxz_transform: np.ndarray, neighb_dist_thresh: float, y_shifts: np.ndarray,
                   x_shifts: np.ndarray, z_shifts: np.ndarray,
                   ignore_shifts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float, float]:
    """
    Finds the shift from those given that is best applied to `yx_base` to match `yx_transform`.

    Args:
        yxz_base: `float [n_spots_base x 3]`.
            Coordinates of spots on base image (yxz units must be same).
        yxz_transform: `float [n_spots_transform x 3]`.
            Coordinates of spots on transformed image (yxz units must be same).
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
        - `median_score` - `float`.
            Median of scores of all shifts.
        - `iqr_score` - `float`.
            Interquartile range of scores of all shifts.
    """
    all_shifts = np.array(np.meshgrid(y_shifts, x_shifts, z_shifts)).T.reshape(-1, 3)
    if ignore_shifts is not None:
        all_shifts = setdiff2d(all_shifts, ignore_shifts)
    nbrs = NearestNeighbors(n_neighbors=1).fit(yxz_transform)
    score = np.zeros(all_shifts.shape[0])
    for i in range(all_shifts.shape[0]):
        yx_shifted = yxz_base + all_shifts[i]
        distances, _ = nbrs.kneighbors(yx_shifted)
        score[i] = shift_score(distances, neighb_dist_thresh)
    best_shift_ind = score.argmax()
    return all_shifts[best_shift_ind], score[best_shift_ind], np.median(score), iqr(score)


def compute_shift(yxz_base: np.ndarray, yxz_transform: np.ndarray, min_score: Optional[float],
                  min_score_auto_param: float, neighb_dist_thresh: float,
                  y_shifts: np.ndarray, x_shifts: np.ndarray, z_shifts: np.ndarray, widen: Optional[List[int]] = None,
                  z_scale: float = 1) -> Tuple[np.ndarray, float, float]:
    """
    This finds the shift from those given that is best applied to `yxz_base` to match `yxz_transform`.

    If the `score` of this is below `min_score`, a widened search is performed.
    If the `score` is above `min_score`, a refined search is done about the best shift so as to find the absolute
    best shift, not the best shift among those given.

    Args:
        yxz_base: `float [n_spots_base x 3]`.
            Coordinates of spots on base image (yxz units must be same).
        yxz_transform: `float [n_spots_transform x 3]`.
            Coordinates of spots on transformed image (yxz units must be same).
        min_score: If score of best shift is below this, will search among the widened shifts.
            If `None`, `min_score` will be set to `median(scores) + min_score_auto_param * iqr(scores)`.
        min_score_auto_param: Parameter used to find `min_score` if `min_score` not given.
            Typical = `5` (definitely more than `1`).
        neighb_dist_thresh: Basically the distance below which neighbours are a good match.
            Typical = `2`.
        y_shifts: `float [n_y_shifts]`.
            All possible shifts to test in y direction, probably made with `np.arange`.
        x_shifts: `float [n_x_shifts]`.
            All possible shifts to test in x direction, probably made with `np.arange`.
        z_shifts: `float [n_z_shifts]`.
            All possible shifts to test in z direction, probably made with `np.arange`.
        widen: `int [3]`.
            By how many shifts to extend search in `[y, x, z]` direction if score below `min_score`.
            This many are added above and below current range.
            If all widen parameters are `0`, widened search is never performed.
            If `None`, set to `[0, 0, 0]`.
        z_scale: By what scale factor to multiply z coordinates to make them same units as xy.
            I.e. `z_pixel_size / xy_pixel_size`.

    Returns:
        - `best_shift` - `float [shift_y, shift_x, shift_z]`.
            Best shift found.
        - `best_score` - `float`.
            Score of best shift.
        - `min_score` - `float`.
            Same as input, unless input was `None` in which case this is the calculated value.
    """
    if widen is None:
        widen = [0, 0, 0]
    yxz_base[:, 2] = yxz_base[:, 2] * z_scale
    yxz_transform[:, 2] = yxz_transform[:, 2] * z_scale
    shift, score, score_median, score_iqr = get_best_shift(yxz_base, yxz_transform, neighb_dist_thresh,
                                                           y_shifts, x_shifts, z_shifts * z_scale)
    # save initial_shifts so don't look over same shifts twice
    initial_shifts = np.array(np.meshgrid(y_shifts, x_shifts, z_shifts*z_scale)).T.reshape(-1, 3)
    if min_score is None:
        # TODO: maybe make min_score equal to min_score_auto_param times 10th highest score.
        #  That way, it is independent of number of shifts searched
        min_score = score_median + min_score_auto_param * score_iqr
    if score < min_score and np.max(widen) > 0:
        # look over extended range of shifts if score below threshold
        y_shifts = extend_array(y_shifts, widen[0])
        x_shifts = extend_array(x_shifts, widen[1])
        z_shifts = extend_array(z_shifts, widen[2])
        shift, score, score_median2, score_iqr2 = get_best_shift(yxz_base, yxz_transform, neighb_dist_thresh,
                                                                 y_shifts, x_shifts, z_shifts * z_scale, initial_shifts)
    if score > min_score:
        # refined search near maxima with half the step
        y_shifts = refined_shifts(y_shifts, shift[0])
        x_shifts = refined_shifts(x_shifts, shift[1])
        z_shifts = refined_shifts(z_shifts, shift[2]/z_scale)
        shift2, score2, _, _ = get_best_shift(yxz_base, yxz_transform, neighb_dist_thresh, y_shifts, x_shifts,
                                              z_shifts * z_scale, initial_shifts)
        if score2 > score:
            shift = shift2
        # final search with a step of 1
        y_shifts = refined_shifts(y_shifts, shift[0], refined_scale=1e-50, extend_scale=1)
        x_shifts = refined_shifts(x_shifts, shift[1], refined_scale=1e-50, extend_scale=1)
        z_shifts = refined_shifts(z_shifts, shift[2]/z_scale, refined_scale=1e-50, extend_scale=1)
        shift, score, _, _ = get_best_shift(yxz_base, yxz_transform, neighb_dist_thresh, y_shifts, x_shifts,
                                            z_shifts * z_scale, initial_shifts)
        shift[2] = shift[2] / z_scale
    return shift.astype(int), score, min_score


# TODO: Not sure what amend_shifts function was for. Does not seem to be used in anything.
# def amend_shifts(shift_info, shifts, spot_details, c, r, neighb_dist_thresh, z_scale):
#     good_shifts = (shift_info['score'] > shift_info['score_thresh']).flatten()
#     if sum(good_shifts) < 2 and len(good_shifts) > 4:
#         raise ValueError(f"{len(good_shifts)-sum(good_shifts)}/{len(good_shifts)} of shifts fell below score threshold")
#     elif sum(good_shifts) < len(good_shifts):
#         coords = ['y', 'x', 'z']
#         shift_info['outlier'] = shift_info['shifts']
#         shift_info[good_shifts, :] = 0
#         for i in range(len(coords)):
#             shifts[coords[i]] = update_shifts(shifts[coords[i]], shift_info['shifts'][good_shifts, i])
