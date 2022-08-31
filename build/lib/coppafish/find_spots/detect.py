import warnings
from typing import Optional, Tuple
import numpy as np
from .. import utils


def detect_spots(image: np.ndarray, intensity_thresh: float, radius_xy: Optional[int],
                 radius_z: Optional[int] = None, remove_duplicates: bool = False,
                 se: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds local maxima in image exceeding ```intensity_thresh```.
    This is achieved through a dilation being run on the whole image.
    Should use for a large se.

    Args:
        image: ```float [n_y x n_x x n_z]```.
            ```image``` to find spots on.
        intensity_thresh: Spots are local maxima in image with ```pixel_value > intensity_thresh```.
        radius_xy: Radius of dilation structuring element in xy plane (approximately spot radius).
        radius_z: Radius of dilation structuring element in z direction (approximately spot radius).
            Must be more than 1 to be 3D.
            If ```None```, 2D filter is used.
        remove_duplicates: Whether to only keep one pixel if two or more pixels are local maxima and have
            same intensity. Only works with integer image.
        se: ```int [se_sz_y x se_sz_x x se_sz_z]```.
            Can give structuring element manually rather than using a cuboid element.
            Must only contain zeros and ones.

    Returns:
        - ```peak_yxz``` - ```int [n_peaks x image.ndim]```.
            yx or yxz location of spots found.
        - ```peak_intensity``` - ```float [n_peaks]```.
            Pixel value of spots found.
    """
    if se is None:
        # Default is a cuboid se of all ones as is quicker than disk and very similar results.
        if radius_z is not None:
            se = np.ones((2*radius_xy-1, 2*radius_xy-1, 2*radius_z-1), dtype=int)
        else:
            se = np.ones((2*radius_xy-1, 2*radius_xy-1), dtype=int)
    if image.ndim == 2 and se.ndim == 3:
        mid_z = int(np.floor((se.shape[2]-1)/2))
        warnings.warn(f"2D image provided but 3D filter asked for.\n"
                      f"Using the middle plane ({mid_z}) of this filter.")
        se = se[:, :, mid_z]

    small = 1e-6  # for computing local maxima: shouldn't matter what it is (keep below 0.01 for int image).
    if remove_duplicates:
        # perturb image by small amount so two neighbouring pixels that did have the same value now differ slightly.
        # hence when find maxima, will only get one of the pixels not both.
        rng = np.random.default_rng(0)   # So shift is always the same.
        # rand_shift must be larger than small to detect a single spot.
        rand_im_shift = rng.uniform(low=small*2, high=0.2, size=image.shape)
        image = image + rand_im_shift

    dilate = utils.morphology.dilate(image, se)
    spots = np.logical_and(image + small > dilate, image > intensity_thresh)
    peak_pos = np.where(spots)
    peak_yxz = np.concatenate([coord.reshape(-1, 1) for coord in peak_pos], axis=1)
    peak_intensity = image[spots]
    return peak_yxz, peak_intensity
