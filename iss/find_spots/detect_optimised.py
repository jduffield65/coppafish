import warnings
from functools import partial
from typing import Optional, Tuple, List
import jax
import numpy as np
from jax import numpy as jnp
from .. import utils


def detect_spots(image: np.ndarray, intensity_thresh: float, radius_xy: Optional[int], radius_z: Optional[int] = None,
                 remove_duplicates: bool = False, se: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds local maxima in image exceeding ```intensity_thresh```.
    This is achieved by looking at neighbours of pixels above intensity_thresh.
    Should use for a small `se` and high `intensity_thresh`.

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
            pad_size_z = radius_z-1
        else:
            se = np.ones((2*radius_xy-1, 2*radius_xy-1), dtype=int)
            pad_size_z = 0
        pad_size_y = radius_xy - 1
        pad_size_x = radius_xy - 1
    else:
        se = utils.morphology.ensure_odd_kernel(se)
        pad_size_y = int((se.shape[0]-1)/2)
        pad_size_x = int((se.shape[1]-1)/2)
        if se.ndim == 3:
            pad_size_z = int((se.shape[2] - 1) / 2)
        else:
            pad_size_z = 0
    if image.ndim == 2 and se.ndim == 3:
        mid_z = int(np.floor((se.shape[2]-1)/2))
        warnings.warn(f"2D image provided but 3D filter asked for.\n"
                      f"Using the middle plane ({mid_z}) of this filter.")
        se = se[:, :, mid_z]

    # set central pixel to 0
    se[np.ix_(*[(np.floor((se.shape[i] - 1) / 2).astype(int),) for i in range(se.ndim)])] = 0
    se_shifts = utils.morphology.filter_optimised.get_shifts_from_kernel(se)

    consider_yxz = np.where(image > intensity_thresh)
    n_consider = consider_yxz[0].shape[0]
    if remove_duplicates:
        # perturb image by small amount so two neighbouring pixels that did have the same value now differ slightly.
        # hence when find maxima, will only get one of the pixels not both.
        rng = np.random.default_rng(0)   # So shift is always the same.
        # rand_shift must be larger than small to detect a single spot.
        rand_im_shift = rng.uniform(low=2e-6, high=0.2, size=n_consider).astype(np.float32)
        image = image.astype(np.float32)
        image[consider_yxz] = image[consider_yxz] + rand_im_shift

    consider_intensity = image[consider_yxz]
    consider_yxz = list(consider_yxz)

    keep = np.asarray(get_local_maxima_jax(image, se_shifts, pad_size_y, pad_size_x, pad_size_z, consider_yxz,
                                           consider_intensity))
    if remove_duplicates:
        peak_intensity = np.round(consider_intensity[keep]).astype(int)
    else:
        peak_intensity = consider_intensity[keep]
    peak_yxz = np.array(consider_yxz).transpose()[keep]
    return peak_yxz, peak_intensity


@partial(jax.jit, static_argnums=(2, 3, 4))
def get_local_maxima_jax(image: jnp.ndarray, se_shifts: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                         pad_size_y: int, pad_size_x: int, pad_size_z: int,
                         consider_yxz: List[jnp.ndarray], consider_intensity: jnp.ndarray) -> jnp.ndarray:
    """
    Finds the local maxima from a given set of pixels to consider.

    Args:
        image: ```float [n_y x n_x x n_z]```.
            ```image``` to find spots on.
        se_shifts: `(image.ndim x  int [n_shifts])`.
            y, x, z shifts which indicate neighbourhood about each spot where local maxima search carried out.
        pad_size_y: How much to zero pad image in y.
        pad_size_x: How much to zero pad image in x.
        pad_size_z: How much to zero pad image in z.
        consider_yxz: `[3 x int [n_consider]]`.
            All yxz coordinates where value in image is greater than an intensity threshold.
        consider_intensity: `float [n_consider]`.
            Value of image at coordinates given by `consider_yxz`.

    Returns:
        `bool [n_consider]`
            Whether each point in `consider_yxz` is a local maxima or not.
    """
    pad_size = [(pad_size_y, pad_size_y), (pad_size_x, pad_size_x), (pad_size_z, pad_size_z)][:image.ndim]
    image = jnp.pad(image, pad_size)
    for i in range(len(pad_size)):
        consider_yxz[i] = consider_yxz[i] + pad_size[i][0]
    keep = jnp.ones(consider_yxz[0].shape[0], dtype=bool)
    for i in range(se_shifts[0].shape[0]):
        # Note that in each iteration, only consider coordinates which can still possibly be local maxima.
        keep = keep * (image[tuple([consider_yxz[j] + se_shifts[j][i] for j in range(image.ndim)])] <=
                       consider_intensity)
    return keep
