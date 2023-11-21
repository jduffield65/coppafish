import warnings
from typing import Optional, Tuple
import numpy as np
from jax import numpy as jnp

from .. import utils


def detect_spots(image: np.ndarray, intensity_thresh: float, radius_xy: Optional[int], radius_z: Optional[int] = None,
                 remove_duplicates: bool = False, se: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds local maxima in image exceeding `intensity_thresh`.
    This is achieved by looking at neighbours of pixels above intensity_thresh.
    Should use for a small `se` and high `intensity_thresh`.

    Args:
        image: `float [n_y x n_x x n_z]`.
            `image` to find spots on.
        intensity_thresh: Spots are local maxima in image with `pixel_value > intensity_thresh`.
        radius_xy: Radius of dilation structuring element in xy plane (approximately spot radius).
        radius_z: Radius of dilation structuring element in z direction (approximately spot radius).
            Must be more than 1 to be 3D.
            If `None`, 2D filter is used.
        remove_duplicates: Whether to only keep one pixel if two or more pixels are local maxima and have
            same intensity. Only works with integer image.
        se: `int [se_sz_y x se_sz_x x se_sz_z]`.
            Can give structuring element manually rather than using a cuboid element.
            Must only contain zeros and ones.

    Returns:
        - `peak_yxz` - `int [n_peaks x image.ndim]`.
            yx or yxz location of spots found.
        - `peak_intensity` - `float [n_peaks]`.
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

    paddings = jnp.asarray([(pad_size_y, pad_size_y), (pad_size_x, pad_size_x), (pad_size_z, pad_size_z)])[:image.ndim]
    keep = np.asarray(get_local_maxima_jax(image, jnp.asarray(se_shifts), paddings, jnp.asarray(consider_yxz), 
                                               jnp.asarray(consider_intensity)))
    if remove_duplicates:
        peak_intensity = np.round(consider_intensity[keep]).astype(int)
    else:
        peak_intensity = consider_intensity[keep]
    peak_yxz = np.array(consider_yxz).transpose()[keep]
    return peak_yxz, peak_intensity


def get_local_maxima_jax(image: jnp.ndarray, se_shifts: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                         pad_sizes: jnp.ndarray, consider_yxz: jnp.ndarray, consider_intensity: jnp.ndarray) \
                             -> jnp.ndarray:
    """
    Finds the local maxima from a given set of pixels to consider.

    Args:
        image (`[n_y x n_x x n_z] ndarray[float]`): `image` to find spots on.
        se_shifts (`[image.ndim x n_consider]` ndarray[int]): y, x, z shifts which indicate neighbourhood about each 
            spot where local maxima search carried out.
        pad_sizes ([image.ndim] ndarray[list of int]): `pad_sizes[i,0]` represents the top padding amount on the image 
            for dimension `i`, `pad_sizes[i,1]` represents the bottom padding amount. `i=0,1,2` represent y, x and z.
        consider_yxz (`[3 x n_consider] ndarray[int]`): all yxz coordinates where value in image is greater than an 
            intensity threshold.
        consider_intensity (`[n_consider] ndarray[float]`): value of image at coordinates given by `consider_yxz`.

    Returns:
        `[n_consider] ndarray[bool]`: whether each point in `consider_yxz` is a local maxima or not.
    """
    image = jnp.pad(image, pad_sizes, mode='constant', constant_values=0)
    consider_yxz_padded = jnp.add(consider_yxz, pad_sizes[:,0][:,None])
    se_shifts_flat = se_shifts.reshape((image.ndim, -1))
    consider_yxz_padded_shifted = jnp.add(consider_yxz_padded[..., None], se_shifts_flat[:, None, :])
    keep = jnp.all(image[tuple(consider_yxz_padded_shifted)] <= consider_intensity[..., None], axis=-1)

    return keep
