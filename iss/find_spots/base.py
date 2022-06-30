import warnings
from scipy.spatial import KDTree
from .. import utils
import numpy as np
from typing import Optional, Tuple, Union, List
import jax.numpy as jnp
import jax
from functools import partial


def spot_yxz(spot_details: np.ndarray, tile: int, round: int, channel: int,
             return_isolated: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Function which gets yxz positions (and whether isolated) of spots on a particular ```tile```, ```round```, ```
    channel``` from ```spot_details``` in find_spots notebook page.

    Args:
        spot_details: ```int16 [n_spots x 7]```.
            ```spot_details[s]``` is ```[tile, round, channel, isolated, y, x, z]``` of spot ```s```.
        tile: Tile of desired spots.
        round: Round of desired spots.
        channel: Channel of desired spots.
        return_isolated: Whether to return isolated status of each spot.

    Returns:
        - ```spot_yxz``` - ```int16 [n_trc_spots x 3]```.
            yxz coordinates of spots on chosen ```tile```, ```round``` and ```channel```.
        - ```spot_isolated``` - ```bool [n_trc_spots]``` (Only returned if ```return_isolated = True```).
            Isolated status (```1``` if isolated, ```0``` if not) of the spots.
    """
    #     Function which gets yxz positions (and whether isolated) of spots on a particular ```tile```, ```round```,
    #     ```channel``` from ```spot_details``` in find_spots notebook page.
    use = np.all((spot_details[:, 0] == tile, spot_details[:, 1] == round, spot_details[:, 2] == channel), axis=0)
    if return_isolated:
        return spot_details[use, 4:], spot_details[use, 3]
    else:
        return spot_details[use, 4:]


def detect_spots_dilate(image: np.ndarray, intensity_thresh: float, radius_xy: Optional[int], radius_z: Optional[int] = None,
                        remove_duplicates: bool = False, se: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
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
        diff_to_int = np.round(image).astype(int) - image
        if np.abs(diff_to_int).max() > 0:
            raise ValueError("image should be integer to remove_duplicates but image is float.")
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


def detect_spots(image: np.ndarray, intensity_thresh: float, radius_xy: Optional[int], radius_z: Optional[int] = None,
                 remove_duplicates: bool = False, se: Optional[np.ndarray] = None):
    """
    Finds local maxima in image exceeding ```intensity_thresh```.
    This is achieved by looking at neighbours of pixels above intensity_thresh.
    Should use for a small se.

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
    se_shifts = utils.morphology.get_shifts_from_kernel(se)

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


def get_isolated(image: np.ndarray, spot_yxz: np.ndarray, thresh: float, radius_inner: float, radius_xy: float,
                 radius_z: Optional[float] = None, filter_image: bool = False) -> np.ndarray:
    """
    Determines whether each spot in ```spot_yxz``` is isolated by getting the value of image after annular filtering
    at each location in ```spot_yxz```.

    Args:
        image: ```float [n_y x n_x x n_z]```.
            image spots were found on.
        spot_yxz: ```int [n_peaks x image.ndim]```.
            yx or yxz location of spots found.
            If axis 1 dimension is more than ```image.ndim```, only first ```image.ndim``` dimensions used
            i.e. if supply yxz, with 2d image, only yx position used.
        thresh: Spots are isolated if annulus filtered image at spot location less than this.
        radius_inner: Inner radius of annulus filtering kernel within which values are all zero.
        radius_xy: Outer radius of annulus filtering kernel in xy direction.
        radius_z: Outer radius of annulus filtering kernel in z direction.
            If ```None```, 2D filter is used.
        filter_image: Whether to get result via filtering whole image first. Will be slower.

    Returns:
        ```bool [n_peaks]```.
            Whether each spot is isolated or not.

    """
    se = utils.strel.annulus(radius_inner, radius_xy, radius_z)
    # With just coords, takes about 3s for 50 z-planes.

    if filter_image:
        # This filtering takes around 40s for 50 z-planes.
        # May get memory error here as uses oa_convolve.
        # If use scipy.ndimage.convolve, same image took 8 minutes but less memory.
        annular_filtered = utils.morphology.imfilter(image, se/se.sum(), padding=0, corr_or_conv='corr')
        isolated = annular_filtered[tuple([spot_yxz[:, j] for j in range(image.ndim)])]
    else:
        isolated = utils.morphology.imfilter_coords(image, se, spot_yxz, padding=0, corr_or_conv='corr') / np.sum(se)
    return isolated < thresh


def check_neighbour_intensity(image: np.ndarray, spot_yxz: np.ndarray, thresh: float = 0) -> np.ndarray:
    """
    Checks whether a neighbouring pixel to those indicated in ```spot_yxz``` has intensity less than ```thresh```.
    The idea is that if pixel has very low intensity right next to it, it is probably a spurious spot.

    Args:
        image: ```float [n_y x n_x x n_z]```.
            image spots were found on.
        spot_yxz: ```int [n_peaks x image.ndim]```.
            yx or yxz location of spots found.
            If axis 1 dimension is more than ```image.ndim```, only first ```image.ndim``` dimensions used
            i.e. if supply yxz, with 2d image, only yx position used.
        thresh: Spots are indicated as ```False``` if intensity at neighbour to spot location is less than this.

    Returns:
        ```float [n_peaks]```.
            ```True``` if no neighbours below thresh.
    """
    if image.ndim == 3:
        transforms = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    elif image.ndim == 2:
        transforms = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    else:
        raise ValueError(f"image has to have two or three dimensions but given image has {image.ndim} dimensions.")
    keep = np.zeros((spot_yxz.shape[0], len(transforms)), dtype=bool)
    for i, t in enumerate(transforms):
        mod_spot_yx = spot_yxz + t
        for j in range(image.ndim):
            mod_spot_yx[:, j] = np.clip(mod_spot_yx[:, j], 0, image.shape[j]-1)
        keep[:, i] = image[tuple([mod_spot_yx[:, j] for j in range(image.ndim)])] > thresh
    return keep.min(axis=1)


def get_isolated_points(spot_yxz: np.ndarray, isolation_dist: float) -> np.ndarray:
    """
    Get the isolated points in a point cloud as those whose neighbour is far.

    Args:
        spot_yxz: ```int [n_peaks x image.ndim]```.
            yx or yxz location of spots found in image.
        isolation_dist: Spots are isolated if nearest neighbour is further away than this.

    Returns:
        ```bool [n_peaks]```. ```True``` for points far from any other point in ```spot_yx```.

    """
    tree = KDTree(spot_yxz)
    # for distances more than isolation_dist, distances will be set to infinity i.e. will be > isolation_dist.
    distances, _ = tree.query(spot_yxz, k=[2], distance_upper_bound=isolation_dist)[0].squeeze()
    return distances > isolation_dist
