import iss.utils.morphology as morphology
import numpy as np

import iss.utils.strel


def detect_spots(image, intensity_thresh, radius_xy, radius_z=None, remove_duplicates=True):
    """
    finds local maxima in image exceeding intensity_thresh.

    :param image: numpy array [nY x nX x nZ]
    :param intensity_thresh: float
        spots are local maxima in image with pixel value > intensity_thresh
    :param radius_xy: integer
        radius of dilation structuring element in xy plane (approximately spot radius)
    :param radius_z: integer, optional.
        radius of dilation structuring element in z direction (approximately spot radius)
        default: None, meaning 2D filter used.
    :param remove_duplicates: boolean, optional.
        Whether to only keep one pixel if two or more pixels are local maxima and have same intensity.
        default: True.
    :return:
    peak_yx: numpy integer array [n_peaks x image.ndim]
        yx or yxz location of spots found.
    peak_intensity: numpy float array [n_peaks] pixel value of spots found.
    """
    if radius_z is not None:
        se = iss.utils.strel.disk_3d(radius_xy, radius_z)
    else:
        se = iss.utils.strel.disk(radius_xy)
    small = 1e-6  # for computing local maxima: shouldn't matter what it is (keep below 0.01 for int image).
    if remove_duplicates:
        # perturb image by small amount so two neighbouring pixels that did have the same value now differ slightly.
        # hence when find maxima, will only get one of the pixels not both.
        np.random.seed(0)   # So shift is always the same.
        # rand_shift must be larger than small to detect a single spot.
        rand_im_shift = np.random.uniform(low=small*2, high=0.2, size=image.shape)
        image = image + rand_im_shift

    dilate = morphology.dilate(image, se)
    spots = np.logical_and(image + small > dilate, image > intensity_thresh)
    peak_pos = np.where(spots)
    peak_yx = np.concatenate([coord.reshape(-1, 1) for coord in peak_pos], axis=1)
    peak_intensity = image[spots]
    return peak_yx, peak_intensity


def get_isolated(image, spot_yx, thresh, radius_inner, radius_xy, radius_z=None):
    """
    determines whether each spot in spot_yx is isolated by getting the value of image after annular filtering
    at each location in spot_yx.

    :param image: numpy array [nY x nX x nZ]
        image spots were found on.
    :param spot_yx: numpy integer array [n_peaks x image.ndim]
        yx or yxz location of spots found.
    :param thresh: float
        spots are isolated if annulus filtered image at spot location less than this.
    :param radius_inner: float
        inner radius of annulus filtering kernel within which values are all zero.
    :param radius_xy: float
        outer radius of annulus filtering kernel in xy direction.
    :param radius_z: float, optional.
        outer radius of annulus filtering kernel in z direction
        default: None meaning 2d filter used.
    :return: numpy boolean array [n_peaks] indicated whether each spot is isolated or not.
    """
    se = iss.utils.strel.annulus(radius_inner, radius_xy, radius_z)
    annular_filtered = morphology.imfilter(image, se/se.sum(), padding=0, corr_or_conv='corr')
    isolated = annular_filtered[tuple([spot_yx[:, j] for j in range(image.ndim)])] < thresh
    return isolated


def check_neighbour_intensity(image, spot_yx, thresh=0):
    """
    checks whether a neighbouring pixel to those indicated in spot_yx has intensity less than thresh.
    idea is that if pixel has very low intensity right next to it, it is probably a spurious spot.

    :param image: numpy array [nY x nX x nZ]
        image spots were found on.
    :param spot_yx: numpy integer array [n_peaks x image.ndim]
        yx or yxz location of spots found.
    :param thresh: float, optional.
        spots are indicated as false if intensity at neighbour to spot location is less than this.
        default: 0.
    :return: numpy boolean array [n_peaks]. True if no neighbours below thresh.
    """
    if image.ndim == 3:
        transforms = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    elif image.ndim == 2:
        transforms = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    else:
        raise ValueError(f"image has to have two or three dimensions but given image has {image.ndim} dimensions.")
    keep = np.zeros((spot_yx.shape[0], len(transforms)), dtype=bool)
    for i, t in enumerate(transforms):
        mod_spot_yx = spot_yx + t
        for j in range(image.ndim):
            mod_spot_yx[:, j] = np.clip(mod_spot_yx[:, j], 0, image.shape[j]-1)
        keep[:, i] = image[tuple([mod_spot_yx[:, j] for j in range(image.ndim)])] > thresh
    return keep.min(axis=1)
