import nd2
import numpy as np
from tqdm import tqdm
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from sklearn.linear_model import HuberRegressor
from scipy.spatial import KDTree
from scipy.ndimage import affine_transform
from scipy import stats
from skimage.filters import sobel
from coppafish.register.preprocessing import custom_shift, split_3d_image, replace_scale, populate_full, \
    merge_subvols, yxz_to_zyx_affine, yxz_to_zyx, NotebookPage
from skimage.registration import phase_cross_correlation
from multiprocessing import Queue


def find_shift_array(subvol_base, subvol_target, position, r_threshold):
    """
    This function takes in 2 3d images which have already been split up into 3d subvolumes. We then find the shift from
    each base subvolume to its corresponding target subvolume.
    NOTE: This function does allow matching to non-corresponding subvolumes in z.
    NOTE: This function performs flattening of the output arrays, and this is done according to np.reshape.
    Args:
        subvol_base: Base subvolume array (n_z_subvolumes, n_y_subvolumes, n_x_subvolumes, z_box, y_box, x_box)
        subvol_target: Target subvolume array (n_z_subvolumes, n_y_subvolumes, n_x_subvolumes, z_box, y_box, x_box)
        position: Position of centre of subvolumes in base array (n_z_subvolumes, n_y_subvolumes, n_x_subvolumes, 3)
        r_threshold: measure of shift quality. If the correlation between corrected base subvol and fixed target
        subvol is beneath this, we store shift as [nan, nan, nan]
    Returns:
        shift: 2D array, with first dimension referring to subvolume index and final dim referring to shift
        (n_z_subvolumes * n_y_subvolumes * n_x_subvolumes, 3)
        shift_corr: 2D array, with first dimension referring to subvolume index and final dim referring to
        shift_corr coef (n_z_subvolumes * n_y_subvolumes * n_x_subvolumes, 1)
    """
    if subvol_base.shape != subvol_target.shape:
        raise ValueError("Subvolume arrays have different shapes")
    z_subvolumes, y_subvolumes, x_subvolumes = subvol_base.shape[0], subvol_base.shape[1], subvol_base.shape[2]
    shift = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, 3))
    shift_corr = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes))
    position = np.reshape(position, (z_subvolumes, y_subvolumes, x_subvolumes, 3))

    for y in range(y_subvolumes):
        for x in range(x_subvolumes):
            shift[:, y, x], shift_corr[:, y, x] = find_z_tower_shifts(subvol_base=subvol_base[:, y, x],
                                                                      subvol_target=subvol_target[:, y, x],
                                                                      position=position[:, y, x].copy(),
                                                                      pearson_r_threshold=r_threshold)

    return np.reshape(shift, (shift.shape[0] * shift.shape[1] * shift.shape[2], 3)), \
        np.reshape(shift_corr, shift.shape[0] * shift.shape[1] * shift.shape[2])


def find_z_tower_shifts(subvol_base, subvol_target, position, pearson_r_threshold, z_neighbours=1):
    """
    This function takes in 2 split up 3d images along one tower of subvolumes in z and computes shifts for each of these
    to its nearest neighbour subvol
    Args:
        subvol_base: (n_z_subvols, z_box, y_box, x_box) Base subvolume array (this is a z-tower of n_z_subvols base
        subvolumes)
        subvol_target: (n_z_subvols, z_box, y_box, x_box) Target subvolume array (this is a z-tower of n_z_subvols
        target subvolumes)
        position: (n_z_subvols, 3) Position of centre of subvolumes (z, y, x)
        pearson_r_threshold: (float) threshold of correlation used in degenerate cases
        z_neighbours: (int) number of neighbouring sub-volumes to merge with the current sub-volume to compute the shift
    Returns:
        shift: (n_z_subvols, 3) shift of each subvolume in z-tower
        shift_corr: (n_z_subvols, 1) correlation coefficient of each subvolume in z-tower
    """
    position = position.astype(int)
    # for the purposes of this section, we'll take position to be the bottom left corner of the subvolume
    position = position - np.array([subvol_base.shape[1], subvol_base.shape[2], subvol_base.shape[3]]) // 2
    z_subvolumes = subvol_base.shape[0]
    z_box = subvol_base.shape[1]
    shift = np.zeros((z_subvolumes, 3))
    shift_corr = np.zeros(z_subvolumes)
    for z in range(z_subvolumes):
        z_start, z_end = int(max(0, z - z_neighbours)), int(min(z_subvolumes, z + z_neighbours + 1))
        merged_subvol_target = merge_subvols(position=np.copy(position[z_start:z_end]),
                                             subvol=subvol_target[z_start:z_end])
        merged_subvol_base = np.zeros(merged_subvol_target.shape)
        box_bottom = position[z_start, 0]
        merged_subvol_base[position[z, 0] - box_bottom:position[z, 0] - box_bottom + z_box] = subvol_base[z]
        # Now we have the merged subvolumes, we can compute the shift
        shift[z], shift_corr[z] = find_zyx_shift(subvol_base=merged_subvol_base, subvol_target=merged_subvol_target,
                                                 pearson_r_threshold=pearson_r_threshold)

    return shift, shift_corr


def find_zyx_shift(subvol_base, subvol_target, pearson_r_threshold=0.9):
    """
    This function takes in 2 3d images and finds the optimal shift from one to the other. We use a phase cross
    correlation method to find the shift.
    Args:
        subvol_base: Base subvolume array (this will contain a lot of zeroes) (n_z_pixels, n_y_pixels, n_x_pixels)
        subvol_target: Target subvolume array (this will be a merging of subvolumes with neighbouring subvolumes)
        (nz_pixels2, n_y_pixels2, n_x_pixels2) size 2 >= size 1
        pearson_r_threshold: Threshold used to accept a shift as valid (float)

    Returns:
        shift: zyx shift (3,)
        shift_corr: correlation coefficient of shift (float)
    """
    if subvol_base.shape != subvol_target.shape:
        raise ValueError("Subvolume arrays have different shapes")
    shift, _, _ = phase_cross_correlation(reference_image=subvol_target, moving_image=subvol_base, upsample_factor=10)
    alt_shift = np.copy(shift)
    # now anti alias the shift in z. To do this, consider that the other possible aliased z shift is the either one
    # subvolume above or below the current shift. (In theory, we could also consider the subvolume 2 above or below,
    # but this is unlikely to be the case in practice as we are already merging subvolumes)
    if shift[0] > 0:
        alt_shift[0] = shift[0] - subvol_base.shape[0]
    else:
        alt_shift[0] = shift[0] + subvol_base.shape[0]

    # Now we need to compute the correlation coefficient of the shift and the anti aliased shift
    shift_base = custom_shift(subvol_base, shift.astype(int))
    alt_shift_base = custom_shift(subvol_base, alt_shift.astype(int))
    # Now compute the correlation coefficients. First create a mask of the nonzero values
    mask = shift_base != 0
    shift_corr = np.corrcoef(shift_base[mask], subvol_target[mask])[0, 1]
    mask = alt_shift_base != 0
    alt_shift_corr = np.corrcoef(alt_shift_base[mask], subvol_target[mask])[0, 1]
    mask = subvol_base != 0
    base_corr = np.corrcoef(subvol_base[mask], subvol_target[mask])[0, 1]

    # Now return the shift with the highest correlation coefficient
    if alt_shift_corr > shift_corr:
        shift = alt_shift
        shift_corr = alt_shift_corr
    if base_corr > shift_corr:
        shift = np.array([0, 0, 0])
        shift_corr = base_corr
    # Now check if the correlation coefficient is above the threshold. If not, set the shift to nan
    if shift_corr < pearson_r_threshold:
        shift = np.array([np.nan, np.nan, np.nan])
        shift_corr = np.max([shift_corr, alt_shift_corr, base_corr])

    return shift, shift_corr


# ols regression to find transform from shifts
def ols_regression(shift, position):
    """
    Args:
        shift: (z_sv x y_sv x x_sv) x 3 array which of shifts in zyx format
        position: (z_sv x y_sv x x_sv) x 3 array which of positions in zyx format
    Returns:
        transform: 3 x 4 affine transform in yxz format with final col being shift
    """

    # We are going to get rid of the shifts where any of the values are nan for regression
    position = position[~np.isnan(shift[:, 0])]
    shift = shift[~np.isnan(shift[:, 0])]

    new_position = position + shift
    position = np.vstack((position.T, np.ones(shift.shape[0]))).T

    # Now compute the regression
    transform, _, _, _ = np.linalg.lstsq(position, new_position, rcond=None)

    # Unsure what taking transpose means for off diagonals here
    return transform.T


def huber_regression(shift, position, predict_shift=True):
    """
    Function to predict shift as a function of position using robust huber regressor. If we do not have >= 3 z-coords
    in position, the z-coords of the affine transform will be estimated as no scaling, and a shift of mean(shift).
    Args:
        shift: n_tiles x 3 ndarray of zyx shifts
        position: n_tiles x 2 ndarray of yx tile coords or n_tiles x 3 ndarray of zyx tile coords
        predict_shift: If True, predict shift as a function of position. If False, predict position as a function of
        position. Default is True. Difference is that if false, we add 1 to each diagonal of the transform matrix.
    Returns:
        transform: 3 x 3 matrix where each row predicts shift of z y z as a function of y index, x index and the final
        row is the offset at 0,0
        or 3
    """
    # We are going to get rid of the shifts where any of the values are nan for regression
    position = position[~np.isnan(shift[:, 0])]
    shift = shift[~np.isnan(shift[:, 0])]
    # Check if we have any shifts to predict
    if len(shift) == 0 and predict_shift:
        transform = np.zeros((3, 4))
        return transform
    elif len(shift) == 0 and not predict_shift:
        transform = np.eye(3, 4)
        return transform
    # Do robust regression
    # Check we have at least 3 z-coords in position
    if len(set(position[:, 0])) <= 2:
        z_coef = np.array([0, 0, 0])
        z_shift = np.mean(shift[:, 0])
        # raise a warning if we have less than 3 z-coords in position
        print('Warning: Less than 3 z-coords in position. Setting z-coords of transform to no scaling and shift of '
              'mean(shift)')
    else:
        huber_z = HuberRegressor(epsilon=2, max_iter=400, tol=1e-4).fit(X=position, y=shift[:, 0])
        z_coef = huber_z.coef_
        z_shift = huber_z.intercept_
    huber_y = HuberRegressor(epsilon=2, max_iter=400, tol=1e-4).fit(X=position, y=shift[:, 1])
    huber_x = HuberRegressor(epsilon=2, max_iter=400, tol=1e-4).fit(X=position, y=shift[:, 2])
    transform = np.vstack((np.append(z_coef, z_shift),
                           np.append(huber_y.coef_, huber_y.intercept_),
                           np.append(huber_x.coef_, huber_x.intercept_)))
    if not predict_shift:
        transform += np.eye(3, 4)

    return transform


# Bridge function for all functions in round registration
def round_registration(anchor_image: np.ndarray, round_image: list, config: dict) -> dict:
    """
    Function to carry out sub-volume round registration on a single tile.
    Args:
        anchor_image: np.ndarray size [n_z, n_y, n_x] of the anchor image
        round_image: list of length n_rounds, where round_image[r] is  np.ndarray size [n_z, n_y, n_x] of round r
        config: dict of configuration parameters for registration

    Returns:
        round_registration_data: dictionary containing the following keys:
        'position': np.ndarray size [z_sv x y_sv x x_sv, 3] of the position of each sub-volume in zyx format
        'shift': np.ndarray size [z_sv x y_sv x x_sv, 3] of the shift of each sub-volume in zyx format
        'shift_corr': np.ndarray size [z_sv x y_sv x x_sv] of the correlation coefficient of each sub-volume shift
        'transform': np.ndarray size [n_rounds, 3, 4] of the affine transform for each round in zyx format
    """
    # Initialize variables from config
    z_subvols, y_subvols, x_subvols = config['subvols']
    z_box, y_box, x_box = config['box_size']
    r_thresh = config['pearson_r_thresh']

    # Create the directory for the round registration data
    round_registration_data = {'position': [], 'shift': [], 'shift_corr': [], 'transform': []}

    if config['sobel']:
        anchor_image = sobel(anchor_image)
        round_image = [sobel(r) for r in round_image]

    # Now compute round shifts for this tile and the affine transform for each round
    pbar = tqdm(total=len(round_image), desc='Computing round transforms')
    for r in range(len(round_image)):
        # Set progress bar title
        pbar.set_description('Computing shifts for round ' + str(r))

        # next we split image into overlapping cuboids
        subvol_base, position = split_3d_image(image=anchor_image, z_subvolumes=z_subvols,
                                               y_subvolumes=y_subvols, x_subvolumes=x_subvols,
                                               z_box=z_box, y_box=y_box, x_box=x_box)
        subvol_target, _ = split_3d_image(image=round_image[r], z_subvolumes=z_subvols, y_subvolumes=y_subvols,
                                          x_subvolumes=x_subvols, z_box=z_box, y_box=y_box, x_box=x_box)

        # Find the subvolume shifts
        shift, corr = find_shift_array(subvol_base, subvol_target, position=position.copy(), r_threshold=r_thresh)
        transform = huber_regression(shift, position, predict_shift=False)
        adjusted_target = affine_transform(round_image[r], transform, order=0)
        global_shift_correction = phase_cross_correlation(reference_image=adjusted_target, moving_image=anchor_image)[0]
        transform[:, 3] += global_shift_correction
        # Append these arrays to the round_shift, round_shift_corr, round_transform and position storage
        round_registration_data['shift'].append(shift)
        round_registration_data['shift_corr'].append(corr)
        round_registration_data['position'].append(position)
        round_registration_data['transform'].append(transform)
        pbar.update(1)
    pbar.close()
    # Convert all lists to numpy arrays
    for key in round_registration_data.keys():
        round_registration_data[key] = np.array(round_registration_data[key])

    return round_registration_data


def channel_registration(fluorescent_bead_path: str = None, anchor_cam_idx: int = 2, n_cams: int = 4,
                         bead_radii: list = [10, 11, 12]) -> np.ndarray:
    """
    Function to carry out channel registration using fluorescent beads. This function assumes that the fluorescent
    bead images are 2D images and that there is one image for each camera.

    Fluorescent beads are assumed to be circles and are detected using the Hough transform. The centre of the detected
    circles are then used to compute the affine transform between each camera and the round_reg_channel_cam via ICP.

    Args:
        fluorescent_bead_path: Path to fluorescent beads directory containing the fluorescent bead images.
        If none then we assume that the channels are registered to each other and just set channel_transforms to
        identity matrices
        anchor_cam_idx: int index of the camera to use as the anchor camera
        n_cams: int number of cameras
        bead_radii: list of possible bead radii

    Returns:
        transform: n_cams x 3 x 4 array of affine transforms taking anchor camera to each other camera
    """
    transform = np.zeros((n_cams, 3, 4))
    # First check if the fluorescent beads path exists. If not, we assume that the channels are registered to each
    # other and just set channel_transforms to identity matrices
    if fluorescent_bead_path is None:
        # Set registration_data['channel_registration']['channel_transform'][c] = np.eye(3) for all channels c
        for c in range(n_cams):
            transform[c] = np.eye(3, 4)
        print('Fluorescent beads directory does not exist. Prior assumption will be that channels are registered to '
              'each other.')
        return transform

    # open the fluorescent bead images as nd2 files
    with nd2.ND2File(fluorescent_bead_path) as fbim:
        fluorescent_beads = fbim.asarray()
    # if fluorescent bead images are for all channels, just take one from each camera

    # TODO better deal with 3D
    if len(fluorescent_beads.shape) == 4:
        nz = fluorescent_beads.shape[0]
        fluorescent_beads = fluorescent_beads[int(nz/2), :, :, :]

    if fluorescent_beads.shape[0] == 28:
        fluorescent_beads = fluorescent_beads[[0, 9, 18, 23]]

    # Now we'll turn each image into a point cloud
    bead_point_clouds = []
    for i in range(n_cams):
        edges = canny(fluorescent_beads[i],  sigma=3, low_threshold=10, high_threshold=50)
        hough_res = hough_circle(edges, bead_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, bead_radii, min_xdistance=10, min_ydistance=10)
        bead_point_clouds.append(np.vstack((cy, cx)).T)

    # Now convert the point clouds from yx to yxz. This is because our ICP algorithm assumes that the point clouds
    # are in 3D space
    bead_point_clouds = [np.hstack((bead_point_clouds[i], np.zeros((bead_point_clouds[i].shape[0], 1))))
                         for i in range(n_cams)]

    # Set up ICP
    initial_transform = np.zeros((n_cams, 4, 3))
    transform = np.zeros((n_cams, 4, 3))
    mse = np.zeros((n_cams, 50))
    target_cams = [i for i in range(n_cams) if i != anchor_cam_idx]
    with tqdm(total=len(target_cams)) as pbar:
        for i in target_cams:
            pbar.set_description('Running ICP for camera ' + str(i))
            # Set initial transform to identity (shift already accounted for)
            initial_transform[i, :3, :3] = np.eye(3)
            # Run ICP
            transform[i], _, mse[i], converged = icp(yxz_base=bead_point_clouds[anchor_cam_idx],
                                                     yxz_target=bead_point_clouds[i],
                                                     start_transform=initial_transform[i], n_iters=50, dist_thresh=5)
            if not converged:
                transform[i] = np.eye(4, 3)
                raise Warning('ICP did not converge for camera ' + str(i) + '. Replacing with identity.')
            pbar.update(1)

    # Need to add in z coord info as not accounted for by registration due to all coords being equal
    transform[:, 2, 2] = 1
    transform[anchor_cam_idx] = np.eye(4, 3)

    # Convert transforms from yxz to zyx
    transform_zyx = np.zeros((4, 3, 4))
    for i in range(4):
        transform_zyx[i] = yxz_to_zyx_affine(transform[i])

    return transform_zyx


# Function to remove outlier shifts
def regularise_shift(shift, tile_origin, residual_threshold):
    """
    Function to remove outlier shifts from a collection of shifts whose start location can be determined by position.
    Uses robust linear regression to compute a prediction of what the shifts should look like and if any shift differs
    from this by more than some threshold it is declared to be an outlier.
    Args:
        shift: Either [n_tiles_use x n_rounds_use x 3] or [n_tiles_use x n_channels_use x 3]
        tile_origin: yxz positions of the tiles [n_tiles_use x 3]
        residual_threshold: This is a threshold above which we will consider a point to be an outlier

    Returns:
        shift_regularised: Either [n_tiles x n_rounds x 4 x 3] or [n_tiles x x_channels x 4 x 3]
    """
    # rearrange columns so that tile origins are in zyx like the shifts are, then initialise commonly used variables
    rc_use = np.arange(shift.shape[1])
    n_tiles_use = tile_origin.shape[0]
    predicted_shift = np.zeros_like(shift)

    # Compute predicted shift for each round/channel from X = position for each tile and Y = shift for each tile
    for elem in rc_use:
        big_transform = ols_regression(shift[:, elem], tile_origin)
        padded_origin = np.vstack((tile_origin.T, np.ones(n_tiles_use)))
        predicted_shift[:, elem] = (big_transform @ padded_origin).T - tile_origin

    # Use these predicted shifts to get rid of outliers
    # The nice thing about this approach is that we can immediately replace the outliers with their prediction
    residual = np.linalg.norm(predicted_shift-shift, axis=2)
    shift_outlier = residual > residual_threshold
    shift[shift_outlier] = predicted_shift[shift_outlier]

    return shift


# Function to remove outlier round scales
def regularise_round_scaling(scale: np.ndarray):
    """
    Function to remove outliers in the scale parameters for round transforms. Experience shows these should be close
    to 1 for x and y regardless of round and should be increasing or decreasing for z dependent on whether anchor
    round came before or after.

    Args:
        scale: 3 x n_tiles_use x n_rounds_use ndarray of z y x scales for each tile and round

    Returns:
        scale_regularised:  n_tiles_use x n_rounds_use x 3 ndarray of z y x regularised scales for each tile and round
    """
    # Define num tiles and separate the z scale and yx scales for different analysis
    n_tiles = scale.shape[1]
    z_scale = scale[0]
    yx_scale = scale[1:]

    # Regularise z_scale. Expect z_scale to vary by round but not by tile.
    # We classify outliers in z as:
    # a.) those that lie outside 1 iqr of the median for z_scales of that round
    # b.) those that increase when the majority decrease or those that decrease when majority increase
    # First carry out removal of outliers of type (a)
    median_z_scale = np.repeat(np.median(z_scale, axis=0)[np.newaxis, :], n_tiles, axis=0)
    iqr_z_scale = np.repeat(stats.iqr(z_scale, axis=0)[np.newaxis, :], n_tiles, axis=0)
    outlier_z = np.abs(z_scale - median_z_scale) > iqr_z_scale
    z_scale[outlier_z] = median_z_scale[outlier_z]

    # Now carry out removal of outliers of type (b)
    delta_z_scale = np.diff(z_scale, axis=1)
    dominant_sign = np.sign(np.median(delta_z_scale))
    outlier_z = np.hstack((np.sign(delta_z_scale) != dominant_sign, np.zeros((n_tiles, 1), dtype=bool)))
    z_scale[outlier_z] = median_z_scale[outlier_z]

    # Regularise yx_scale: No need to account for variation in tile or round
    outlier_yx = np.abs(yx_scale - 1) > 0.01
    yx_scale[outlier_yx] = 1

    return scale


# Bridge function for all regularisation
def regularise_transforms(registration_data: dict, tile_origin: np.ndarray, residual_threshold: float,
                          use_tiles: list, use_rounds: list):
    """
    Function to regularise affine transforms by comparing them to affine transforms from other tiles.
    As the channel transforms do not depend on tile, they do not need to be regularised.
    Args:
        registration_data: dictionary of registration data
        tile_origin: n_tiles x 3 tile origin in zyx
        residual_threshold: threshold for classifying outlier shifts
        use_tiles: list of tiles in use
        use_rounds: list of rounds in use

    Returns:
        registration_data: dictionary of registration data with regularised transforms
    """
    # Extract transforms
    round_transform = np.copy(registration_data['round_registration']['transform_raw'])

    # Code becomes easier when we disregard tiles, rounds, channels not in use
    n_tiles, n_rounds = round_transform.shape[0], round_transform.shape[1]
    tile_origin = tile_origin[use_tiles]
    round_transform = round_transform[use_tiles][:, use_rounds]

    # Regularise round transforms
    round_transform[:, :, :, 3] = regularise_shift(shift=round_transform[:, :, :, 3], tile_origin=tile_origin,
                                                   residual_threshold=residual_threshold)
    round_scale = np.array([round_transform[:, :, 0, 0], round_transform[:, :, 1, 1], round_transform[:, :, 2, 2]])
    round_scale_regularised = regularise_round_scaling(round_scale)
    round_transform = replace_scale(transform=round_transform, scale=round_scale_regularised)
    round_transform = populate_full(sublist_1=use_tiles, list_1=np.arange(n_tiles),
                                    sublist_2=use_rounds, list_2=np.arange(n_rounds),
                                    array=round_transform)

    registration_data['round_registration']['transform'] = round_transform

    return registration_data


# Function which runs a single iteration of the icp algorithm
def get_transform(yxz_base: np.ndarray, yxz_target: np.ndarray, transform_old: np.ndarray, dist_thresh: float,
                  robust=False):
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
            Typical: ```5```.
        robust: Boolean option to make regression robust. Selecting true will result in the algorithm maximising
            correntropy as opposed to minimising mse.

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
    # Step 1 computes matching, since yxz_target is a subset of yxz_base, we will loop through yxz_target and find
    # their nearest neighbours within yxz_transform, which is the initial transform applied to yxz_base
    yxz_base_pad = np.pad(yxz_base, [(0, 0), (0, 1)], constant_values=1)
    yxz_transform = yxz_base_pad @ transform_old
    yxz_transform_tree = KDTree(yxz_transform)
    # the next query works the following way. For each point in yxz_target, we look for the closest neighbour in the
    # anchor, which we have now applied the initial transform to. If this is below dist_thresh, we append its distance
    # to distances and append the index of this neighbour to neighbour
    distances, neighbour = yxz_transform_tree.query(yxz_target, distance_upper_bound=dist_thresh)
    neighbour = neighbour.flatten()
    distances = distances.flatten()
    use = distances < dist_thresh
    n_matches = np.sum(use)
    error = np.sqrt(np.mean(distances[use] ** 2))
    base_pad_use = yxz_base_pad[neighbour[use], :]
    target_use = yxz_target[use, :]

    if not robust:
        transform = np.linalg.lstsq(base_pad_use, target_use, rcond=None)[0]
    else:
        sigma = dist_thresh / 2
        target_pad_use = np.pad(target_use, [(0, 0), (0, 1)], constant_values=1)
        D = np.diag(np.exp(-0.5 * (np.linalg.norm(base_pad_use @ transform_old - target_use, axis=1)/sigma) ** 2))
        transform = (target_pad_use.T @ D @ base_pad_use @ np.linalg.inv(base_pad_use.T @ D @ base_pad_use))[:3, :4].T

    return transform, neighbour, n_matches, error


# Simple ICP implementation, calls above until no change
def icp(yxz_base, yxz_target, dist_thresh, start_transform, n_iters, robust=False):
    """
    Applies n_iters rounds of the above least squares regression
    Args:
        yxz_base: ```float [n_base_spots x 3]```.
            Coordinates of spots you want to transform.
        yxz_target: ```float [n_target_spots x 3]```.
            Coordinates of spots in image that you want to transform ```yxz_base``` to.
        start_transform: initial transform
        dist_thresh: If neighbours closer than this, they are used to compute the new transform.
            Typical: ```3```.
        n_iters: max number of times to compute regression
        robust: whether to compute robust icp
    Returns:
        - ```transform``` - ```float [4 x 3]```.
            Updated affine transform.
        - ```n_matches``` - ```int```.
            Number of neighbours which have distance less than ```dist_thresh```.
        - ```error``` - ```float```.
            Average distance between ```neighbours``` below ```dist_thresh```.
        - converged - bool
            True if completed in less than n_iters and false o/w
    """
    # initialise transform
    transform = start_transform
    n_matches = np.zeros(n_iters)
    error = np.zeros(n_iters)
    prev_neighbour = np.ones(yxz_target.shape[0]) * yxz_base.shape[0]

    # Update transform. We want this to have max n_iters iterations. We will end sooner if all neighbours do not change
    # in 2 successive iterations. Define the variables for iteration 0 before we start the loop
    transform, neighbour, n_matches[0], error[0] = get_transform(yxz_base, yxz_target, transform, dist_thresh, robust)
    i = 0
    while i + 1 < n_iters and not all(prev_neighbour == neighbour):
        # update i and prev_neighbour
        prev_neighbour, i = neighbour, i + 1
        transform, neighbour, n_matches[i], error[i] = get_transform(yxz_base, yxz_target, transform, dist_thresh,
                                                                     robust)
    # now fill in any variables that were not completed due to early exit
    n_matches[i:] = n_matches[i] * np.ones(n_iters - i)
    error[i:] = error[i] * np.ones(n_iters - i)
    converged = i < n_iters

    return transform, n_matches, error, converged


def brightness_scale(preseq: np.ndarray, seq: np.ndarray, intensity_percentile: int = 99, 
                     sub_image_size: int = 500):
    """
    Function to find scale factor m and constant c such that m * preseq + c ~ seq. This is done by a regression on
    the pixels of brightness < brightness_thresh_percentile as these likely won't be spots. The regression is done by
    least squares.

    This function isn't really registration but needs registration to be run before it can be used and here seems
    like a good place to put it.
    Args:
        preseq: (n_z x n_y x n_x) ndarray of presequence image.
        seq: (n_z x n_y x n_x) ndarray of sequence image.
        intensity_percentile: float brightness percentile such that all pixels with brightness > this percentile
        (in the preseq im only) are used for regression. Default: `99`.
        sub_image_size: int size of sub-images to use for regression. This is because the images are not perfectly
            registered and so we need to find the best registered sub-image to use for regression. Default: `500`.

    Returns:
        scale: float scale factor `m`.
        sub_image_seq: (sub_image_size, sub_image_size) ndarray of sequence image used for regression.
        sub_image_preseq: (sub_image_size, sub_image_size) ndarray of presequence image used for regression.
    """
    # Find the bottom intensity_percentile pixels from the image to linear regress with to exclude any spots
    # Create 2 masks, and take the intersection of them
    # Super important to have very well registered images for this to work, so to do this, we'll scan through 
    # small sub-images, and find the shifts between them. We'll then do the brightness matching on the best 
    # registered sub-images
    assert preseq.shape == seq.shape, "Presequence and sequence images must have the same shape"
    tile_size = seq.shape[-1]
    n_sub_images = int(tile_size / sub_image_size)
    sub_image_shifts = np.zeros((n_sub_images, n_sub_images, 3))
    sub_image_shift_score = np.zeros((n_sub_images, n_sub_images))
    for i in range(n_sub_images):
        for j in range(n_sub_images):
            sub_image_preseq = preseq[:, i * sub_image_size:(i + 1) * sub_image_size,
                              j * sub_image_size:(j + 1) * sub_image_size]
            sub_image_seq = seq[:, i * sub_image_size:(i + 1) * sub_image_size,
                            j * sub_image_size:(j + 1) * sub_image_size]
            sub_image_shifts[i, j] = phase_cross_correlation(sub_image_preseq, sub_image_seq)[0]
            sub_image_shift_score[i, j] = np.corrcoef(sub_image_preseq.ravel(),
                                                      custom_shift(sub_image_seq, sub_image_shifts[i, j].astype(int))
                                                      .ravel())[0, 1]
    # Now find the best sub-image
    if np.sum(np.isnan(sub_image_shift_score)) == n_sub_images ** 2:
        print('Warning: No sub-image shifts found. Setting scale to 1 and returning original images.')
        return 1, sub_image_seq, sub_image_preseq
    best_sub_image = np.argwhere(sub_image_shift_score == np.nanmax(sub_image_shift_score))[0]
    sub_image_seq = seq[:, best_sub_image[0] * sub_image_size:(best_sub_image[0] + 1) * sub_image_size,
                        best_sub_image[1] * sub_image_size:(best_sub_image[1] + 1) * sub_image_size]
    sub_image_preseq = preseq[:, best_sub_image[0] * sub_image_size:(best_sub_image[0] + 1) * sub_image_size,
                              best_sub_image[1] * sub_image_size:(best_sub_image[1] + 1) * sub_image_size]
    sub_image_seq = custom_shift(sub_image_seq, sub_image_shifts[best_sub_image[0], best_sub_image[1]].astype(int))

    # Now find the top intensity_percentile pixels from the image to linear regress with any background
    mask = sub_image_preseq > np.percentile(sub_image_preseq, intensity_percentile)

    sub_image_preseq_flat = sub_image_preseq[mask].ravel()
    sub_image_seq_flat = sub_image_seq[mask].ravel()
    # Least squares to find im = m * im_pre best fit coefficients
    m = np.linalg.lstsq(sub_image_preseq_flat[:, None], sub_image_seq_flat, rcond=None)[0]
    
    return m, sub_image_seq, sub_image_preseq


def compute_brightness_scale(nbp: NotebookPage, nbp_basic: NotebookPage, nbp_file: NotebookPage, 
                             nbp_extract: NotebookPage, mid_z: int, z_rad: int, t: int, r: int, c: int, 
                             queue: Queue = None):
    """
    Applies affine transforms to the presequence and sequence image for tile t, round r, channel c. Then applies 
    `brightness_scale`.

    Args:
        nbp (NotebookPage): `register` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_extract (NotebookPage): `extract` notebook page.
        mid_z (int): middle z plane to be used.
        z_rad (int): span on z planes, +- from `mid_z`.
        t (int): tile index.
        r (int): round index.
        c (int): channel index.
        queue (Queue, optional): multiprocess `Queue` object, the return is `put` into this object. Default: None, no 
            queue object.
    """
    if nbp_extract.file_type == '.npy':
        from ..utils.npy import load_tile
    elif nbp_extract.file_type == '.zarr':
        from ..utils.zarray import load_tile

    transform_pre = yxz_to_zyx_affine(nbp.transform[t, nbp_basic.pre_seq_round, c],
                                        new_origin=np.array([mid_z-z_rad, 0, 0]))
    transform_seq = yxz_to_zyx_affine(nbp.transform[t, r, c], new_origin=np.array([mid_z-z_rad, 0, 0]))
    preseq = yxz_to_zyx(load_tile(nbp_file, nbp_basic, t=t, r=nbp_basic.pre_seq_round, c=c,
                                    yxz=[None, None, np.arange(mid_z-z_rad, mid_z+z_rad)]))
    seq = yxz_to_zyx(load_tile(nbp_file, nbp_basic, t=t, r=r, c=c,
                                yxz=[None, None, np.arange(mid_z-z_rad, mid_z+z_rad)]))
    preseq = affine_transform(preseq, transform_pre)
    seq = affine_transform(seq, transform_seq)
    output = brightness_scale(preseq, seq)
    if queue is not None:
        queue.put(output)

    return output
