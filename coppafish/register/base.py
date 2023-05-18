import os
import pickle
import numpy as np
from sklearn.linear_model import HuberRegressor
from scipy.spatial import KDTree
from scipy import stats
from skimage.filters import sobel
from skimage.exposure import match_histograms
from scipy.ndimage import affine_transform
from ..utils.npy import load_tile
from coppafish.register.preprocessing import custom_shift, yxz_to_zyx, save_compressed_image, split_3d_image, \
    replace_scale, populate_full, merge_subvols, invert_affine
from skimage.registration import phase_cross_correlation
from coppafish.setup import NotebookPage


def find_shift_array(subvol_base, subvol_target, position, r_threshold):
    """
    This function takes in 2 split up 3d images and finds the optimal shift from each subvolume in 1 to it's corresponding
    subvolume in the other.
    Args:
        subvol_base: Base subvolume array
        subvol_target: Target subvolume array
        position: Position of centre of subvolumes (z, y, x)
        r_threshold: threshold of correlation used in degenerate cases
    Returns:
        shift: 2D array, with first dimension referring to subvolume index and final dim referring to shift.
        shift: 2D array, with first dimension referring to subvolume index and final dim referring to shift_corr coef.
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
        subvol_base: Base subvolume array (this is a z-tower of base subvolumes)
        subvol_target: Target subvolume array (this is a z-tower of target subvolumes)
        position: Position of centre of subvolumes (z, y, x)
        pearson_r_threshold: threshold of correlation used in degenerate cases
        z_neighbours: number of neighbouring subvolumes to merge with the current subvolume to compute the shift
    Returns:
        shift: 2D array, with first dimension referring to subvolume index and final dim referring to shift.
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


def find_zyx_shift(subvol_base, subvol_target, pearson_r_threshold=0.4):
    """
    This function takes in 2 3d images and finds the optimal shift from one to the other. We use a phase cross
    correlation method to find the shift.
    Args:
        subvol_base: Base subvolume array (this will contain a lot of zeroes)
        subvol_target: Target subvolume array (this will be a merging of subvolumes with neighbouring subvolumes)
        pearson_r_threshold: Threshold used to accept a shift as valid

    Returns:
        shift: zyx shift
        shift_corr: correlation coefficient of shift
    """
    if subvol_base.shape != subvol_target.shape:
        raise ValueError("Subvolume arrays have different shapes")
    shift, _, _ = phase_cross_correlation(reference_image=subvol_target, moving_image=subvol_base,
                                                   upsample_factor=10)
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

    # Now return the shift with the highest correlation coefficient
    if alt_shift_corr > shift_corr:
        shift = alt_shift
        shift_corr = alt_shift_corr
    # Now check if the correlation coefficient is above the threshold. If not, set the shift to nan
    if shift_corr < pearson_r_threshold:
        shift = np.array([np.nan, np.nan, np.nan])
        shift_corr = max(shift_corr, alt_shift_corr)

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


def huber_regression(shift, position):
    """
    Function to predict shift as a function of position using robust huber regressor.
    Args:
        shift: n_tiles x 3 ndarray of zyx shifts
        position: n_tiles x 2 ndarray of yx tile coords
    Returns:
        transform: 3 x 3 matrix where each row predicts shift of z y z as a function of y index, x index and the final
        row is the offset at 0,0
    """
    # Do robust regression
    huber_z = HuberRegressor().fit(X=position, y=shift[:, 0])
    huber_y = HuberRegressor().fit(X=position, y=shift[:, 1])
    huber_x = HuberRegressor().fit(X=position, y=shift[:, 2])
    transform = np.vstack((np.append(huber_z.coef_, huber_z.intercept_), np.append(huber_y.coef_, huber_y.intercept_),
                           np.append(huber_x.coef_, huber_x.intercept_)))

    return transform


# Bridge function for all functions in round registration
def round_registration(nbp_file: NotebookPage, nbp_basic: NotebookPage, config: dict, registration_data: dict,
                           t: int, pbar) -> dict:
    """
    Function to carry out subvolume registration on a single tile.
    Args:
        nbp_file: File Names notebook page
        nbp_basic: Basic info notebook page
        config: Register page of the config dictionary
        registration_data: dictionary with registration data
        t: tile
        pbar: Progress bar (makes this part of code difficult to run independently of main pipeline)

    Returns:
        registration_data updated after the subvolume registration
    """
    z_subvols, y_subvols, x_subvols = config['z_subvols'], config['y_subvols'], config['x_subvols']
    z_box, y_box, x_box = config['z_box'], config['y_box'], config['x_box']
    r_thresh = config['r_thresh']

    # Load in the anchor npy volume
    anchor_image = sobel(yxz_to_zyx(load_tile(nbp_file, nbp_basic, t, nbp_basic.anchor_round,
                                                   nbp_basic.anchor_channel)))

    save_compressed_image(nbp_file, anchor_image, t, nbp_basic.anchor_round, nbp_basic.anchor_channel)

    # Now compute round shifts for this tile and the affine transform for each round
    for r in nbp_basic.use_rounds:
        # Set progress bar title
        pbar.set_description('Computing shifts for tile ' + str(t) + ', round ' + str(r))

        # Load in imaging npy volume.
        target_image = yxz_to_zyx(sobel(load_tile(nbp_file, nbp_basic, t, r, nbp_basic.anchor_channel)))

        # save a small subset for reg diagnostics
        save_compressed_image(nbp_file, target_image, t, r, nbp_basic.anchor_channel)

        # next we split image into overlapping cuboids
        subvol_base, position = split_3d_image(image=anchor_image, z_subvolumes=z_subvols,
                                               y_subvolumes=y_subvols, x_subvolumes=x_subvols,
                                               z_box=z_box, y_box=y_box, x_box=x_box)
        subvol_target, _ = split_3d_image(image=target_image, z_subvolumes=z_subvols, y_subvolumes=y_subvols,
                                          x_subvolumes=x_subvols, z_box=z_box, y_box=y_box, x_box=x_box)

        # Find the subvolume shifts

        shift, corr = find_shift_array(subvol_base, subvol_target, position=position.copy(), r_threshold=r_thresh)

        # Append these arrays to the round_shift, round_shift_corr, round_transform and position storage
        registration_data['round_registration']['position'] = position
        registration_data['round_registration']['round_shift'][t, r] = shift
        registration_data['round_registration']['round_shift_corr'][t, r] = corr
        registration_data['round_registration']['round_transform_raw'][t, r] = ols_regression(shift, position)
        pbar.update(1)

    # Add tile to completed tiles
    registration_data['round_registration']['tiles_completed'].append(t)

    # Now save the registration data dictionary externally
    with open(os.path.join(nbp_file.output_dir, 'registration_data.pkl'), 'wb') as f:
        pickle.dump(registration_data, f)

    return registration_data


def channel_registration(nbp_file: NotebookPage, nbp_basic: NotebookPage, config: dict, registration_data: dict,
                           t: int, pbar) -> dict:
    """
    Function to carry out subvolume registration on a single tile.
    Args:
        nbp_file: File Names notebook page
        nbp_basic: Basic info notebook page
        config: Register page of the config dictionary
        registration_data: dictionary with registration data
        t: tile
        pbar: Progress bar (makes this part of code difficult to run independently of main pipeline)

    Returns:
        registration_data updated after the subvolume registration
    """
    # Now begin the channel registration.
    #  * We will not compute subvolume shifts for this step
    #  * Instead we will use scales from previous experiments and we will find shifts for each channel via a single
    #    3D cross correlation

    # We will use the round with the highest round_shift_corr. We will move the anchor into this round's space
    # by applying the round transform to the anchor image.
    # best_round = np.argmax(np.sum(registration_data['round_registration']['round_shift_corr'][t], axis=1))
    # Now load the anchor image, convert to zyx and apply sobel filter.
    # Take small subset of this image for speed
    # anchor_image = sobel(yxz_to_zyx(load_tile(nbp_file, nbp_basic, t, nbp_basic.anchor_round, nbp_basic.anchor_channel)))
    # z_mid, y_mid, x_mid = np.array(anchor_image.shape) // 2
    # adjusted_anchor = \
    #     affine_transform(anchor_image,
    #                      invert_affine(registration_data['round_registration']['round_transform_raw'][t, best_round]))
    # adjusted_anchor = adjusted_anchor[z_mid - 16:z_mid + 16, y_mid - 250:y_mid + 250, x_mid - 250:x_mid + 250]
    # Now we will loop through channels of this round. We will initially apply the prior channel transform to each
    # channel and then find the shift between the adjusted anchor and adjusted channel.
    prior_channel_transform = np.load(os.path.join(os.getcwd(), 'coppafish/setup/', 'prior_channel_transform.npy'))

    for c in nbp_basic.use_channels:
        # Set progress bar title
        pbar.set_description('Computing shifts for tile ' + str(t) + ', channel ' + str(c))

        # Load in imaging npy volume.
        # target_image = yxz_to_zyx(sobel(load_tile(nbp_file, nbp_basic, t, best_round, c)))
        # target_image = target_image[z_mid - 16:z_mid + 16, y_mid - 250:y_mid + 250, x_mid - 250:x_mid + 250]

        # save a small subset for reg diagnostics
        # save_compressed_image(nbp_file, target_image, t, best_round, c)

        # Apply the prior channel transform to the target image. This will undo the chromatic aberration, so that
        # the channel shift is only due to the registration
        # target_image = affine_transform(target_image, prior_channel_transform[c])

        # Find the shift between the adjusted anchor and the adjusted target
        # shift, _, _ = phase_cross_correlation(reference_image=target_image, moving_image=adjusted_anchor,
        #                                       upsample_factor=10)

        # Now use our custom_shift function to manually shift the adjusted anchor and compute the correlation
        # adjusted_anchor_shifted = custom_shift(adjusted_anchor, shift.astype(int))
        # mask = adjusted_anchor_shifted != 0
        # corr = np.corrcoef(adjusted_anchor_shifted[mask], target_image[mask])[0, 1]
        # if corr < config['r_threshold']:
        #     shift = np.array([0, 0, 0])
        shift = np.array([0, 0, 0])
        # Append these arrays to the channel_shift, channel_shift_corr, channel_transform and position storage
        registration_data['channel_registration']['reference_round'][t] = 10
        registration_data['channel_registration']['channel_shift'][t, c] = shift
        registration_data['channel_registration']['channel_shift_corr'][t, c] = 0
        registration_data['channel_registration']['channel_transform_raw'][t, c] = \
            np.vstack((prior_channel_transform[c].T, shift)).T
        pbar.update(1)

    # Add tile to completed tiles
    registration_data['channel_registration']['tiles_completed'].append(t)

    # Now save the registration data dictionary externally
    with open(os.path.join(nbp_file.output_dir, 'registration_data.pkl'), 'wb') as f:
        pickle.dump(registration_data, f)

    return registration_data


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


# Function to remove outlier channel scales
def regularise_channel_scaling(scale: np.ndarray):
    """
    Function to remove outliers in the scale parameters for channel transforms. Experience shows these should be close
    to 1 for z regardless of channel and should be the same for the same channel.

    Args:
        scale: 3 x n_tiles_use x n_channels_use ndarray of z y x scales for each tile and channel

    Returns:
        scale_regularised:  n_tiles_use x n_channels_use x 3 ndarray of z y x regularised scales for each tile and chan
    """
    # Define num tiles and separate the z scale and yx scales for different analysis
    n_tiles = scale.shape[1]
    z_scale = scale[0]
    yx_scale = scale[1:]

    # Regularise z_scale: No need to account for variation in tile or channel
    outlier_z = np.abs(z_scale - 1) > 0.01
    z_scale[outlier_z] = 1

    # Regularise yx_scale: Account for variation in channel
    median_yx_scale = np.repeat(np.median(yx_scale, axis=1)[:, np.newaxis], n_tiles, axis=1)
    iqr_yx_scale = np.repeat(stats.iqr(yx_scale, axis=1)[:, np.newaxis], n_tiles, axis=1)
    outlier_yx = np.abs(yx_scale - median_yx_scale) > iqr_yx_scale
    yx_scale[outlier_yx] = median_yx_scale[outlier_yx]

    return scale


# Bridge function for all regularisation
def regularise_transforms(registration_data: dict, tile_origin: np.ndarray, residual_threshold: float,
                          use_tiles: list, use_rounds: list, use_channels: list):
    """
    Function to regularise affine transforms
    Args:
        registration_data: dictionary of registration data
        tile_origin: n_tiles x 3 tile origin in zyx
        residual_threshold: threshold for classifying outlier shifts
        use_tiles: list of tiles in use
        use_rounds: list of rounds in use
        use_channels: list of channels in use

    Returns:
        round_transform_regularised: n_tiles x n_rounds x 3 x 4 affine transforms regularised
        channel_transform_regularised: n_tiles x n_channels x 3 x 4 affine transforms regularised
    """
    # Extract transforms
    round_transform = np.copy(registration_data['round_registration']['round_transform_raw'])
    channel_transform = np.copy(registration_data['channel_registration']['channel_transform_raw'])

    # Code becomes easier when we disregard tiles, rounds, channels not in use
    n_tiles, n_rounds, n_channels = round_transform.shape[0], round_transform.shape[1], channel_transform.shape[1]
    tile_origin = tile_origin[use_tiles]
    round_transform = round_transform[use_tiles][:, use_rounds]
    channel_transform = channel_transform[use_tiles][:, use_channels]

    # Regularise round transforms
    round_transform[:, :, :, 3] = regularise_shift(shift=round_transform[:, :, :, 3], tile_origin=tile_origin,
                                                   residual_threshold=residual_threshold)
    round_scale = np.array([round_transform[:, :, 0, 0], round_transform[:, :, 1, 1], round_transform[:, :, 2, 2]])
    round_scale_regularised = regularise_round_scaling(round_scale)
    round_transform = replace_scale(transform=round_transform, scale=round_scale_regularised)
    round_transform = populate_full(sublist_1=use_tiles, list_1=np.arange(n_tiles),
                                    sublist_2=use_rounds, list_2=np.arange(n_rounds),
                                    array=round_transform)

    # Regularise channel transforms
    channel_transform[:, :, :, 3] = regularise_shift(shift=channel_transform[:, :, :, 3], tile_origin=tile_origin,
                                                     residual_threshold=residual_threshold)
    channel_scale = np.array([channel_transform[:, :, 0, 0], channel_transform[:, :, 1, 1],
                              channel_transform[:, :, 2, 2]])
    channel_transform = replace_scale(transform=channel_transform, scale=channel_scale)
    channel_transform = populate_full(sublist_1=use_tiles, list_1=np.arange(n_tiles),
                                      sublist_2=use_channels, list_2=np.arange(n_channels),
                                      array=channel_transform)

    registration_data['round_registration']['round_transform'] = round_transform
    registration_data['channel_registration']['channel_transform'] = channel_transform

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
def icp(yxz_base, yxz_target, dist_thresh, start_transform, n_iters, robust):
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
