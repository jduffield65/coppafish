import os
import pickle
import numpy as np
from scipy.spatial import KDTree
from scipy import stats
from skimage.filters import sobel
from skimage.exposure import match_histograms
from scipy.ndimage import affine_transform
from ..utils.npy import load_tile
from coppafish.register.preprocessing import custom_shift, yxz_to_zyx, save_compressed_image, split_3d_image
from skimage.registration import phase_cross_correlation
from coppafish.setup import NotebookPage


def find_shift_array(subvol_base, subvol_target, r_threshold):
    """
    This function takes in 2 split up 3d images and finds the optimal shift from each subvolume in 1 to it's corresponding
    subvolume in the other.
    Args:
        subvol_base: Base subvolume array
        subvol_target: Target subvolume array
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

    for y in range(y_subvolumes):
        for x in range(x_subvolumes):
            shift[:, y, x], shift_corr[:, y, x] = find_z_tower_shifts(subvol_base=subvol_base[:, y, x],
                                                                      subvol_target=subvol_target[:, y, x],
                                                                      r_threshold=r_threshold)
    return np.reshape(shift, (shift.shape[0] * shift.shape[1] * shift.shape[2], 3)), \
        np.reshape(shift_corr, shift.shape[0] * shift.shape[1] * shift.shape[2])


def find_z_tower_shifts(subvol_base, subvol_target, r_threshold):
    """
    This function takes in 2 split up 3d images along one tower of subvolumes in z and computes shifts for each of these
    to its nearest neighbour subvol
    Args:
        subvol_base: Base subvolume array (this is a z-tower of base subvolumes)
        subvol_target: Target subvolume array (this is a z-tower of target subvolumes)
        r_threshold: threshold of correlation used in degenerate cases
    Returns:
        shift: 4D array, with first 3 dimensions referring to subvolume index and final dim referring to shift.
    """
    z_subvolumes = subvol_base.shape[0]
    z_box = subvol_base.shape[1]
    shift = np.zeros((z_subvolumes, 3))
    shift_corr = np.zeros(z_subvolumes)
    # skip tells us when we need to start comparing z subvols to the subvol above it (in the case that it is
    # closer to this, or below it in the other case). Reset this whenever we start a new z-tower
    skip = 0
    for z in range(z_subvolumes):
        # Handle case where skip allows us to leave the z-range
        if z + skip >= z_subvolumes:
            shift[z] = np.nan
            break
        # Now proceed as normal: Compute shift and then disambiguate shift.
        # If disambiguation changes the correspondence of the z_subvolumes then subvol_skip will be updated to +/- 1
        temporary_shift, _, _ = phase_cross_correlation(subvol_target[z + skip], subvol_base[z], upsample_factor=10)
        shift[z], shift_corr[z], subvol_skip = disambiguate_z(subvol_base[z],
                                                              subvol_target[z + skip],
                                                              temporary_shift,
                                                              r_threshold=r_threshold, z_box=z_box)
        skip += subvol_skip

    return shift, shift_corr


# Complicated but necessary function to get rid of aliasing issues in Fourier Shifts
def disambiguate_z(base_image, target_image, shift, r_threshold, z_box):
    """
    Function to disambiguate the 2 possible shifts obtained via aliasing.
    Args:
        base_image: nz x ny x nx ndarray
        target_image: nz x ny x nx ndarray
        shift: z y x shift
        r_threshold: threshold that r_statistic must exceed for us to accept z shift of 0 as opposed to alisases
        z_box: z_box size

    Returns:
        shift: z y x corrected shift
        shift_corr: shift correlation of the best shift
        subvol_skip: number of z subvols we have skipped (-1, 0 or +1)
    """
    # First we have to compute alternate shift. Usually anti aliasing algorithms would test shift and shift + z_box and
    # shift - z_box. In our case however, we only look at one other shift as otherwise means considering a shift that
    # is more than 1 box wrong, which is unlikely due to our skip policy
    if shift[0] >= 0:
        alt_shift = shift - [z_box, 0, 0]
    else:
        alt_shift = shift + [z_box, 0, 0]

    # Now we need to actually shift the base image under both shift and alt_shift
    shift_base = custom_shift(base_image, np.round(shift).astype(int))
    alt_shift_base = custom_shift(base_image, np.round(alt_shift).astype(int))

    # Now, if either of the shift_base or alt_shift base is all 0 then it's correlation coeff is undefined
    if np.max(abs(shift_base)) == 0:
        shift_corr = 0
        # We only want the corr coeff where the alt shifted anchor image exists
        valid = alt_shift_base != 0
        alt_shift_corr = stats.pearsonr(alt_shift_base[valid], target_image[valid])[0]
    elif np.max(abs(alt_shift_base)) == 0:
        alt_shift_corr = 0
        # only want the corr coeff where the shifted anchor image exists
        valid = shift_base != 0
        shift_corr = stats.pearsonr(shift_base[valid], target_image[valid])[0]
    else:
        # only want the corr coeff where the shifted anchor images exist
        valid = shift_base != 0
        shift_corr = stats.pearsonr(shift_base[valid], target_image[valid])[0]
        valid = alt_shift_base != 0
        alt_shift_corr = stats.pearsonr(alt_shift_base[valid], target_image[valid])[0]

    # Now comes the point where we choose between shift and alt_shift. Choose the shift which maximises corr if this is
    # above r_thresh and return nan otherwise
    best_shift = [shift, alt_shift][np.argmax([shift_corr, alt_shift_corr])]
    best_shift_corr = max(shift_corr, alt_shift_corr)
    subvol_skip = np.sign(best_shift[0] - shift[0])
    if best_shift_corr > r_threshold:
        corrected_shift = best_shift
    else:
        corrected_shift = np.array([np.nan, np.nan, np.nan])

    return corrected_shift, max(shift_corr, alt_shift_corr), int(subvol_skip)


# ols regression to find transform from shifts
def ols_regression_robust(shift, position):
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


# Function which runs a single iteration of the icp algorithm, in use
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


# Simple ICP implementation, calls above until
def icp(yxz_base, yxz_target, dist_thresh, start_transform, n_iters, robust):
    """
    Applies n_iters rrounds of the above least squares regression
Args:
        yxz_base: ```float [n_base_spots x 3]```.
            Coordinates of spots you want to transform.
        yxz_target: ```float [n_target_spots x 3]```.
            Coordinates of spots in image that you want to transform ```yxz_base``` to.
        start_transform: initial transform
        dist_thresh: If neighbours closer than this, they are used to compute the new transform.
            Typical: ```3```.
        n_iters: number of times to compute regression


    Returns:
        - ```transform``` - ```float [4 x 3]```.
            Updated affine transform.
        - ```n_matches``` - ```int```.
            Number of neighbours which have distance less than ```dist_thresh```.
        - ```error``` - ```float```.
            Average distance between ```neighbours``` below ```dist_thresh```.
    """

    # initialise transform
    transform = start_transform
    n_matches = np.zeros(n_iters)
    error = np.zeros(n_iters)

    for i in range(n_iters):
        transform, _, n_matches[i], error[i] = get_transform(yxz_base, yxz_target, transform,
                                                                     dist_thresh, robust)
        if i > 0 and n_matches[i] == n_matches[i-1]:
            n_matches[i:] = n_matches[i] * np.ones(n_iters - i)
            break

    return transform, n_matches, error


# Simple function to get rid of outlier transforms by comparing with other tiles
def regularise_transforms(transform, residual_threshold, tile_origin, use_tiles, rc_use):
    """
    Function to regularise outliers in either round_transform or channel_transform.
    Without loss of generality, we explain how this works for round_transforms, bearing in mind the same logic applies
    to channel transforms.
    For each fixed round r, we have a collection of transforms indexed by the tiles in use. We look at the shifts
    associated with each of these tiles and see if that looks reasonable based on the position of the tiles.
    More formally, this means that for each fixed r, we compute a linear regression to determine expected shifts.
    If a shift belonging to tile t falls outside residual_threshold of the prediction, the shift for tile t round r
    is replaced with the predicted shift.
    We don't do regression on scale parameters but rather just analyse them for outliers.
    Args:
        transform: Either [n_tiles x n_rounds x 4 x 3] or [n_tiles x n_channels x 4 x 3]
        residual_threshold: This is a threshold above which we will consider a point to be an outlier
        tile_origin: yxz positions of the tiles [n_tiles x 3]
        use_tiles: list of tiles in use

    Returns:
        transform_regularised: Either [n_tiles x n_rounds x 4 x 3] or [n_tiles x x_channels x 4 x 3]
    """
    # rearrange columns so that tile origins are in zyx like the shifts are, then initialise commonly used variables
    tile_origin = np.roll(tile_origin, 1, axis=1)[use_tiles]
    n_tiles_use = len(use_tiles)
    shift = transform[use_tiles, :, :, 3]
    predicted_shift = np.zeros_like(shift)

    # This gives us a different set of shifts for each round/channel, let's compute the expected shift for each of these
    # rounds/channels independently. Since this could be either a round or channel, we call it an elem
    for elem in rc_use:
        big_transform = ols_regression_robust(shift[:, elem], tile_origin)
        padded_origin = np.vstack((tile_origin.T, np.ones(n_tiles_use)))
        predicted_shift[:, elem] = (big_transform @ padded_origin).T - tile_origin

    # Use these predicted shifts to get rid of outliers
    residual = np.linalg.norm(predicted_shift-shift, axis=2)
    shift_outlier = residual > residual_threshold
    # The nice thing about this approach is that we can immediately replace the outliers with their prediction
    # Now we do something similar for the scales, though we don't really assume these will vary so we just take medians
    scale = np.swapaxes(np.array([transform[use_tiles, :, 0, 0].T, transform[use_tiles, :, 1, 1].T,
                                  transform[use_tiles, :, 2, 2].T]), 0, 2)
    scale_median = np.median(scale, axis=0)
    scale_iqr = stats.iqr(scale, axis=0)
    scale_low = scale < scale_median - 1.5 * scale_iqr
    scale_high = scale > scale_median + 1.5 * scale_iqr
    scale_outlier = (scale_low + scale_high).any(axis=-1)

    # Now we have everything to generate the regularised transforms
    transform_regularised = transform.copy()
    for t in range(n_tiles_use):
        for elem in rc_use:
            if shift_outlier[t, elem] or scale_outlier[t, elem]:
                transform_regularised[t, elem] = np.vstack((np.diag(scale_median[elem]), predicted_shift[t, elem])).T

    return transform_regularised


def subvolume_registration(nbp_file: NotebookPage, nbp_basic: NotebookPage, config: dict, registration_data: dict,
                           t: int, pbar) -> dict:
    """
    Function to carry out subvolume registration on a single tile.
    Args:
        nbp_file: File Names notebook page
        nbp_basic: Basic info notebook page
        config: Register page of the config dictionary
        registration_data: dictionary with the following keys
            * tiles_completed (list)
            * position ( (z_subvols x y subvols x x_subvols) x 3 ) ndarray
            * round_shift ( n_tiles x n_rounds x (z_subvols x y subvols x x_subvols) x 3 ) ndarray
            * channel_shift ( n_tiles x n_channels x (z_subvols x y subvols x x_subvols) x 3 ) ndarray
            * round_transform (n_tiles x n_rounds x 3 x 4) ndarray
            * channel_transform (n_tiles x n_channels x 3 x 4) ndarray
            * round_shift_corr ( n_tiles x n_rounds x (z_subvols x y subvols x x_subvols) ) ndarray
            * channel_shift_corr ( n_tiles x n_channels x (z_subvols x y subvols x x_subvols) ) ndarray
        t: tile
        pbar: Progress bar

    Returns:
        registration_data updated after the subvolume registration
    """
    z_subvols, y_subvols, x_subvols = config['z_subvols'], config['y_subvols'], config['x_subvols']
    z_box, y_box, x_box = config['z_box'], config['y_box'], config['x_box']
    r_thresh = config['r_threshold']

    # Load in the anchor npy volume
    # Save the unfiltered version as well for histogram matching later, switch both to zyx
    anchor_image_unfiltered = yxz_to_zyx(load_tile(nbp_file, nbp_basic, t, nbp_basic.anchor_round,
                                                   nbp_basic.anchor_channel))
    anchor_image = sobel(anchor_image_unfiltered)

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
        shift, corr = find_shift_array(subvol_base, subvol_target, r_threshold=r_thresh)

        # Append these arrays to the round_shift, round_shift_corr, round_transform and position storage
        registration_data['position'] = position
        registration_data['round_shift'][t, r] = shift
        registration_data['round_shift_corr'][t, r] = corr
        registration_data['round_transform'][t, r] = ols_regression_robust(shift, position)

    # Now begin the channel registration
    correction_matrix = np.vstack((registration_data['round_transform'][t, nbp_basic.n_rounds // 2], [0, 0, 0, 1]))
    # scipy's affine transform function requires affine transform be inverted
    anchor_image_corrected = affine_transform(anchor_image, np.linalg.inv(correction_matrix))

    # Now register all channels to corrected anchor
    for c in nbp_basic.use_channels:
        # Update progress bar
        pbar.set_description('Computing shifts for tile ' + str(t) + ', channel ' + str(c))

        # Load in imaging npy volume from middle round for tile t channel c
        target_image = yxz_to_zyx(load_tile(nbp_file, nbp_basic, t, nbp_basic.n_rounds // 2, c))

        # Match histograms to unfiltered anchor and then sobel filter
        target_image = sobel(match_histograms(target_image, anchor_image_unfiltered))

        # next we split image into overlapping cuboids
        subvol_base, _ = split_3d_image(image=anchor_image_corrected, z_subvolumes=z_subvols,
                                        y_subvolumes=y_subvols, x_subvolumes=x_subvols,
                                        z_box=z_box, y_box=y_box, x_box=x_box)
        subvol_target, _ = split_3d_image(image=target_image, z_subvolumes=z_subvols,
                                          y_subvolumes=y_subvols, x_subvolumes=x_subvols,
                                          z_box=z_box, y_box=y_box, x_box=x_box)

        # Find the subvolume shifts.
        # NB: These are shifts from corrected anchor (in coord frame of r_mid, anchor_channel) to r_mid, c
        if c == nbp_basic.anchor_channel:
            shift, corr = np.zeros((z_subvols * y_subvols * x_subvols, 3)), \
                np.ones(z_subvols * y_subvols * x_subvols)
        else:
            shift, corr = find_shift_array(subvol_base, subvol_target, r_threshold=r_thresh)

        # Save data into our working dictionary
        registration_data['channel_shift'][t, c] = shift
        registration_data['channel_shift_corr'][t, c] = corr
        registration_data['channel_transform'][t, c] = ols_regression_robust(shift, position)

    # Add tile to completed tiles
    registration_data['tiles_completed'].append(t)

    # Now save the registration data dictionary externally
    with open(os.path.join(nbp_file.output_dir, 'registration_data.pkl'), 'wb') as f:
        pickle.dump(registration_data, f)

    return registration_data

# def anti_alias(shift, box):
#     n_shifts = shift.shape[0]
#     num_neighb = np.zeros((n_shifts, 3), dtype=int)
#     # Compute number of neighbours for each of these options
#     for i in range(n_shifts):
#         num_neighb[i, 0] = len([s for s in shift if s < shift[i] - box/2])
#         num_neighb[i, 1] = len([s for s in shift if abs(s - shift[i]) < box/2])
#         num_neighb[i, 2] = len([s for s in shift if s > shift[i] + box/2])
#
#     # Now correct for aliased shifts, this will be incorrect for nan shifts but this doesn't affect anything as these
#     # nan + wrong = nan
#     shift_correction = box * (np.argmax(num_neighb, axis=1) - 1)
#     shift += shift_correction
#     return shift
#
#
# def clean_shift_data(shift, box_size, zero_bias=False):
#     """
#     Function to clean the shift data we obtain before doing regression on it.
#     Args:
#         shift: z_sv x y_sv x x_sv x 3 ndarray of zyx shifts
#         position: z_sv x y_sv x x_sv x 3 ndarray of zyx positions
#         box_size: zyx box dims
#         zero_bias: whether to down regulate zeroes (typically yes for round transforms, no o/w)
#
#     Returns:
#         shift_clean: z_sv x y_sv x x_sv x 3 ndarray of zyx cleaned shifts
#     """
#     # initialise data into nice form
#     z_sv, y_sv, x_sv = shift.shape[0], shift.shape[1], shift.shape[2]
#     shift = np.reshape(shift, (shift.shape[0] * shift.shape[1] * shift.shape[1], 3))
#     # At this stage we 'flatten' the shift matrix a bit
#     shift_clean = np.copy(shift)
#     z_box = box_size[0]
#
#     # correct for the zero_bias
#     if zero_bias:
#         box_shift = abs(shift[:, 0] - z_box) <= 0.1
#         neg_box_shift = abs(shift[:, 0] + z_box) <= 0.1
#         alias_zero = neg_box_shift + box_shift
#         shift_clean[alias_zero] = np.nan
#
#     # First we will do the antialiasing step. This is only a concern for z so don't worry about xy
#     # This works by moving the shift 1 up and 1 down. Then looking at number of neighbours in a neighbourhood
#     # of that z. We then choose the option (-1, 0, 1) corresponding to most neighbours.
#     # NB: We only compare shifts from the same layer of the z_subvolumes
#     chunk = y_sv * x_sv
#     for z in range(z_sv):
#         shift_clean[z * chunk:(z + 1) * chunk, 0] = anti_alias(shift_clean[z * chunk:(z + 1) * chunk, 0], z_box)
#
#     return shift_clean
