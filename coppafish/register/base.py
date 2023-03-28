import numpy as np
from scipy.spatial import KDTree
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from .preprocessing import custom_shift
from skimage.registration import phase_cross_correlation


def find_shift_array(subvol_base, subvol_target, r_threshold):
    """
    This function takes in 2 split up 3d images and finds the optimal shift from each subvolume in 1 to it's corresponding
    subvolume in the other.
    Args:
        subvol_base: Base subvolume array
        subvol_target: Target subvolume array
        r_threshold: threshold of correlation used in degenerate cases
    Returns:
        shift: 4D array, with first 3 dimensions referring to subvolume index and final dim referring to shift.
    """
    if subvol_base.shape != subvol_target.shape:
        raise ValueError("Subvolume arrays have different shapes")
    z_subvolumes, y_subvolumes, x_subvolumes = subvol_base.shape[0], subvol_base.shape[1], subvol_base.shape[2]
    z_box = subvol_base.shape[3]
    shift = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, 3))

    for y in range(y_subvolumes):
        for x in range(x_subvolumes):
            # skip tells us when we need to start comparing z subvols to the subvol above it (in the case that it is
            # closer to this, or below it in the other case). Reset this whenever we start a new z-tower
            skip = 0
            for z in range(z_subvolumes):
                # Handle case where skip allows us to leave the z-range
                if z + skip >= z_subvolumes:
                    shift[z, y, x] = np.nan
                    break
                # Now proceed as normal
                shift[z, y, x], _, _ = phase_cross_correlation(subvol_target[z + skip, y, x], subvol_base[z, y, x],
                                                               upsample_factor=10)
                corrected_shift, shift_corr, alt_shift_corr = disambiguate_z(subvol_base[z, y, x],
                                                                             subvol_target[z + skip, y, x],
                                                                             shift[z, y, x], r_threshold=0.2)

                # CASE 1: deal with the very degenerate case where corrected_shift = nan, this means that a shift of 0
                # (as predicted by PCC) fails to be similar to the target image. In this case, we can look at neighbours
                # and see if they do better. If they do then make the shift to the appropriate neighbour
                if np.isnan(corrected_shift[0]):
                    neighbours = {z-1, z+1}.intersection(set(np.arange(z_subvolumes, dtype=int)))
                    # Now loop over neighbours, find their shifts and
                    for neighb in neighbours:
                        candidate_shift, _, _ = phase_cross_correlation(subvol_target[neighb, y, x],
                                                                        subvol_base[z, y, x], upsample_factor=10)
                        shift_base = custom_shift(subvol_base[z, y, x], candidate_shift.astype(int))
                        shift_mask = shift_base > 0
                        candidate_corr = stats.pearsonr(shift_base[shift_mask], subvol_base[z, y, x, shift_mask])[0]
                        if candidate_corr > r_threshold:
                            corrected_shift = z_box * (neighb - z) + candidate_shift
                            skip = skip + neighb - z
                            break
                    # If our corrected_shift has been updated by the shifts to one of the neighbours then next line will
                    # replace the shift, else next line will just save this shift as nan
                    shift[z, y, x] = corrected_shift
                else:
                    # If the corrected shift is not the same as the original shift it is due to aliasing. The range of
                    # shifts given is [-z_box/2, z_box/2] so if the aliased shift is the true shift then
                    # subvol_base[z,y,x] has more in common with either subvol_target[z-1,y,x] or
                    # subvol_target[z+1, y, x], so update the skip parameter
                    if corrected_shift[0] != shift[z, y, x, 0] and alt_shift_corr > shift_corr:
                        if corrected_shift[0] > shift[z, y, x, 0]:
                            skip += 1
                        else:
                            skip -= 1
                        # Finally, update the shift to the correct one
                        shift[z, y, x] = corrected_shift
    return shift


# Custom regression implementation of Theil-Sen. No longer in use.
def find_affine_transform_robust_custom(shift, position, num_pairs, boundary_erosion, image_dims, dist_thresh,
                                        resolution, view=False):
    """
    Uses a custom Theil-Sen variant to find optimal affine transform matching the shifts to their positions.
    Args:
        shift: z_sv x y_sv x x_sv x 3 array which of shifts in zyx format
        position: z_sv x y_sv x x_sv x 3 array which of positions in zyx format
        num_pairs: Number of pairs to consider. Must be in range [0, num_samples choose 2]
        boundary_erosion: 3 x 1 array of z, y, x boundary erosion terms
        image_dims: 3 x 1 array of z, y, x image dims
        dist_thresh 3 x 1 array of distance thresholds of z, y and x points that pairs of points must be apart if
        we use them in the algorithm
        Resolution: number of bins in histogram whose mode we return (range = median +/- iqr)
        view: option to view the regression
    Returns:
        transform: 3 x 4 affine transform in yxz format with final col being shift
    """

    # Initialise returned variable
    transform = np.zeros((3, 4))

    # Ensure that num_pairs is an integer and all inputted arrays are ndarrays
    num_pairs = int(num_pairs)
    boundary_erosion = np.array(boundary_erosion)
    image_dims = np.array(image_dims)
    dist_thresh = np.array(dist_thresh)
    z_subvols, y_subvols, x_subvols = shift.shape[:3]
    scale = np.zeros((num_pairs, 3))
    intercept = np.zeros((num_pairs, 3))

    # Now start looping through pairs of shifts randomly, subject to following constraints:
    # Exclude [0, 0, 0] shifts as these are only returned when the program fails to find a shift
    # Exclude shifts too close together, as these have too low a resolution to detect a difference
    # Exlude shifts too close to boundary
    i = 0
    with tqdm(total=num_pairs) as pbar:
        pbar.set_description(f"Robust Estimation of scale and shift parameters")
        while i < num_pairs:
            # Generate random indices
            z1, y1, x1 = np.random.randint(0, z_subvols), np.random.randint(1, y_subvols-1), np.random.randint(1, x_subvols-1)
            z2, y2, x2 = np.random.randint(0, z_subvols), np.random.randint(1, y_subvols-1), np.random.randint(1, x_subvols-1)
            # Use this to generate scales and intercepts
            # in_range = (boundary_erosion < position[z1, y1, x1]) * (position[z1, y1, x1] < image_dims - boundary_erosion) * \
            #            (boundary_erosion < position[z2, y2, x2]) * (position[z2, y2, x2] < image_dims - boundary_erosion)
            both_shifts_nonzero = np.min(shift[z1, y1, x1] * shift[z2, y2, x2]) > 0
            points_sufficiently_far = abs(position[z1, y1, x1] - position[z2, y2, x2]) > dist_thresh
            if both_shifts_nonzero and all(points_sufficiently_far):
                scale[i] = np.ones(3) + (shift[z2, y2, x2] - shift[z1, y1, x1]) / (position[z2, y2, x2] - position[z1, y1, x1])
                intercept[i] = shift[z1, y1, x1] - (scale[i] - 1) * position[z1, y1, x1]
                i += 1

                pbar.update(1)
    # Now that we have these randomly sampled scales and intercepts, let's robustly estimate their parameters
    scale_median, scale_iqr = np.median(scale, axis=0), stats.iqr(scale, axis=0)
    intercept_median, intercept_iqr = np.median(intercept, axis=0), stats.iqr(intercept, axis=0)

    # Now create histograms within 1 IQR of each median and take the top value as our estimate
    scale_bin_z_val, scale_bin_z_index, _ = plt.hist(scale[:, 0], bins=resolution,
                                                     range=(
                                                     scale_median[0] - scale_iqr[0]/2, scale_median[0] + scale_iqr[0]/2))
    scale_bin_y_val, scale_bin_y_index, _ = plt.hist(scale[:, 1], bins=resolution,
                                                     range=(
                                                     scale_median[1] - scale_iqr[1]/2, scale_median[1] + scale_iqr[1]/2))
    scale_bin_x_val, scale_bin_x_index, _ = plt.hist(scale[:, 2], bins=resolution,
                                                     range=(
                                                     scale_median[2] - scale_iqr[2]/2, scale_median[2] + scale_iqr[2]/2))
    intercept_bin_z_val, intercept_bin_z_index, _ = plt.hist(intercept[:, 0], bins=resolution,
                                                             range=(intercept_median[0] - intercept_iqr[0]/2,
                                                                    intercept_median[0] + intercept_iqr[0]/2))
    intercept_bin_y_val, intercept_bin_y_index, _ = plt.hist(intercept[:, 1], bins=resolution,
                                                             range=(intercept_median[1] - intercept_iqr[1]/2,
                                                                    intercept_median[1] + intercept_iqr[1]/2))
    intercept_bin_x_val, intercept_bin_x_index, _ = plt.hist(intercept[:, 2], bins=resolution,
                                                             range=(intercept_median[2] - intercept_iqr[2]/2,
                                                                    intercept_median[2] + intercept_iqr[2]/2))

    # Finally, obtain our estimates
    scale_estimate = np.array([scale_bin_z_index[np.argmax(scale_bin_z_val)],
                               scale_bin_y_index[np.argmax(scale_bin_y_val)],
                               scale_bin_x_index[np.argmax(scale_bin_x_val)]])
    intercept_estimate = np.array([intercept_bin_z_index[np.argmax(intercept_bin_z_val)],
                                   intercept_bin_y_index[np.argmax(intercept_bin_y_val)],
                                   intercept_bin_x_index[np.argmax(intercept_bin_x_val)]])

    # Finally, build the viewer
    if view:
        plt.subplot(3, 2, 1)
        plt.plot(scale_bin_z_index[:-1], scale_bin_z_val, 'b')
        plt.vlines(scale_estimate[0], 0, np.max(scale_bin_z_val), colors='r',label='z-scale estimate='+
                                                                                   str(np.round(scale_estimate[0], 5)))
        plt.title('Histogram of z-scale estimates')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(intercept_bin_z_index[:-1], intercept_bin_z_val, 'g')
        plt.vlines(intercept_estimate[0], 0, np.max(intercept_bin_z_val), colors='r', label='z-intercept estimate=' +
                                                                        str(np.round(intercept_estimate[0], 5)))
        plt.title('Histogram of z-intercept estimates')
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(scale_bin_y_index[:-1], scale_bin_y_val, 'b')
        plt.vlines(scale_estimate[1], 0, np.max(scale_bin_y_val), colors='r', label='y-scale estimate=' +
                                                                                    str(np.round(scale_estimate[1],5)))
        plt.title('Histogram of y-scale estimates')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(intercept_bin_y_index[:-1], intercept_bin_y_val, 'g')
        plt.vlines(intercept_estimate[1], 0, np.max(intercept_bin_y_val), colors='r', label='y-intercept estimate=' +
                                                                                str(np.round(intercept_estimate[1], 5)))
        plt.title('Histogram of y-intercept estimates')
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(scale_bin_x_index[:-1], scale_bin_x_val, 'b')
        plt.vlines(scale_estimate[2], 0, np.max(scale_bin_x_val), colors='r', label='x-scale estimate=' +
                                                                                    str(np.round(scale_estimate[2],5)))
        plt.title('Histogram of x-scale estimates')
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(intercept_bin_x_index[:-1], intercept_bin_x_val, 'g')
        plt.vlines(intercept_estimate[2], 0, np.max(intercept_bin_x_val), colors='r', label='x-intercept estimate=' +
                                                                                str(np.round(intercept_estimate[2], 5)))
        plt.title('Histogram of x-intercept estimates')
        plt.legend()

        plt.suptitle('Robust Theil Stein Estimation with Parameters: num_pairs = ' + str(num_pairs) +
                     ' (' + str(int(100*num_pairs/(0.5*(z_subvols*y_subvols*x_subvols)**2))) + '%), boundary_erosion = ' +
                     str(boundary_erosion) +
                     ' (' + str((100*boundary_erosion/image_dims).astype(int)) + '%), ' +
                     'image_dims = ' + str(image_dims)  + ', dist_thresh = ' + str(dist_thresh) + ', resolution = ' +
                     str(resolution), fontsize=12)
        plt.show()

    # Populate the transform array
    np.fill_diagonal(transform, scale_estimate)
    transform[:, 3] = intercept_estimate

    return transform, scale_median, scale_iqr, intercept_median, intercept_iqr


# ols with outlier removal, in use
def ols_regression_robust(shift, position, spread):
    """
    Uses OLS regression on data points within 1 IQR of the mean.
    Args:
        shift: z_sv x y_sv x x_sv x 3 array which of shifts in zyx format
        position: z_sv x y_sv x x_sv x 3 array which of positions in zyx format
        spread: number of iqrs away from the median which we intend to use in dataset
    Returns:
        transform: 3 x 4 affine transform in yxz format with final col being shift
    """
    # First reshape the shift and position array to the point where their first 3 axes become 1 and final axis untouched
    shift = np.reshape(shift, (shift.shape[0] * shift.shape[1] * shift.shape[2], 3))
    position = np.reshape(position, (position.shape[0] * position.shape[1] * position.shape[2], 3))

    # We are going to get rid of the shifts where any of the values are nan for regression
    position = position[~np.isnan(shift[:, 0])]
    shift = shift[~np.isnan(shift[:, 0])]

    # Now take median and IQR to filter inliers from outliers
    median = np.nanmedian(shift, axis=0, )
    iqr = stats.iqr(shift, axis=0, nan_policy='omit')
    # valid would be n_shifts x 3 array, we want the columns to get collapsed into 1 where all 3 conditions hold,
    # so collapse by setting all along the first axis
    valid = (shift <= median + iqr * spread).all(axis=1) * (shift >= median - iqr * spread).all(axis=1)

    # Disregard outlier shifts, then pad the position to accommodate for global shift
    shift = shift[valid]
    position = position[valid]
    new_position = position + shift
    position = np.vstack((position.T, np.ones(sum(valid)))).T

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
        if i > 0 and n_matches[i] > n_matches[i-1]:
            n_matches[i:] = n_matches[i] * np.ones(n_iters - i)
            break

    return transform, n_matches, error


# Simple function to get rid of outlier transforms by comparing with other tiles
def regularise_transforms(transform, residual_threshold, tile_origin):
    """
    Function to regularise outliers in the transform section. Outliers will be detected in both the shift and
    scaling components of the transforms and we will replace poor tiles transforms with good transforms in neighbouring
    tiles.
    Args:
        transform: Either [n_tiles x n_rounds x 4 x 3] or [n_tiles x n_channels x 4 x 3]
        residual_threshold: This is a threshold above which we will consider a point to be an outlier
        tile_origin: yxz positions of the tiles [n_tiles x 3]

    Returns:
        transform_regularised: Either [n_tiles x n_rounds x 4 x 3] or [n_tiles x x_channels x 4 x 3]
    """
    # First swap columns so that tile origins are in zyx like the shifts are
    i, j = 0, 1
    tile_origin.T[[i, j]] = tile_origin.T[[j, i]]
    i, j = 0, 2
    tile_origin.T[[i, j]] = tile_origin.T[[j, i]]
    # Now initialise commonly used variables
    n_tiles, n_trials = transform.shape[:2]
    shift = transform[:, :, :, 3]
    predicted_shift = np.zeros_like(shift)
    # This gives us a different set of shifts for each round/channel, let's compute the regressions for each of these
    # rounds/channels independently
    for trial in range(n_trials):
        padded_origin = np.vstack((tile_origin.T, np.ones(n_tiles))).T
        big_transform, _, _, _ = np.linalg.lstsq(padded_origin, shift[:, trial], rcond=None)
        predicted_shift[:, trial] = (padded_origin @ big_transform)[:, :3]

    # Use these predicted shifts to get rid of outliers
    residual = np.linalg.norm(predicted_shift-shift, axis=2)
    shift_outlier = residual > residual_threshold
    # The nice thing about this approach is that we can immediately replace the outliers with their prediction
    # Now we do something similar for the scales, though we don't really assume these will vary so we just take medians
    scale = np.swapaxes(np.array([transform[:, :, 0, 0].T, transform[:, :, 1, 1].T, transform[:, :, 2, 2].T]), 0, 2)
    scale_median = np.median(scale, axis=0)
    scale_iqr = stats.iqr(scale, axis=0)
    scale_low = scale < scale_median - 1.5 * scale_iqr
    scale_high = scale > scale_median + 1.5 * scale_iqr
    scale_outlier = (scale_low + scale_high).any(axis=-1)

    # Now we have everything to generate the regularised transforms
    transform_regularised = transform.copy()
    for t in range(n_tiles):
        for trial in range(n_trials):
            if shift_outlier[t, trial] or scale_outlier[t, trial]:
                transform_regularised[t, trial] = np.vstack((np.diag(scale_median[trial]), predicted_shift[t, trial])).T

    return transform_regularised


# Complicated but necessary function to get rid of aliasing issues in Fourier Shifts
def disambiguate_z(base_image, target_image, shift, r_threshold):
    """
    Function to disambiguate the 2 possible shifts obtained via aliasing.
    Args:
        base_image: nz x ny x nx ndarray
        target_image: nz x ny x nx ndarray
        shift: z y x shift
        r_threshold: threshold that r_statistic must exceed for us to accept z shift of 0 ass opposed to alisases
    Returns:
        shift: z y x corrected shift
    """
    # First we have to compute alternate shift
    if shift[0] >= 0:
        alt_shift = shift - np.array([12, 0, 0])
    else:
        alt_shift = shift + np.array([12, 0, 0])
    # Now we need to compute the shift of base image under both shift and alt_shift
    shift_base = custom_shift(base_image, np.round(shift).astype(int))
    alt_shift_base = custom_shift(base_image, np.round(alt_shift).astype(int))
    shift_mask = shift_base != 0
    alt_shift_mask = alt_shift_base != 0

    # We run into problems when the shift is 0. In this case, aliases of it would be multiples of the box length and
    # so would have no overlap.
    if np.round(shift[0]) == 0:
        shift_corr = stats.pearsonr(shift_base[shift_mask], target_image[shift_mask])[0]
        alt_shift_corr = 0
    else:
        shift_corr = stats.pearsonr(shift_base[shift_mask], target_image[shift_mask])[0]
        alt_shift_corr = stats.pearsonr(alt_shift_base[alt_shift_mask], target_image[alt_shift_mask])[0]

    # We have 2 cases to consider, when shift[0] = 0 and else
    if np.round(shift[0]) == 0:
        # This is the degenerate case, so if we pass the score thresh accept no z-shift
        if shift_corr > r_threshold:
            corrected_shift = shift
        else:
            corrected_shift = np.array([np.nan, shift[1], shift[2]])
    else:
        # This is the non-degenerate case
        # If shift correlation does better, then make that default else use the alternative
        if shift_corr > alt_shift_corr:
            corrected_shift = shift
        else:
            corrected_shift = alt_shift

    return corrected_shift, shift_corr, alt_shift_corr
