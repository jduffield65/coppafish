import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel
from sklearn.linear_model import RANSACRegressor
from coppafish.setup import NotebookPage, Notebook
from coppafish.utils.npy import load_tile
matplotlib.use('MacOSX')


def split_3d_image(image, z_subvolumes, y_subvolumes, x_subvolumes):
    """
    Splits a 3D image into y_subvolumes * x_subvolumes * z_subvolumes subvolumes.

    Parameters
    ----------
    image : ndarray
        The 3D image to be split.
    y_subvolumes : int
        The number of subvolumes to split the image into in the y dimension.
    x_subvolumes : int
        The number of subvolumes to split the image into in the x dimension.
    z_subvolumes : int
        The number of subvolumes to split the image into in the z dimension.

    Returns
    -------
    subvolumes : ndarray
        An array of subvolumes. The first three dimensions index the subvolume, the rest store the actual data.
    positions: ndarray
        (y_subvolumes * x_subvolumes * z_sub_volumes) The middle coord of each subtile
    """

    # Calculate the size of each subvolume in each dimension
    z_size = int(image.shape[0] / z_subvolumes)
    y_size = int(image.shape[1] / y_subvolumes)
    x_size = int(image.shape[2] / x_subvolumes)

    # Create an array to store the subvolumes in
    subvolumes = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, z_size, y_size, x_size))
    positions = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, 3))

    # Split the image into subvolumes and store them in the array
    for z in range(z_subvolumes):
        for y in range(y_subvolumes):
            for x in range(x_subvolumes):
                subvolumes[z, y, x] = image[ z * z_size:(z + 1) * z_size, y * y_size:(y + 1) * y_size, x * x_size:(x + 1) * x_size]
                positions[z, y, x] = [(z + 1/2) * z_size, (y + 1/2) * y_size, (x + 1/2) * x_size]

    return subvolumes, positions


def find_shift_array(subvol_base, subvol_target):
    """
    This function takes in 2 split up 3d images and finds the optimal shift from each subvolume in 1 to it's corresponding
    subvolume in the other.
    Args:
        subvol_base: Base subvolume array
        subvol_target: Target subvolume array

    Returns:
        shift: 4D array, with first 3 dimensions referring to subvolume index and final dim referring to shift.
    """

    if subvol_base.shape != subvol_target.shape:
        raise ValueError("Subvolume arrays have different shapes")

    z_subvolumes, y_subvolumes, x_subvolumes = subvol_base.shape[0], subvol_base.shape[1], subvol_base.shape[2]
    shift = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, 3))

    for z in range(z_subvolumes):
        for y in range(y_subvolumes):
            for x in range(x_subvolumes):
                shift[z, y, x], _, _ = phase_cross_correlation(subvol_target[z, y, x], subvol_base[z, y, x],
                                                               upsample_factor=10)

    return shift


def find_random_shift_array(base_image, target_image, num_samples, z_box_size, y_box_size, x_box_size):
    """
    This function takes in 2 images and finds num_samples random boxes uniformly across the target_image, each of size
    z_box, y_box, x_box. The corresponding box is found in the base image and the shift between these computed with
    a phase cross correlation algorithm.

    Args:
        base_image: Base image array
        target_image: Target image array
        num_samples: The number of random boxes we intend to create
        z_box_size: The z-dimension of the random boxes
        y_box_size: The y-dimension of the random boxes
        x_box_size: The x-dimension of the random boxes

    Returns:
        shift: num_samples x 3 array of shift in zyx coords
        position: num_samples x 3 array of position of centre of box in zyx coords
    """

    position = np.zeros((num_samples, 3))
    shift = np.zeros((num_samples, 3))
    z_image_size, y_image_size, x_image_size = base_image.shape

    for i in range(num_samples):
        origin = np.array([np.random.randint(z_image_size - z_box_size), np.random.randint(y_image_size - y_box_size),
                           np.random.randint(x_image_size - x_box_size)])
        position[i] = origin + np.array([z_box_size, y_box_size, x_box_size])/2
        random_base = base_image[origin[0]:origin[0] + z_box_size, origin[1]:origin[1] + y_box_size,
                      origin[2]:origin[2] + x_box_size]
        random_target = target_image[origin[0]:origin[0] + z_box_size, origin[1]:origin[1] + y_box_size,
                        origin[2]:origin[2] + x_box_size]
        shift[i], _, _ = phase_cross_correlation(random_target, random_base, upsample_factor=10)

    return shift, position


def find_affine_transform(shift, position):
    """
    Function which finds the best affine transform taking coords position[z,y,x] to position[z,y,x] + shift[z,y,x] for
    all subvolumes [z,y,x].
    Args:
        shift:(z_subvolumes * y_subvolumes * x_subvolumes * 3) array of shifts
        position: (z_subvolumes * y_subvolumes * x_subvolumes * 3) array of positions of base points

    Returns:
        transform: (3 x 4) Best affine transform fitting the transform.
    """

    # First, pad the position array
    position = np.pad(position, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1)

    # Now, get rid of outliers in the shifts
    shift_abs = np.linalg.norm(shift, axis=3)
    use = (np.median(shift_abs) - stats.iqr(shift_abs) < shift_abs) * \
          (shift_abs < np.median(shift_abs) + stats.iqr(shift_abs))

    # Complete regression on shifts that we intend to use
    omega_t = np.linalg.lstsq(position[use], shift[use])

    # Compute the transform from the regression matrix
    transform = omega_t[0].T + np.pad(np.eye(3), ((0, 0), (0, 1)))

    return transform


def find_affine_transform_robust(shift, position):
    """
    Uses RANSAC to find optimal affine transform matching the shifts to their positions.
    Args:
        shift: n_samples x 3 array which of shifts in zyx format
        position: n_samples x 3 array which of positions in zyx format

    Returns:
        transform: 3 x 4 affine transform in yxz format with final col being shift
    """
    # Define the regressor
    ransac = RANSACRegressor()

    ransac.fit(position, shift)

    transform = np.hstack((ransac.estimator_.coef_ + np.eye(3), ransac.estimator_.intercept_[:, np.newaxis]))

    return transform


def register(nbp_basic: NotebookPage, nbp_file: NotebookPage):
    """
    Registration pipeline. Returns register Notebook Page.
    Args:
        nbp_basic: (NotebookPage) Basic Info notebook page
        nbp_file: (NotebookPage) File Names notebook page

    Returns:
        nbp: (NotebookPage) Register notebook page
    """

    # Initialise frequently used variables
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
    r_ref, c_ref = nbp_basic.ref_round, nbp_basic.ref_channel

    # First we'll compute the transforms from (r_ref, c_ref) to (r, c_ref)
    round_transform = np.zeros((n_tiles, n_rounds, 3, 4))
    # After this, we'll compute the transforms from (n_rounds // 2, c_ref) to (n_rounds // 2, c) for all c in use
    channel_transform = np.zeros((n_tiles, n_channels, 3, 4))
    # These will then be combined into a single array by composition
    transform = np.zeros((n_tiles, n_rounds, n_channels, 3, 4))

    # In order to determine these transforms, we take 500 random cuboids at teh same point in both images, and examine
    # their shifts using a phase cross correlation algorithm. We then use a phase cross correlation algorithm to
    # find the shift from each of these cuboids to its corresponding cuboid in the target image.
    # We then use a robust regression algorithm (RANSAC) to find the affine transform that best fits these shifts.
    round_shift = np.zeros((n_tiles, n_rounds, 500, 3))
    round_position = np.zeros((n_tiles, n_rounds, 500, 3))
    channel_shift = np.zeros((n_tiles, n_channels, 500, 3))
    channel_position = np.zeros((n_tiles, n_channels, 500, 3))

    with tqdm(total=1 * (n_rounds + len(use_channels))) as pbar:
        pbar.set_description(f"Inter Round and Inter Channel Transforms")
        # Find the inter round transforms and inter channel transforms
        for t in use_tiles:
            # Load in the anchor npy volume, only need to do this once per tile
            anchor_image = sobel(load_tile(nbp_file, nbp_basic, t, r_ref, c_ref))
            for r in use_rounds:
                pbar.set_postfix({'tile': f'{t}', 'round': f'{r}'})
                # Load in imaging npy volume.
                target_image = sobel(load_tile(nbp_file, nbp_basic, t, r, c_ref))

                # next we generate random cuboids from each of these images and obtain our shift arrays
                round_shift[t, r], round_position[t, r] = find_random_shift_array(anchor_image, target_image, 500, 10,
                                                                                  200, 200)

                # Use these subvolumes shifts to find the affine transform taking the volume (t, r_ref, c_ref) to
                # (t, r, c_ref)
                round_transform[t, r] = find_affine_transform_robust(round_shift[t,r], round_position[t,r])
                pbar.update(1)

            # Begin the channel transformation calculations. This requires us to use a central round.
            r = n_rounds // 2
            ref_image = sobel(load_tile(nbp_file, nbp_basic, t, r, c_ref))
            for c in use_channels:
                pbar.set_postfix({'tile': f'{t}', 'channel': f'{c}'})
                # Load in imaging npy volume
                target_image = sobel(load_tile(nbp_file, nbp_basic, t, r, c))

                # next we generate random cuboids from each of these images and obtain our shift arrays
                channel_shift[t, c], channel_position[t, c] = find_random_shift_array(anchor_image, target_image, 500,
                                                                                      10, 200, 200)

                # Use these subvolumes shifts to find the affine transform taking the volume (t, r, c_ref) to
                # (t, r, c)
                channel_transform[t, c] = find_affine_transform_robust(channel_shift[t, c], channel_position[t, c])
                pbar.update(1)

            # Combine all transforms
            for r in use_rounds:
                for c in use_channels:
                    # Next we need to compose these affine transforms.
                    transform[t, r, c] = channel_transform[t, c, :3, :3] @ round_transform[t, r] + \
                                         channel_transform[t, c, :, 3]


def view_shift_hist(shift):
    """
        View histograms for shift array.
    Args:
        shift: Shift array for subvolumes.
    """

    shift_abs = np.linalg.norm(shift, axis=-1)

    plt.subplot(2, 2, 1)
    plt.hist(np.reshape(shift_abs, 400), 100)
    plt.vlines(x=np.median(shift_abs), ymin=0, ymax=200, colors='y', linestyles='dotted')
    plt.vlines(x=np.median(shift_abs) - 2 * stats.iqr(shift_abs), ymin=0, ymax=200, color='red')
    plt.vlines(x=np.median(shift_abs) + 2 * stats.iqr(shift_abs), ymin=0, ymax=200, color='red')
    plt.title(label='Histogram of absolute value of the shifts. Median = ' + str(np.median(shift_abs)) + '. IQR = ' +
                    str(stats.iqr(shift_abs)))

    plt.subplot(2, 2, 2)
    plt.hist(np.reshape(shift, (400, 3))[:, 0], 100)
    plt.vlines(x=np.median(shift[:, :, :, 0]), ymin=0, ymax=200, colors='y', linestyles='dotted')
    plt.vlines(x=np.median(shift[:, :, :, 0]) - 2 * stats.iqr(shift[:, :, :, 0]), ymin=0, ymax=200, color='red')
    plt.vlines(x=np.median(shift[:, :, :, 0]) + 2 * stats.iqr(shift[:, :, :, 0]), ymin=0, ymax=200, color='red')
    plt.title(label='Histogram of the z shifts. Median = ' + str(np.median(shift[:, :, :, 0])) + '. IQR = ' +
                    str(stats.iqr(shift[:, :, :, 0])))

    plt.subplot(2, 2, 3)
    plt.hist(np.reshape(shift, (400, 3))[:, 1], 100)
    plt.vlines(x=np.median(shift[:, :, :, 1]), ymin=0, ymax=200, colors='y', linestyles='dotted')
    plt.vlines(x=np.median(shift[:, :, :, 1]) - 2 * stats.iqr(shift[:, :, :, 1]), ymin=0, ymax=200, color='red')
    plt.vlines(x=np.median(shift[:, :, :, 1]) + 2 * stats.iqr(shift[:, :, :, 1]), ymin=0, ymax=200, color='red')
    plt.title(label='Histogram of the y shifts. Median = ' + str(np.median(shift[:, :, :, 1])) + '. IQR = ' +
                    str(stats.iqr(shift[:, :, :, 1])))

    plt.subplot(2, 2, 4)
    plt.hist(np.reshape(shift, (400, 3))[:, 2], 100)
    plt.vlines(x=np.median(shift[:, :, :, 2]), ymin=0, ymax=200, colors='y', linestyles='dotted')
    plt.vlines(x=np.median(shift[:, :, :, 2]) - 2 * stats.iqr(shift[:, :, :, 2]), ymin=0, ymax=200, color='red')
    plt.vlines(x=np.median(shift[:, :, :, 2]) + 2 * stats.iqr(shift[:, :, :, 2]), ymin=0, ymax=200, color='red')
    plt.title(label='Histogram of the x shifts. Median = ' + str(np.median(shift[:, :, :, 2])) + '. IQR = ' +
                    str(stats.iqr(shift[:, :, :, 2])))

    plt.show()


def view_shift_scatter(shift, position, z_box, y_box, x_box, alpha = 1.0, score = None, save_robust_regression = False,
                       view_robust_regression = False):
    """
    Function which shows the regression in x, y and z.
    Args:
        shift: (y_subvols, x_subvols, z_subvols, 3) or (n_shifts, 3)
        position: (y_subvols, x_subvols, z_subvols, 3) or (n_points, 3)
        alpha: (float) opacity of points in scatter plot
        score: (y_subvols, x_subvols, z_subvols, 1) or (n_shifts, 1) score of shifts
        view_robust_regression: option to view RANSAC algorithms robust linear regression prediction
    Returns:
        scale: (3 x 1 ) z, y, x scales
        shift (3 x 1)  z, y, x shift
    """
    if np.ndim(shift) == 4:
        z_subvols, y_subvols, x_subvols, _ = shift.shape

        z_shift = np.reshape(shift[:, :, :, 0], y_subvols * x_subvols * z_subvols)
        y_shift = np.reshape(shift[:, :, :, 1], y_subvols * x_subvols * z_subvols)
        x_shift = np.reshape(shift[:, :, :, 2], y_subvols * x_subvols * z_subvols)

        z_pos = np.reshape(position[:, :, :, 0], y_subvols * x_subvols * z_subvols)
        y_pos = np.reshape(position[:, :, :, 1], y_subvols * x_subvols * z_subvols)
        x_pos = np.reshape(position[:, :, :, 2], y_subvols * x_subvols * z_subvols)

    else:
        z_shift = shift[:, 0]
        y_shift = shift[:, 1]
        x_shift = shift[:, 2]

        z_pos = position[:, 0]
        y_pos = position[:, 1]
        x_pos = position[:, 2]

    if score is not None:
        q1 = np.quantile(score, 0.02)
        plt.subplot(3, 1, 1)
        plt.scatter(z_pos[score > q1], z_shift[score > q1], alpha=alpha, c='blue')
        plt.scatter(z_pos[score < q1], z_shift[score < q1], alpha=alpha, c='red')
        plt.title('z shifts against z positions')

        plt.subplot(3, 1, 2)
        plt.scatter(y_pos[score > q1], y_shift[score > q1], alpha=alpha, c='blue')
        plt.scatter(y_pos[score < q1], y_shift[score < q1], alpha=alpha, c='red')
        plt.title('y shifts against y positions')

        plt.subplot(3, 1, 3)
        plt.scatter(x_pos[score > q1], x_shift[score > q1], alpha=alpha, c='blue')
        plt.scatter(x_pos[score < q1], x_shift[score < q1], alpha=alpha, c='red')
        plt.title('x shifts against x positions')

    else:
        plt.subplot(3, 1, 1)
        plt.scatter(z_pos, z_shift, alpha=alpha, c='blue')
        plt.title('z shifts against z positions')

        plt.subplot(3, 1, 2)
        plt.scatter(y_pos, y_shift, alpha=alpha, c='blue')
        plt.title('y shifts against y positions')

        plt.subplot(3, 1, 3)
        plt.scatter(x_pos, x_shift, alpha=alpha, c='blue')
        plt.title('x shifts against x positions')

    ransac_z = RANSACRegressor()
    ransac_x = RANSACRegressor()
    ransac_y = RANSACRegressor()
    ransac_z.fit(z_pos[:, np.newaxis], z_shift)
    ransac_y.fit(y_pos[:, np.newaxis], y_shift)
    ransac_x.fit(x_pos[:, np.newaxis], x_shift)

    line_input_z = np.arange(z_pos.min(), z_pos.max())[:, np.newaxis]
    line_z_ransac = ransac_z.predict(line_input_z)
    line_input_y = np.arange(y_pos.min(), y_pos.max())[:, np.newaxis]
    line_y_ransac = ransac_y.predict(line_input_y)
    line_input_x = np.arange(x_pos.min(), x_pos.max())[:, np.newaxis]
    line_x_ransac = ransac_x.predict(line_input_x)

    plt.subplot(3, 1, 1)
    plt.plot(line_input_z, line_z_ransac,label='Scale ='+str((ransac_z.estimator_.coef_ + 1)[0]) +
                                               '. Intercept = ' + str(ransac_z.estimator_.intercept_))
    plt.legend(loc='upper right', prop={'size': 6})

    plt.subplot(3, 1, 2)
    plt.plot(line_input_y, line_y_ransac,label='Scale ='+str((ransac_y.estimator_.coef_ + 1)[0]) +
                                               '. Intercept = ' + str(ransac_y.estimator_.intercept_))
    plt.legend(loc='upper right', prop={'size': 6})

    plt.subplot(3, 1, 3)
    plt.plot(line_input_x, line_x_ransac,label='Scale ='+str((ransac_x.estimator_.coef_ + 1)[0]) +
                                               '. Intercept = ' + str(ransac_x.estimator_.intercept_))
    plt.legend(loc='upper right', prop={'size': 6})

    plt.subplots_adjust(hspace=1)

    plt.suptitle('samples='+str(y_shift.shape[0])+', z_box='+str(z_box)+',y_box='+str(y_box)+', x_box='+str(x_box),
                 fontsize=14)

    if save_robust_regression:
        plt.savefig('/Users/reillytilbury/Desktop/Tile 0/Registration Images/samples='+str(y_shift.shape[0])+
                    ', z_box='+str(z_box)+',y_box='+str(y_box)+', x_box='+str(x_box)+'.png',dpi=400,bbox_inches='tight')

    if view_robust_regression:
        plt.show()

    return np.array([(ransac_z.estimator_.coef_ + 1)[0], (ransac_y.estimator_.coef_ + 1)[0],
                     (ransac_x.estimator_.coef_ + 1)[0]]) , np.array([ransac_z.estimator_.intercept_,
                                                                      ransac_y.estimator_.intercept_,
                                                                      ransac_x.estimator_.intercept_])


def view_shift_scatter_multivariate(shift, position, z_box, y_box, x_box, alpha = 1.0, score = None, save_robust_regression = False,
                       view_robust_regression = False):
    """
        Function which shows the regression in x, y and z. Works similarly to view_shift_scatter but now regression is
        done in 3D instead of independently.
        Args:
            shift: (y_subvols, x_subvols, z_subvols, 3) or (n_shifts, 3)
            position: (y_subvols, x_subvols, z_subvols, 3) or (n_points, 3)
            alpha: (float) opacity of points in scatter plot
            score: (y_subvols, x_subvols, z_subvols, 1) or (n_shifts, 1) score of shifts
            view_robust_regression: option to view RANSAC algorithms robust linear regression prediction
        Returns:
            scale: (3 x 1 ) z, y, x scales
            shift (3 x 1)  z, y, x shift
        """
    if np.ndim(shift) == 4:
        z_subvols, y_subvols, x_subvols, _ = shift.shape

        z_shift = np.reshape(shift[:, :, :, 0], y_subvols * x_subvols * z_subvols)
        y_shift = np.reshape(shift[:, :, :, 1], y_subvols * x_subvols * z_subvols)
        x_shift = np.reshape(shift[:, :, :, 2], y_subvols * x_subvols * z_subvols)

        z_pos = np.reshape(position[:, :, :, 0], y_subvols * x_subvols * z_subvols)
        y_pos = np.reshape(position[:, :, :, 1], y_subvols * x_subvols * z_subvols)
        x_pos = np.reshape(position[:, :, :, 2], y_subvols * x_subvols * z_subvols)

    else:
        z_shift = shift[:, 0]
        y_shift = shift[:, 1]
        x_shift = shift[:, 2]

        z_pos = position[:, 0]
        y_pos = position[:, 1]
        x_pos = position[:, 2]

    if score is not None:
        q1 = np.quantile(score, 0.02)
        plt.subplot(3, 1, 1)
        plt.scatter(z_pos[score > q1], z_shift[score > q1], alpha=alpha, c='blue')
        plt.scatter(z_pos[score < q1], z_shift[score < q1], alpha=alpha, c='red')
        plt.title('z shifts against z positions')

        plt.subplot(3, 1, 2)
        plt.scatter(y_pos[score > q1], y_shift[score > q1], alpha=alpha, c='blue')
        plt.scatter(y_pos[score < q1], y_shift[score < q1], alpha=alpha, c='red')
        plt.title('y shifts against y positions')

        plt.subplot(3, 1, 3)
        plt.scatter(x_pos[score > q1], x_shift[score > q1], alpha=alpha, c='blue')
        plt.scatter(x_pos[score < q1], x_shift[score < q1], alpha=alpha, c='red')
        plt.title('x shifts against x positions')

    else:
        plt.subplot(3, 1, 1)
        plt.scatter(z_pos, z_shift, alpha=alpha, c='blue')
        plt.title('z shifts against z positions')

        plt.subplot(3, 1, 2)
        plt.scatter(y_pos, y_shift, alpha=alpha, c='blue')
        plt.title('y shifts against y positions')

        plt.subplot(3, 1, 3)
        plt.scatter(x_pos, x_shift, alpha=alpha, c='blue')
        plt.title('x shifts against x positions')

    ransac = RANSACRegressor()
    ransac.fit(np.hstack((z_pos[:, np.newaxis],y_pos[:, np.newaxis], x_pos[:, np.newaxis])),
               np.hstack((z_shift[:,np.newaxis], y_shift[:,np.newaxis], x_shift[:, np.newaxis])))

    line_input_z = np.arange(z_pos.min(), z_pos.max())[:, np.newaxis]
    zeros_z = np.zeros(line_input_z.shape[0])[:, np.newaxis]
    line_z_ransac = ransac.predict(np.hstack((line_input_z, zeros_z, zeros_z)))[:, 0]
    line_input_y = np.arange(y_pos.min(), y_pos.max())[:, np.newaxis]
    zeros_y = np.zeros(line_input_y.shape[0])[:, np.newaxis]
    line_y_ransac = ransac.predict(np.hstack((zeros_y, line_input_y, zeros_y)))[:, 1]
    line_input_x = np.arange(x_pos.min(), x_pos.max())[:, np.newaxis]
    zeros_x = np.zeros(line_input_x.shape[0])[:, np.newaxis]
    line_x_ransac = ransac.predict(np.hstack((zeros_x, zeros_x, line_input_x)))[:, 2]

    plt.subplot(3, 1, 1)
    plt.plot(line_input_z, line_z_ransac, label='Scale =' + str((ransac.estimator_.coef_ + 1)[0,0]) +
                                                '. Intercept = ' + str(ransac.estimator_.intercept_[0]))
    plt.legend(loc='upper right', prop={'size': 6})

    plt.subplot(3, 1, 2)
    plt.plot(line_input_y, line_y_ransac, label='Scale =' + str((ransac.estimator_.coef_ + 1)[1,1]) +
                                                '. Intercept = ' + str(ransac.estimator_.intercept_[1]))
    plt.legend(loc='upper right', prop={'size': 6})

    plt.subplot(3, 1, 3)
    plt.plot(line_input_x, line_x_ransac, label='Scale =' + str((ransac.estimator_.coef_ + 1)[2,2]) +
                                                '. Intercept = ' + str(ransac.estimator_.intercept_[2]))
    plt.legend(loc='upper right', prop={'size': 6})

    plt.subplots_adjust(hspace=1)

    plt.suptitle(
        'samples=' + str(y_shift.shape[0]) + ', z_box=' + str(z_box) + ',y_box=' + str(y_box) + ', x_box=' + str(x_box),
        fontsize=14)

    if save_robust_regression:
        plt.savefig('/Users/reillytilbury/Desktop/Tile 0/Registration Images/Multivariate/samples=' + str(y_shift.shape[0]) +
                    ', z_box=' + str(z_box) + ',y_box=' + str(y_box) + ', x_box=' + str(x_box) + '.png', dpi=400,
                    bbox_inches='tight')

    if view_robust_regression:
        plt.show()

    return np.array([(ransac.estimator_.coef_ + 1)[0,0], (ransac.estimator_.coef_ + 1)[1,1],
                     (ransac.estimator_.coef_ + 1)[2,2]]), np.array([ransac.estimator_.intercept_[0],
                                                                     ransac.estimator_.intercept_[1],
                                                                     ransac.estimator_.intercept_[2]])


def view_robust_custom_regressor(X: np.ndarray, Y: np.ndarray, z_box, y_box, x_box, num_pairs = None, alpha = 1.0,
                            save_robust_regression = False, view_robust_regression = False):
    """
    Robust regression algorithm for inputs X (num_samples * num_features) and Y (num_responses * num_features).
    This assumes a diagonal affine relation relating Y and X. So need num_samples = num_response.
    Args:
        X: input data (n_samples, n_features)
        Y: output data (n_responses, n_features)
        num_pairs: Number of pairs to randomly select. If left blank, we'll simply pick all pairs.

    Returns:
        scale: (n_features) diagonal of the matrix relating Y and X
        shift: (n_features) shift between Y and X

    """
    num_samples = X.shape[0]
    num_features = X.shape[1]
    # Initialise number of pairings
    if num_pairs is None or num_pairs > num_samples * (num_samples - 1) / 2:
        num_pairs = int(num_samples * (num_samples - 1) / 2)
    scale = np.zeros((num_pairs, num_features))
    intercept = np.zeros((num_pairs, num_features))

    # Now randomly select num_pairs pairs and compute the scales and intercepts
    for i in range(num_pairs):
        # Generate random indices
        index1 = np.random.randint(0,num_samples)
        index2 = np.random.randint(0,num_samples)
        # Use this to generate scales and intercepts
        scale[i] = (Y[index2] - Y[index1])/(X[index2] - X[index1])
        intercept[i] = Y[index1] - scale[i] * X[index1]

    median_scale = np.nanmedian(scale,axis=0)
    iqr_scale = stats.iqr(scale, axis=0,nan_policy='omit')
    median_intercept = np.nanmedian(intercept, axis=0)
    iqr_intercept = stats.iqr(intercept, axis=0,nan_policy='omit')


    # Now plot the histograms and scatter plots.
    # First, histograms
    plt.subplot(3,3,1)
    s1, t1, _ = plt.hist((scale[:, 0]+1)[abs(scale[:,0])<np.inf],bins=1000,
                         range=(median_scale[0]-iqr_scale[0],median_scale[0]+iqr_scale[0]))
    plt.vlines(x=t1[np.argmax(s1)],ymin=0,ymax=np.max(s1),colors='r',label='Scale='+str(t1[np.argmax(s1)]))
    plt.legend(loc='upper right', prop={'size': 6})
    plt.title('Histogram of scale factors for the z-component')

    plt.subplot(3,3,4)
    s4, t4, _ = plt.hist((scale[:, 1]+1)[abs(scale[:,1])<np.inf],bins=1000,
                         range=(median_scale[1]-iqr_scale[1],median_scale[1]+iqr_scale[1]))
    plt.vlines(x=t4[np.argmax(s4)],ymin=0,ymax=np.max(s4),colors='r',label='Scale='+str(t4[np.argmax(s4)]))
    plt.legend(loc='upper right', prop={'size': 6})
    plt.title('Histogram of scale factors for the y-component')


    plt.subplot(3,3,7)
    s7, t7, _ = plt.hist((scale[:, 2]+1)[abs(scale[:,2])<np.inf],bins=1000,
                         range=(median_scale[2] - iqr_scale[2], median_scale[2] + iqr_scale[2]))
    plt.vlines(x=t7[np.argmax(s7)],ymin=0,ymax=np.max(s7),colors='r',label='Scale='+str(t7[np.argmax(s7)]))
    plt.legend(loc='upper right', prop={'size': 6})
    plt.title('Histogram of scale factors for the x-component')

    plt.subplot(3, 3, 2)
    s2, t2, _ = plt.hist((intercept[:, 0])[abs(intercept[:,0])<np.inf], bins=1000,
                         range=(median_intercept[0]-iqr_intercept[0],median_intercept[0]+iqr_intercept[0]))
    plt.vlines(x=t2[np.argmax(s2)],ymin=0,ymax=np.max(s2),colors='r',label='Scale='+str(t2[np.argmax(s2)]))
    plt.legend(loc='upper right', prop={'size': 6})
    plt.title('Histogram of intercepts (shifts) for the z-component')

    plt.subplot(3, 3, 5)
    s5, t5, _ = plt.hist((intercept[:, 1])[abs(intercept[:,1])<np.inf], bins=1000,
                         range=(median_intercept[1]-iqr_intercept[1],median_intercept[1]+iqr_intercept[1]))
    plt.vlines(x=t5[np.argmax(s5)],ymin=0,ymax=np.max(s5),colors='r',label='Scale='+str(t5[np.argmax(s5)]))
    plt.legend(loc='upper right', prop={'size': 6})
    plt.title('Histogram of intercepts (shifts) for the y-component')

    plt.subplot(3, 3, 8)
    s8, t8, _ = plt.hist((intercept[:, 2])[abs(intercept[:,2])<np.inf],bins=1000,
                         range=(median_intercept[2]-iqr_intercept[2],median_intercept[2]+iqr_intercept[2]))
    plt.vlines(x=t8[np.argmax(s8)],ymin=0,ymax=np.max(s8),colors='r',label='Scale='+str(t8[np.argmax(s8)]))
    plt.legend(loc='upper right', prop={'size': 6})
    plt.title('Histogram of intercepts (shifts) for the x-component')

    scale_guess = np.zeros(3)
    intercept_guess = np.zeros(3)

    #  Now the scatter plots
    plt.subplot(3, 3, 3)
    plt.plot(X[:,0], (scale_guess[0] - 1) * X[:,0] + intercept_guess[0])
    plt.scatter(X[:, 0], Y[:, 0], alpha=alpha, c='blue')
    plt.title('z shifts against z positions')

    plt.subplot(3, 3, 6)
    plt.plot(X[:,1], (scale_guess[1] - 1) * X[:,1] + intercept_guess[1])
    plt.scatter(X[:, 1], Y[:, 1], alpha=alpha, c='blue')
    plt.title('y shifts against y positions')

    plt.subplot(3, 3, 9)
    plt.plot(X[:,2], (scale_guess[2] - 1) * X[:,2] + intercept_guess[2])
    plt.scatter(X[:, 2], Y[:, 2], alpha=alpha, c='blue')
    plt.title('x shifts against x positions')

    plt.suptitle(
        'samples=' + str(num_samples) + ', z_box=' + str(z_box) + ',y_box=' + str(y_box) + ', x_box=' + str(x_box),
        fontsize=12)

    plt.subplots_adjust(hspace=0.75)
    plt.subplots_adjust(wspace=0.75)

    if save_robust_regression:
        plt.savefig(
            '/Users/reillytilbury/Desktop/Tile 0/Registration Images/Custom/samples=' + str(num_samples) +
            ', z_box=' + str(z_box) + ',y_box=' + str(y_box) + ', x_box=' + str(x_box) + '.png', dpi=400,
            bbox_inches='tight')

    if view_robust_regression:
        plt.show()

    return scale_guess, intercept_guess


def view_random_cuboids(num_samples: int, target_image: np.ndarray, base_image: np.ndarray, z_box_size: int, y_box_size: int,
                   x_box_size: int, alpha, custom: False):
    """
    Function to view relationship between random sample of cuboids and their optimal shifts.

    Args:
        num_samples: number of random samples
        target_image: imaging round image
        base_image: anchor image
        z_box_size: z random cuboid size
        y_box_size: y random cuboid size
        x_box_size: x random cuboid size

    """
    position = np.zeros((num_samples, 3))
    shift = np.zeros((num_samples, 3))
    score = np.zeros(num_samples)
    z_image_size, y_image_size, x_image_size = base_image.shape

    for i in range(num_samples):
        plt.clf()
        origin = np.array([np.random.randint(z_image_size - z_box_size), np.random.randint(y_image_size - y_box_size),
                           np.random.randint(x_image_size - x_box_size)])
        position[i] = origin + np.array([z_box_size, y_box_size, x_box_size])
        random_base = base_image[origin[0]:origin[0] + z_box_size, origin[1]:origin[1] + y_box_size,
                      origin[2]:origin[2] + x_box_size]
        random_target = target_image[origin[0]:origin[0] + z_box_size, origin[1]:origin[1] + y_box_size,
                        origin[2]:origin[2] + x_box_size]
        shift[i], score[i], _ = phase_cross_correlation(random_target, random_base, upsample_factor=10)

    # Now plot the shifts against the positions
    if not custom:
        scale_zyx, shift_zyx = view_shift_scatter_multivariate(shift, position, z_box=z_box_size,y_box=y_box_size,
                                                               x_box=x_box_size, alpha=alpha, score=score,
                                                               save_robust_regression=True, view_robust_regression=False)
    else:
        scale_zyx, shift_zyx = view_robust_custom_regressor(position, shift, z_box=z_box_size, y_box=y_box_size,
                                                               x_box=x_box_size, num_pairs=10000, alpha=alpha,
                                                               save_robust_regression=True,
                                                               view_robust_regression=False)

    return  scale_zyx, shift_zyx


# anchor_image = sobel(np.load('/Users/reillytilbury/Desktop/Tile 0/round_Cp_anchor_t0c18.npy'))
# r0c18_image = sobel(np.load('/Users/reillytilbury/Desktop/Tile 0/round_Cp_r0_t0c18.npy'))

# next we generate random cuboids from each of these images and obtain our shift arrays
# shift, position = find_random_shift_array(anchor_image, r0c18_image, 500, 10, 200, 200)

# Use these subvolumes shifts to find the affine transform taking the volume (t, r_ref, c_ref) to
# (t, r, c_ref)
# transform = find_affine_transform_robust(shift, position)

# nb = Notebook('//128.40.224.65\SoyonHong\Christina Maat\ISS Data + Analysis\E-2210-001 6mo CP vs Dry\CP/v5_24NOV22\output/notebook.npz',
#               'C:/Users\Reilly\Desktop\Sample Notebooks\Christina/New Registration Test/E-2210-001_CP_settings_v3_22NOV22.ini')
# register(nbp_basic=nb.basic_info, nbp_file=nb.file_names)

# shift = np.zeros((5, 4, 4, 3))
# scale = np.zeros((5, 4, 4, 3))
# sample = [100, 200, 300, 500, 1000]
# z_box = [5, 8, 10, 15]
# xy_box = [20, 50, 100, 200]
#
# for i in range(5):
#     for j in range(4):
#         for k in range(4):
#             plt.clf()
#             scale[i, j, k], shift[i, j, k] = random_cuboids(sample[i], r0c18_image, anchor_image,
#                                                             z_box[j],xy_box[k],xy_box[k], 50/sample[i],custom=True)
#             print(sample[i],z_box[j],xy_box[k])
#
# np.save('/Users/reillytilbury/Desktop/Tile 0/Registration Images/shift.npy', shift)
# np.save('/Users/reillytilbury/Desktop/Tile 0/Registration Images/scale.npy', scale)
#
# print('Hello')