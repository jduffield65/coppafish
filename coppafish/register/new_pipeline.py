import matplotlib
import napari
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel
from skimage.exposure import match_histograms
from coppafish.setup import NotebookPage, Notebook
from coppafish.utils.npy import load_tile
matplotlib.use('Qt5Agg')


def split_3d_image(image, z_subvolumes, y_subvolumes, x_subvolumes, z_box, y_box, x_box):
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
    subvolume : ndarray
        An array of subvolumes. The first three dimensions index the subvolume, the rest store the actual data.
    position: ndarray
        (y_subvolumes * x_subvolumes * z_sub_volumes) The middle coord of each subtile
    """
    z_image, y_image, x_image = image.shape

    # Allow 0.5 of a box either side and then split the middle with subvols evenly spaced points, ie into subvols - 1
    # intervals. Then use integer division. e.g actual unit distance is 12.5, this gives a unit distance of 12 so
    # should never overshoot
    z_unit = (z_image - z_box) // (z_subvolumes - 1)
    y_unit = (y_image - y_box) // (y_subvolumes - 1)
    x_unit = (x_image - x_box) // (x_subvolumes - 1)

    # Create an array to store the subvolumes in
    subvolume = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, z_box, y_box, x_box))

    # Create an array to store the positions in
    position = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, 3))

    # Split the image into subvolumes and store them in the array
    for z in range(z_subvolumes):
        for y in range(y_subvolumes):
            for x in range(x_subvolumes):
                z_centre, y_centre, x_centre = z_box//2 + z * z_unit, y_box//2 + y * y_unit, x_box//2 + x * x_unit
                z_start, z_end = z_centre - z_box//2, z_centre + z_box//2
                y_start, y_end = y_centre - y_box//2, y_centre + y_box//2
                x_start, x_end = x_centre - x_box//2, x_centre + x_box//2

                subvolume[z, y, x] = image[z_start:z_end, y_start:y_end, x_start:x_end]
                position[z, y, x] = np.array([(z_start + z_end)//2, (y_start + y_end)//2, (x_start + x_end)//2])

    return subvolume, position


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


def find_affine_transform_robust_custom(shift, position, num_pairs, boundary_erosion, image_dims, dist_thresh,
                                        resolution, view=False):
    """
    Uses a custom Theil-Stein variant to find optimal affine transform matching the shifts to their positions.
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
                                                     scale_median[0] - scale_iqr[0], scale_median[0] + scale_iqr[0]))
    scale_bin_y_val, scale_bin_y_index, _ = plt.hist(scale[:, 1], bins=resolution,
                                                     range=(
                                                     scale_median[1] - scale_iqr[1], scale_median[1] + scale_iqr[1]))
    scale_bin_x_val, scale_bin_x_index, _ = plt.hist(scale[:, 2], bins=resolution,
                                                     range=(
                                                     scale_median[2] - scale_iqr[2], scale_median[2] + scale_iqr[2]))
    intercept_bin_z_val, intercept_bin_z_index, _ = plt.hist(intercept[:, 0], bins=resolution,
                                                             range=(intercept_median[0] - intercept_iqr[0],
                                                                    intercept_median[0] + intercept_iqr[0]))
    intercept_bin_y_val, intercept_bin_y_index, _ = plt.hist(intercept[:, 1], bins=resolution,
                                                             range=(intercept_median[1] - intercept_iqr[1],
                                                                    intercept_median[1] + intercept_iqr[1]))
    intercept_bin_x_val, intercept_bin_x_index, _ = plt.hist(intercept[:, 2], bins=resolution,
                                                             range=(intercept_median[2] - intercept_iqr[2],
                                                                    intercept_median[2] + intercept_iqr[2]))

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


def compose_affine(A1, A2):
    """
    Function to compose 2 affine transfroms
    Args:
        A1: 3 x 4 affine transform
        A2: 3 x 4 affine transform
    Returns:
        A1 * A2: Composed Affine transform
    """
    # Add Final row, compose and then get rid of final row
    A1 = np.vstack((A1, np.array([0, 0, 0, 1])))
    A2 = np.vstack((A2, np.array([0, 0, 0, 1])))

    composition = (A1 @ A2)[:3, :4]

    return composition


def invert_affine(A):
    """
    Function to invert affine transform.
    Args:
        A: 3 x 4 affine transform

    Returns:
        inverse: 3 x 4 affine inverse transform
    """
    inverse = np.zeros((3, 4))

    inverse[:3, :3] = np.linalg.inv(A[:3, :3])
    inverse[:, 3] = -np.linalg.inv(A[:3, :3]) @ A[:, 3]

    return inverse


def reformat_affine(A, z_scale):
    """
    Function to convert 3 x 4 matrix in z, y, x coords into a 4 x 3 matrix of y, x, z coords and rescale the shift

    Args:
        A: Original transform in old format (3 x 4)
        z_scale: How much to scale z-components by (float)

    Returns:
        A_reformatted: 4 x 3 transform with associated changes

    """
    # Append to A a bottom row
    A = np.vstack((A, np.array([0,0,0,1])))

    # First, convert everything into z, y, x by multiplying by a matrix that swaps rows
    row_shuffler = np.zeros((4,4))
    row_shuffler[0, 2] = 1
    row_shuffler[1, 0] = 1
    row_shuffler[2, 1] = 1
    row_shuffler[3, 3] = 1

    # Finally, compute the matrix in the new basis
    A = np.linalg.inv(row_shuffler) @ A @ row_shuffler

    # Next, multiply the shift part of A by the expansion factor
    A[2, 3] = z_scale * A[2, 3]

    # Remove the final row
    A = A[:3,:4]

    # Finally, transpose the matrix
    A = A.T

    return A


def register(nbp_basic: NotebookPage, nbp_file: NotebookPage, config: dict):

    """
    Registration pipeline. Returns register Notebook Page.
    Args:
        nbp_basic: (NotebookPage) Basic Info notebook page
        nbp_file: (NotebookPage) File Names notebook page
        config: Register part of the config dictionary

    Returns:
        nbp: (NotebookPage) Register notebook page
    """

    # Initialise frequently used variables
    nbp = NotebookPage("register")
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
    r_ref, c_ref = nbp_basic.ref_round, nbp_basic.ref_channel
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy

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
    round_shift = []
    round_position = []
    channel_shift = []
    channel_position = []

    with tqdm(total=(n_rounds + len(use_channels))) as pbar:

        pbar.set_description(f"Inter Round and Inter Channel Transforms")
        # Find the inter round transforms and inter channel transforms
        for t in use_tiles:

            # Load in the anchor npy volume, only need to do this once per tile
            anchor_image = sobel(load_tile(nbp_file, nbp_basic, t, r_ref, c_ref))
            anchor_image_unfiltered = load_tile(nbp_file, nbp_basic, t, r_ref, c_ref)

            # Software was written for z y x, so change it from y x z
            anchor_image = np.swapaxes(anchor_image, 0, 2)
            anchor_image = np.swapaxes(anchor_image, 1, 2)

            for r in use_rounds:
                pbar.set_postfix({'tile': f'{t}', 'round': f'{r}'})

                # Load in imaging npy volume.
                target_image = sobel(load_tile(nbp_file, nbp_basic, t, r, c_ref))
                target_image = np.swapaxes(target_image, 0, 2)
                target_image = np.swapaxes(target_image, 1, 2)

                # next we split image into overlapping cuboids
                subvol_base, position = split_3d_image(image=anchor_image, z_subvolumes=5, y_subvolumes=15, x_subvolumes=15,
                                                       z_box=12, y_box=300, x_box=300)
                subvol_target, _ = split_3d_image(image=target_image, z_subvolumes=5, y_subvolumes=15, x_subvolumes=15,
                                                  z_box=12, y_box=300, x_box=300)

                # Find the subvolume shifts
                shift = find_shift_array(subvol_base, subvol_target)

                # Append these arrays to the round_shift and round_position storage
                round_shift.append(shift)
                round_position.append(position)

                # Use these subvolumes shifts to find the affine transform taking the volume (t, r_ref, c_ref) to
                # (t, r, c_ref)
                round_transform[t, r] = find_affine_transform_robust_custom(shift=shift, position=position,
                                                                            num_pairs=1e5,
                                                                            boundary_erosion=[5, 100, 100],
                                                                            image_dims=anchor_image.shape,
                                                                            dist_thresh=[5, 250, 250], resolution=30)
                pbar.update(1)

            # Begin the channel transformation calculations. This requires us to use a central round.
            r = n_rounds // 2
            # Remove reference channel from comparison as the shift from C18 to C18 is identity
            use_channels.remove(c_ref)
            for c in use_channels:
                pbar.set_postfix({'tile': f'{t}', 'channel': f'{c}'})
                # Load in imaging npy volume
                target_image = load_tile(nbp_file, nbp_basic, t, r, c)
                target_image = np.swapaxes(target_image, 0, 2)
                target_image = np.swapaxes(target_image, 1, 2)

                # Match histograms to unfiltered anchor and then sobel filter
                target_image = match_histograms(target_image, anchor_image_unfiltered)
                target_image = sobel(target_image)

                # next we split image into overlapping cuboids
                subvol_base, position = split_3d_image(image=anchor_image, z_subvolumes=5, y_subvolumes=15, x_subvolumes=15,
                                                       z_box=12, y_box=300, x_box=300)
                subvol_target, _ = split_3d_image(image=target_image, z_subvolumes=5, y_subvolumes=15, x_subvolumes=15,
                                                  z_box=12, y_box=300, x_box=300)

                # Find the subvolume shifts
                shift = find_shift_array(subvol_base, subvol_target)

                # Append these arrays to the round_shift and round_position storage
                channel_shift.append(shift)
                channel_position.append(position)

                # Use these subvolumes shifts to find the affine transform taking the volume (t, r, c_ref) to
                # (t, r, c)
                aux_transform = find_affine_transform_robust_custom(shift=shift, position=position,
                                                                    num_pairs=1e5,
                                                                    boundary_erosion=[5, 100, 100],
                                                                    image_dims=anchor_image.shape,
                                                                    dist_thresh=[5, 250, 250], resolution=30)

                # Now correct for the round shift to get a round independent affine transform
                channel_transform[t, c] = compose_affine(aux_transform, invert_affine(round_transform[t, r]))
                pbar.update(1)

            # Add reference channel back to use_channels
            use_channels.append(c_ref)
            # Combine all transforms
            for r in use_rounds:
                for c in use_channels:
                    # Next we need to compose these affine transforms. Remember, affine transforms cannot be composed
                    # by simple matrix multiplication
                    transform[t, r, c] = reformat_affine(compose_affine(channel_transform[t, c], round_transform[t, r]),
                                                         z_scale)

    nbp.transform = transform

    return nbp


# From this point, we only have viewers.


def view_overlay(base_image, target_image, transform):
    """
    Function to overlay 2 images in napari.
    Args:
        base_image: base image
        target_image: target image
        transform: 3 x 4 array
    """

    def napari_viewer():
        from magicgui import magicgui
        prevlayer = None

        @magicgui(layout="vertical", auto_call=True,
                  x_offset={'max': 10000, 'min': -10000, 'step': 1, 'adaptive_step': False},
                  y_offset={'max': 10000, 'min': -10000, 'step': 1, 'adaptive_step': False},
                  z_offset={'max': 10000, 'min': -10000, 'step': 1, 'adaptive_step': False},
                  x_scale={'max': 5, 'min': .2, 'value': 1.0, 'adaptive_step': False, 'step': .001},
                  y_scale={'max': 5, 'min': .2, 'value': 1.0, 'adaptive_step': False, 'step': .001},
                  z_scale={'max': 5, 'min': .2, 'value': 1.0, 'adaptive_step': False, 'step': .1},
                  x_rotate={'max': 180, 'min': -180, 'step': 1.0, 'adaptive_step': False},
                  y_rotate={'max': 180, 'min': -180, 'step': 1.0, 'adaptive_step': False},
                  z_rotate={'max': 180, 'min': -180, 'step': 1.0, 'adaptive_step': False},
                  )
        def _napari_extension_move_points(layer: napari.layers.Layer, x_offset: float, y_offset: float, z_offset: float,
                                          x_scale: float, y_scale: float, z_scale: float, x_rotate: float,
                                          y_rotate: float, z_rotate: float, use_defaults: bool) -> None:
            """Add, subtracts, multiplies, or divides to image layers with equal shape."""
            nonlocal prevlayer
            if not hasattr(layer, "_true_rotate"):
                layer._true_rotate = [0, 0, 0]
                layer._true_translate = [0, 0, 0]
                layer._true_scale = [1, 1, 1]
            if prevlayer != layer:
                prevlayer = layer
                on_layer_change(layer)
                return
            if use_defaults:
                z_offset = y_offset = x_offset = 0
                z_scale = y_scale = x_scale = 1
                z_rotate = y_rotate = x_rotate = 0
            layer._true_rotate = [z_rotate, y_rotate, x_rotate]
            layer._true_translate = [z_offset, y_offset, x_offset]
            layer._true_scale = [z_scale, y_scale, x_scale]
            layer.affine = napari.utils.transforms.Affine(rotate=layer._true_rotate, scale=layer._true_scale,
                                                          translate=layer._true_translate)
            layer.refresh()

        def on_layer_change(layer):
            e = _napari_extension_move_points
            widgets = [e.z_scale, e.y_scale, e.x_scale, e.z_offset, e.y_offset, e.x_offset, e.z_rotate, e.y_rotate,
                       e.x_rotate]
            for w in widgets:
                w.changed.pause()
            e.z_scale.value, e.y_scale.value, e.x_scale.value = layer._true_scale
            e.z_offset.value, e.y_offset.value, e.x_offset.value = layer._true_translate
            e.z_rotate.value, e.y_rotate.value, e.x_rotate.value = layer._true_rotate
            for w in widgets:
                w.changed.resume()
            print("Called change layer")

        # _napari_extension_move_points.layer.changed.connect(on_layer_change)
        v = napari.Viewer()
        # add our new magicgui widget to the viewer
        v.window.add_dock_widget(_napari_extension_move_points, area="right")
        v.axes.visible = True
        return v
    # Create a napari viewer and add and transform first image and overlay this with second image
    viewer = napari_viewer()
    viewer.add_image(base_image, blending='additive', colormap='bop_red',
                     affine=np.vstack((transform, np.array([0, 0, 0, 1]))),
                     contrast_limits=(0, 0.3 * np.max(base_image)))
    viewer.add_image(target_image, blending='additive', colormap='bop_red',
                     contrast_limits=(0, 0.3 * np.max(target_image)))


def view_round_regression(shift, position, tile, round, num_pairs, boundary_erosion, image_dims, dist_thresh, resolution):
    """
    Wrapper function for above regression on rounds with viewer set to true

    Args:
        shift: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of shifts in zyx format
        position: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of positions in zyx format
        tile: tile under consideration
        round: round under consideration
        num_pairs: Number of pairs to consider. Must be in range [0, num_samples choose 2]
        boundary_erosion: 3 x 1 array of z, y, x boundary erosion terms
        image_dims: 3 x 1 array of z, y, x image dims
        dist_thresh 3 x 1 array of distance thresholds of z, y and x points that pairs of points must be apart if
        we use them in the algorithm
        Resolution: number of bins in histogram whose mode we return (range = median +/- iqr)

    """

    find_affine_transform_robust_custom(shift[tile, round], position[tile, round], num_pairs, boundary_erosion,
                                        image_dims, dist_thresh, resolution, view=True)


def view_channel_regression(shift, position, tile, channel, num_pairs, boundary_erosion, image_dims, dist_thresh,
                          resolution):
    """
    Wrapper function for above regression on channels with viewer set to true

    Args:
        shift: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of shifts in zyx format
        position: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of positions in zyx format
        tile: tile under consideration
        channel: channel under consideration
        num_pairs: Number of pairs to consider. Must be in range [0, num_samples choose 2]
        boundary_erosion: 3 x 1 array of z, y, x boundary erosion terms
        image_dims: 3 x 1 array of z, y, x image dims
        dist_thresh 3 x 1 array of distance thresholds of z, y and x points that pairs of points must be apart if
        we use them in the algorithm
        Resolution: number of bins in histogram whose mode we return (range = median +/- iqr)

    """

    find_affine_transform_robust_custom(shift[tile, channel], position[tile, channel], num_pairs, boundary_erosion,
                                        image_dims, dist_thresh, resolution, view=True)


def view_regression_scatter(shift, position, transform):
    """
    view 3 scatter plots for each data set shift vs positions
    Args:
        shift: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of shifts in zyx format
        position: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of positions in zyx format
        transform: 3 x 4 affine transform obtained by previous robust regression
    """

    shift = shift.reshape((shift.shape[0] * shift.shape[1] * shift.shape[2], 3)).T
    position = position.reshape((position.shape[0] * position.shape[1] * position.shape[2], 3)).T

    z_range = np.arange(np.min(position[0]), np.max(position[0]))
    yx_range = np.arange(np.min(position[1]), np.max(position[1]))

    plt.subplot(1, 3, 1)
    plt.scatter(position[0], shift[0], alpha=1e3/shift.shape[1])
    plt.plot(z_range, (transform[0, 0] - 1) * z_range + transform[0,3])
    plt.title('Z-Shifts vs Z-Positions')

    plt.subplot(1, 3, 2)
    plt.scatter(position[1], shift[1], alpha=1e3 / shift.shape[1])
    plt.plot(yx_range, (transform[1, 1] - 1) * yx_range + transform[1, 3])
    plt.title('Y-Shifts vs Y-Positions')

    plt.subplot(1, 3, 3)
    plt.scatter(position[2], shift[2], alpha=1e3 / shift.shape[1])
    plt.plot(yx_range, (transform[2, 2] - 1) * yx_range + transform[2, 3])
    plt.title('X-Shifts vs X-Positions')

    plt.show()


def vector_field_plotter(shift, position, transform, eps):
    """
    Function to plot a simple vector field of shifts, scaled up, if they are not outliers, as a function of position.

    Args:
        shift: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of shifts in zyx format
        position: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of positions in zyx format
        transform: 3 x 4 transform as given by registration algorithm
        eps: float > 0 defining the multiplying factor for the error threshold. error threshold = eps * iqr(shift)

    """

    # To attempt to get rid of outliers we'll just say that a point is an inlier if its shift lies in the prediction
    # range between the 2 possible values of the scale and intercept
    z_subvols, y_subvols, x_subvols = shift.shape[:3]
    inlier = np.zeros((z_subvols, y_subvols, x_subvols), dtype=bool)
    thresh = eps * stats.iqr(shift)

    # Set outlier shifts to 0
    for z in range(z_subvols):
        for y in range(y_subvols):
            for x in range(x_subvols):
                in_range = (transform @ np.pad(position[z,y,x], (0,1)) > position[z,y,x] + shift[z, y, x] - thresh * np.ones(3)) * \
                           (transform @ np.pad(position[z,y,x], (0,1)) < position[z,y,x] + shift[z, y, x] + thresh * np.ones(3))
                inlier[z, y, x] = all(in_range)
                if not inlier[z, y, x]:
                    shift[z, y, x] = np.array([0, 0, 0])
                else:
                    # Make the direction data for the arrows. As we'd like to see how the shifts vary,
                    # subtract the actual shift
                    shift[z, y, x] = shift[z, y, x] - transform[:, 3]
    # Now make the 3D viewer.
    ax = plt.figure().add_subplot(projection='3d')
    # Make the grid
    z, y, x = np.meshgrid(np.arange(y_subvols), np.arange(z_subvols), np.arange(x_subvols))
    u, v, w = shift[:,:, :, 0], shift[:,:, :, 1], shift[:,:, :, 2]
    q = ax.quiver(x, y, z, u, v, w, length=0.1, cmap='Reds', normalize=True)
    q.set_array(np.random.rand(np.prod(x.shape)))

    plt.show()


# # anchor_image = sobel(np.load('/Users/reillytilbury/Desktop/Tile 0/round_Cp_anchor_t0c18.npy'))
# # r0c18_image = sobel(np.load('/Users/reillytilbury/Desktop/Tile 0/round_Cp_r0_t0c18.npy'))
#
# # next we generate random cuboids from each of these images and obtain our shift arrays
# # shift, position = find_random_shift_array(  anchor_image, r0c18_image, 500, 10, 200, 200)
#
# # Use these subvolumes shifts to find the affine transform taking the volume (t, r_ref, c_ref) to
# # (t, r, c_ref)
# # transform = find_affine_transform_robust(shift, position)
#
# Load in notebook
# nb = Notebook("//128.40.224.65\SoyonHong\Christina Maat\ISS Data + Analysis\E-2210-001 6mo CP vs Dry\CP/v6_1tile_28NOV22\output/notebook_1tile.npz",
#               "C:/Users\Reilly\Desktop\E-2210-001_CP_1tile_v6_settings.ini")

# # Run first with no histogram matching
# transform, round_transform, channel_transform, round_shift, round_position, channel_shift, channel_position = \
#     register(nb.basic_info, nb.file_names, False)
# np.save('C:/Users/Reilly/Desktop/Diagnostics/No Hist Matching/channel_position.npy', np.array(channel_position))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/No Hist Matching/channel_shift.npy', np.array(channel_shift))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/No Hist Matching/channel_transform.npy', np.array(channel_transform))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/No Hist Matching/round_position.npy', np.array(round_position))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/No Hist Matching/round_shift.npy', np.array(round_shift))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/No Hist Matching/round_transform.npy', np.array(round_transform))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/No Hist Matching/transform.npy', np.array(transform))
#
# Run second with histogram matching
# transform, round_transform, channel_transform, round_shift, round_position, channel_shift, channel_position = \
#     register(nb.basic_info, nb.file_names, True)
# np.save('C:/Users/Reilly/Desktop/Diagnostics/Custom/channel_position.npy', np.array(channel_position))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/Custom/channel_shift.npy', np.array(channel_shift))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/Custom/channel_transform.npy', np.array(channel_transform))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/Custom/round_position.npy', np.array(round_position))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/Custom/round_shift.npy', np.array(round_shift))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/Custom/round_transform.npy', np.array(round_transform))
# np.save('C:/Users/Reilly/Desktop/Diagnostics/Custom/transform.npy', np.array(transform))

# # anchor_image = sobel(np.load('/Users/reillytilbury/Desktop/Tile 0/round_Cp_anchor_t0c18.npy'))
# # target_image = sobel(np.load('/Users/reillytilbury/Desktop/Tile 0/round_Cp_r0_t0c5.npy'))
# # anchor_subvolume, pos = split_3d_image(anchor_image, 5, 15, 15, 12, 300, 300)
# # target_subvolume, _ = split_3d_image(target_image, 5, 15, 15, 12, 300, 300)
# # shift = find_shift_array(subvol_base=anchor_subvolume, subvol_target=target_subvolume)
# # find_affine_transform_robust_custom(shift, pos, 1e5, [5,200,200], [64,2304,2304], [5,200,200], 100)
# # shift = np.zeros((5, 4, 4, 3))
# # scale = np.zeros((5, 4, 4, 3))
# # sample = [100, 200, 300, 500, 1000]
# # z_box = [5, 8, 10, 15]
# # xy_box = [20, 50, 100, 200]
# #
# # for i in range(5):
# #     for j in range(4):
# #         for k in range(4):
# #             plt.clf()
# #             scale[i, j, k], shift[i, j, k] = random_cuboids(sample[i], r0c18_image, anchor_image,
# #                                                             z_box[j],xy_box[k],xy_box[k], 50/sample[i],custom=True)
# #             print(sample[i],z_box[j],xy_box[k])
# #
# # np.save('/Users/reillytilbury/Desktop/Tile 0/Registration Images/shift.npy', shift)
# # np.save('/Users/reillytilbury/Desktop/Tile 0/Registration Images/scale.npy', scale)
# #
# # print('Hello')