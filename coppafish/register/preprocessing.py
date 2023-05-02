import os
import pickle
import numpy as np
from coppafish.setup import NotebookPage


def load_reg_data(nbp_file: NotebookPage, nbp_basic: NotebookPage, config: dict):
    """
    Function to load in pkl file of previously obtained registration data if it exists.
    Args:
        nbp_file: File Names notebook page
        nbp_basic: Basic info notebook page
        config: register page of config dictionary
    Returns:
        registration_data: dictionary with the following keys
        * tiles_completed (list)
        * position ( (z_subvols x y subvols x x_subvols) x 3 ) ndarray
        * round_shift ( n_tiles x n_rounds x (z_subvols x y subvols x x_subvols) x 3 ) ndarray
        * channel_shift ( n_tiles x n_channels x (z_subvols x y subvols x x_subvols) x 3 ) ndarray
        * round_transform (n_tiles x n_rounds x 3 x 4) ndarray
        * channel_transform (n_tiles x n_channels x 3 x 4) ndarray
        * round_shift_corr ( n_tiles x n_rounds x (z_subvols x y subvols x x_subvols) ) ndarray
        * channel_shift_corr ( n_tiles x n_channels x (z_subvols x y subvols x x_subvols) ) ndarray
    """
    # Check if the registration data file exists
    if os.path.isfile(os.path.join(nbp_file.output_dir, 'registration_data.pkl')):
        with open(os.path.join(nbp_file.output_dir, 'registration_data.pkl'), 'rb') as f:
            registration_data = pickle.load(f)
    else:
        n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
        z_subvols, y_subvols, x_subvols = config['z_subvols'], config['y_subvols'], config['x_subvols']
        registration_data = {'tiles_completed': [],
                             'position': np.zeros((z_subvols * y_subvols * x_subvols, 3)),
                             'round_shift': np.zeros((n_tiles, n_rounds, z_subvols * y_subvols * x_subvols, 3)),
                             'channel_shift': np.zeros((n_tiles, n_channels, z_subvols * y_subvols * x_subvols, 3)),
                             'round_transform': np.zeros((n_tiles, n_rounds, 3, 4)),
                             'channel_transform': np.zeros((n_tiles, n_channels, 3, 4)),
                             'round_shift_corr': np.zeros((n_tiles, n_rounds, z_subvols * y_subvols * x_subvols)),
                             'channel_shift_corr': np.zeros((n_tiles, n_channels, z_subvols * y_subvols * x_subvols))
                             }
    return registration_data


def save_compressed_image(nbp_file: NotebookPage, image: np.ndarray, t: int, r: int, c: int):
    """
    Save low quality cropped images for reg diagnostics

    Args:
        nbp_file: file_names notebook page
        image: zyx image to be saved in compressed form
        t: tile
        r: round
        c: channel

    Returns:
        N/A
    """

    # Check directory exists otherwise create it
    if not os.path.isdir(os.path.join(nbp_file.output_dir, 'reg_images')):
        os.makedirs(os.path.join(nbp_file.output_dir, 'reg_images'))

    mid_z, mid_y, mid_x = image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2
    # save a small subset for reg diagnostics
    np.save(os.path.join(nbp_file.output_dir, 'reg_images/') + 't' + str(t) + 'r' + str(r) + 'c' + str(c),
            (256 * image / np.max(image)).astype(np.uint8)
            [mid_z - 5: mid_z + 5, mid_y - 250: mid_y + 250, mid_x - 250: mid_x + 250])


def replace_scale(transform: np.ndarray, scale: np.ndarray):
    """
    Function to replace the diagonal of transform with new scales
    Args:
        transform: n_tiles x n_rounds x 3 x 4 or n_tiles x n_channels x 3 x 4 of zyx affine transforms
        scale: n_tiles x n_rounds x 3 or n_tiles x n_channels x 3 of zyx scales

    Returns:
        transform: n_tiles x n_rounds x 3 x 4 or n_tiles x n_channels x 3 x 4 of zyx affine transforms
    """
    # Loop through dimensions i: z = 0, y = 1, x = 2
    for i in range(3):
        transform[:, :, i, i] = scale[:, :, i]

    return transform


def populate_full(sublist_1, list_1, sublist_2, list_2, array):
    """
    Function to convert array from len(sublist1) x len(sublist2) to len(list1) x len(list2), listting elems not in sublists
    as 0
    Args:
        sublist_1: sublist in the 0th dim
        list_1: entire list in 0th dim
        sublist_2: sublist in the 1st dim
        list_2: entire list in 1st dim
        array: array to be converted

    Returns:
        full_array: len(list1) x len(list2) ndarray
    """
    full_array = np.zeros((len(list_1), len(list_2)))
    for i in range(len(sublist_1)):
        for j in range(len(sublist_2)):
            full_array[sublist_1[i], sublist_2[j]] = array[i, j]
    return full_array


def create_shift_images(shift, tilepos_yx):
    """
    function to create images where shift is in the position specified by tile position.
    Args:
        shift: n_tiles_use x 3 zyx shift
        tilepos_yx: n_tile_use x 2 yx position of tiles

    Returns:
        shift_im: 3 x n_rows x n_cols where first axis specifies z, y, x respectively and the rest are images
    """
    # Initialise images
    n_rows = np.max(tilepos_yx[:, 0]) - np.min(tilepos_yx[:, 0]) + 1
    n_cols = np.max(tilepos_yx[:, 1]) - np.min(tilepos_yx[:, 1]) + 1
    n_tiles_use = tilepos_yx.shape[0]
    im = np.zeros((3, n_rows, n_cols)) * np.nan

    # Create images. These will have first axis referred to as y and the next as x. This is consistent with how
    # matplotlib plots things so will look correct
    for t in range(n_tiles_use):
        im[:, tilepos_yx[0, t], tilepos_yx[1, t]] = shift[t]

    return im


def stack_images(im1, im2):
    # First stack vertically
    n_cols = im1.shape[1]
    nan_bar = np.zeros(n_cols) * np.nan
    im_stack = np.vstack((nan_bar, im1, nan_bar, im2, nan_bar))

    # Now add horizontal borders
    n_rows = im_stack.shape[0]
    nan_bar = np.zeros(n_rows) * np.nan
    im_stack = np.vstack((nan_bar, im_stack.T, nan_bar)).T

    return im_stack


def yxz_to_zyx(image: np.ndarray):
    """
    Function to convert image from yxz to zyx
    Args:
        image: yxz image

    Returns:
        image_new: zyx image
    """
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return image


def n_matches_to_frac_matches(nbp_basic: NotebookPage, n_matches: np.ndarray, spot_no: np.ndarray):
    """
    Function to convert n_matches to fraction of matches
    Args:
        nbp_basic: basic info nbp
        n_matches: n_tiles x n_rounds x n_channels x n_iters
        spot_no: n_tiles x (n_rounds + 1) x n_channels

    Returns:
        frac_matches: n_tiles x n_rounds x n_channels x n_iters
    """
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    frac_matches = np.zeros_like(n_matches)

    for t in use_tiles:
        for r in use_rounds:
            for c in use_channels:
                frac_matches[t, r, c] = n_matches[t, r, c] / spot_no[t, r, c]

    return frac_matches


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
        (y_subvolumes * x_subvolumes * z_sub_volumes) x 3 The middle coord of each subtile
    """
    z_image, y_image, x_image = image.shape

    # Allow 0.5 of a box either side and then split the middle with subvols evenly spaced points, ie into subvols - 1
    # intervals. Then use integer division. e.g actual unit distance is 12.5, this gives a unit distance of 12 so
    # should never overshoot
    if z_subvolumes > 1:
        z_unit = (z_image - z_box) // (z_subvolumes - 1)
    else:
        z_unit = 0
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

    # Reshape the position array
    position = np.reshape(position, (z_subvolumes * y_subvolumes * x_subvolumes, 3))

    return subvolume, position


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


def yxz_to_zyx_affine(A, z_scale):
    """
        Function to convert 4 x 3 matrix in y, x, z coords into a 3 x 4 matrix of z, y, x coords and rescale the
        z-shift

        Args:
            A: Original transform in old format (4 x 3)
            z_scale: How much to unscale z-components by (float)

        Returns:
            A_reformatted: 3 x 4 transform with associated changes
            """

    # Append a bottom row to A
    A = np.vstack((A.T, np.array([0, 0, 0, 1])))

    # First, convert everything into z, y, x by multiplying by a matrix that swaps rows
    row_shuffler = np.zeros((4, 4))
    row_shuffler[0, 2] = 1
    row_shuffler[1, 0] = 1
    row_shuffler[2, 1] = 1
    row_shuffler[3, 3] = 1
    # Invert row shuffler as this was the transform from zyx to yxz. We want to go the other way.
    row_shuffler = np.linalg.inv(row_shuffler)

    A = np.linalg.inv(row_shuffler) @ A @ row_shuffler

    # Next, divide the shift part of A by the expansion factor
    A[2, 3] = A[2, 3] / z_scale

    # Remove the final row
    A = A[:3, :4]

    return A


def zyx_to_yxz_affine(A, z_scale):
    """
    Function to convert 3 x 4 matrix in z, y, x coords into a 4 x 3 matrix of y, x, z coords and rescale the shift

    Args:
        A: Original transform in old format (3 x 4)
        z_scale: How much to scale z-components by (float)

    Returns:
        A_reformatted: 4 x 3 transform with associated changes

    """
    # Append to A a bottom row
    A = np.vstack((A, np.array([0, 0, 0, 1])))

    # First, convert everything into z, y, x by multiplying by a matrix that swaps rows
    row_shuffler = np.zeros((4, 4))
    row_shuffler[0, 2] = 1
    row_shuffler[1, 0] = 1
    row_shuffler[2, 1] = 1
    row_shuffler[3, 3] = 1

    # compute the matrix in the new basis
    A = np.linalg.inv(row_shuffler) @ A @ row_shuffler

    # Next, multiply the shift part of A by the expansion factor
    A[2, 3] = z_scale * A[2, 3]

    # Remove the final row
    A = A[:3, :4]

    # Finally, transpose the matrix
    A = A.T

    return A


def change_basis(A, new_origin, z_scale):
    """
    Takes in 4 x 3 yxz * yxz transform where z coord is in xy pixels and convert to 4 x 4 zyx * zyx. Same as above
    but allows for change in origin.
    # TODO: Replace all cases of reformat affine with change_basis
    Args:
        A: 4 x 3 yxz * yxz transform
        new_origin: new origin (zyx)
        z_scale: pixel_size_z/pixel_size_xy

    """
    # Transform saved as yxz * yxz but needs to be zyx * zyx. Convert this to something napari will understand. I think
    # this includes making the shift the final column as opposed to our convention of making the shift the final row
    affine_transform = np.vstack((A.T, np.array([0, 0, 0, 1])))

    row_shuffler = np.zeros((4, 4))
    row_shuffler[0, 1] = 1
    row_shuffler[1, 2] = 1
    row_shuffler[2, 0] = 1
    row_shuffler[3, 3] = 1

    # Now compute the affine transform, in the new basis
    affine_transform = np.linalg.inv(row_shuffler) @ affine_transform @ row_shuffler

    # z shift needs to be converted to z-pixels as opposed to yx
    affine_transform[0, 3] = affine_transform[0, 3] / z_scale

    # also add new origin conversion for shift
    affine_transform[:3, 3] += (affine_transform[:3, :3] - np.eye(3)) @ new_origin

    return affine_transform


def reformat_array(A, nbp_basic, round):
    """
    Reformatting function to send A from (n_tiles * n_rounds) x z_subvol x x_subvol x y_subvol x 3 array to a new array
    with dimensions n_tiles x n_rounds x z_subvol x x_subvol x y_subvol x 3 and equivalently with channels
    Args:
        A: round, pos arrays for shift and position need to be reformatted
        nbp_basic: basic info page of notebook
        round: boolean option (True if working with round_shifts, False if working with channel_shifts)

    Returns:
        B: Reformatted array
    """

    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
    use_rounds, use_channels = nbp_basic.use_rounds, nbp_basic.use_channels
    c_ref = nbp_basic.anchor_channel
    counter = 0

    if round:
        B = np.zeros((n_tiles, n_rounds, A.shape[1], A.shape[2], A.shape[3], A.shape[4]))
        for t in range(n_tiles):
            for r in use_rounds:
                B[t, r] = A[counter]
                counter += 1
    else:
        # This scenario arises if we don't save the transforms for the anchor channel (identity)
        use_channels.remove(c_ref)
        B = np.zeros((n_tiles, n_channels, A.shape[1], A.shape[2], A.shape[3], A.shape[4]))
        for t in range(n_tiles):
            for c in use_channels:
                B[t, c] = A[counter]
                counter += 1
        use_channels.append(c_ref)

    return B


def custom_shift(array: np.ndarray, offset: np.ndarray, constant_values=0):
    """
    Custom-built function to compute array shifted by a certain offset
    Args:
        array: array to be shifted
        offset: shift value
        constant_values: by default this is 0

    Returns:
        new_array: array shifted by offset with constant value 0
    """
    array = np.asarray(array)
    offset = np.atleast_1d(offset)
    assert len(offset) == array.ndim
    new_array = np.empty_like(array)

    def slice1(o):
        return slice(o, None) if o >= 0 else slice(0, o)

    new_array[tuple(slice1(o) for o in offset)] = (
        array[tuple(slice1(-o) for o in offset)])

    for axis, o in enumerate(offset):
        new_array[(slice(None),) * axis +
                  (slice(0, o) if o >= 0 else slice(o, None),)] = constant_values

    return new_array