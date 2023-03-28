import numpy as np


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