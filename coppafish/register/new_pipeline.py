import numpy as np
from skimage.registration import phase_cross_correlation
from coppafish.setup import NotebookPage
from coppafish.utils.npy import load_tile

def split_3d_image(image, y_subvolumes, x_subvolumes, z_subvolumes):
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
    y_size = int(image.shape[0] / y_subvolumes)
    x_size = int(image.shape[1] / x_subvolumes)
    z_size = int(image.shape[2] / z_subvolumes)

    # Create an array to store the subvolumes in
    subvolumes = np.zeros((y_subvolumes, x_subvolumes, z_subvolumes, y_size, x_size, z_size))
    positions = np.zeros((y_subvolumes, x_subvolumes, z_subvolumes, 3))

    # Split the image into subvolumes and store them in the array
    for y in range(y_subvolumes):
        for x in range(x_subvolumes):
            for z in range(z_subvolumes):
                subvolumes[y, x, z] = image[y * y_size:(y + 1) * y_size, x * x_size:(x + 1) * x_size, z * z_size:(z + 1) * z_size]
                positions[y, x, z] = [(y + 1/2) * y_size, (x + 1/2) * x_size, (z + 1/2) * z_size]

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

    y_subvolumes, x_subvolumes, z_subvolumes = subvol_base.shape[0], subvol_base.shape[1], subvol_base.shape[2]
    shift = np.zeros((y_subvolumes, x_subvolumes, z_subvolumes, 3))

    for y in range(y_subvolumes):
        for x in range(x_subvolumes):
            for z in range(z_subvolumes):
                shift[y, x, z], _, _ = phase_cross_correlation(subvol_target[y, x, z], subvol_base[y, x, z],
                                                               upsample_factor=10)

    return shift


def find_affine_transform(shift, position):
    """
    Function which finds the best affine transform taking coords position[y,x,z] to position[y,x,z] + shift[y,x,z] for
    all subvolumes [y,x,z].
    Args:
        shift:(y_subvolumes * x_subvolumes * z_subvolumes * 3) array of shifts
        position: (y_subvolumes * x_subvolumes * z_subvolumes * 3) array of positions of base points

    Returns:
        transform: (3 x 4) Best affine transform fitting the transform.
    """

    # First, pad the position array
    position = np.pad(position, ((0,0),(0,1)), constant_values=1)

    # Now, get rid of outliers in the shifts
    shift_abs = np.linalg.norm(shift, axis=1)
    use = (shift_abs > np.mean(shift_abs) - np.std(shift_abs)) * (shift_abs < np.mean(shift_abs) + np.std(shift_abs))

    # Complete regression on shifts that we intend to use
    position = position[use]
    shift = shift[use]
    omega = np.linalg.lstsq(position.T, shift.T)

    # Compute the transform from the regression matrix
    transform = omega.T + np.pad(np.eye(3), ((0,0),(0,1)))

    return transform


def register(nbp_basic: NotebookPage, nbp_file: NotebookPage):
    """
    Registration pipeline. Returns register Notebook Page which right now, just contains affine transform array and
    subvolume shift arrays.

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

    # Find the inter round transforms and inter channel transforms
    for t in use_tiles:
        # Load in the anchor npy volume, only need to do this once per tile
        anchor_image = load_tile(nbp_file, nbp_basic, t, r_ref, c_ref)
        for r in use_rounds:
            # Load in imaging npy volume
            target_image = load_tile(nbp_file, nbp_basic, t, r, c_ref)

            # Split up the anchor and target into subvolumes. As we're not trying to calculate chromatic aberration, we
            # don't need many yx sub_volumes
            anchor_subvolume, position = split_3d_image(anchor_image, 4, 4, 3)
            target_subvolume, _ = split_3d_image(target_image, 4, 4, 3)

            # Find the best shifts from each of these subvolumes to their corresponding subvolume
            shift = find_shift_array(anchor_subvolume, target_subvolume)

            # Use these subvolumes shifts to find the affine transform taking the volume (t, r_ref, c_ref) to
            # (t, r, c_ref)
            round_transform[t, r] = find_affine_transform(shift, position)

        # Begin the channel transformation calculations. This requires us to use a central round.
        r = n_rounds // 2
        for c in use_channels:
            # Load in imaging npy volume
            target_image = load_tile(nbp_file, nbp_basic, t, r, c_ref)

            # Split up the anchor and target into subvolumes. As we're trying to calculate chromatic aberration,
            # increase the yx precision now by adding more subvolumes
            anchor_subvolume, position = split_3d_image(anchor_image, 10, 10, 3)
            target_subvolume, _ = split_3d_image(target_image, 10, 10, 3)

            # Find the best shifts from each of these subvolumes to their corresponding subvolume
            shift = find_shift_array(anchor_subvolume, target_subvolume)

            # Use these subvolumes shifts to find the affine transform taking the volume (t, r, c_ref) to
            # (t, r, c)
            channel_transform[t, c] = find_affine_transform(shift, position)

        # Combine all transforms
        for r in use_rounds:
            for c in use_channels:
                # Compose the linear part of both of these first
                transform[t, r, c, :, :3] = channel_transform[t, c, :, :3] @ round_transform[r, c, :, :3]
                # Compose the shifts next
                transform[t, r, c, :, 4] = channel_transform[t, c, :, 4] + round_transform[r, c, :, 4]
        