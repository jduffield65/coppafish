import numpy as np
from skimage import filters
from skimage import registration

from coppafish import utils
from coppafish.setup import NotebookPage


def register_cameras(nbp_basic: NotebookPage, nbp_file: NotebookPage, config: dict):
    """
    Script to align cameras, as the initial registration assumes these are all aligned. We find and return a shift
    from the shift camera to each other camera. Since the other registration finds the shift from the anchor cam to
    the shift cam + the true shift, adding these gives shift from anchor cam to all cams + true shift.

    Args:
        nbp_basic: basic info page of Notebook we're performing registration on (Notebook Page)
        nbp_file: file names page of Notebook we're performing registration on (Notebook Page)
        config: dictionary containing register_initial info (dict)

    Returns:
        shift: (n_tiles x n_rounds x n_channels x 3) array containing camera shifts from cam[shift_channel] to cam[c]
        for each round c in use. This is then just copied across n_tiles and n_rounds.
    """

    # Initialise recurring variables
    cam = nbp_basic.channel_camera
    laser = nbp_basic.channel_laser
    shift_channel = config['shift_channel']
    if shift_channel is None:
        shift_channel = nbp_basic.anchor_channel
    sample_tile = min(nbp_basic.use_tiles)
    sample_round = min([i for i in nbp_basic.use_rounds if i > 0])
    # Load and filter anchor image
    shift_channel_image = \
        filters.sobel(utils.nd2.get_raw_images(nbp_basic, nbp_file, [sample_tile], [sample_round], 
                                               [shift_channel], list(np.arange(nbp_basic.nz // 2 - 10,
                                               nbp_basic.nz // 2 + 10)))[0, 0, 0])
    # Initialise the variable that we will return
    shift = np.zeros((nbp_basic.n_channels, 3), dtype=int)

    # The body of the code starts here. We will be finding the shifts taking cam[shift_channel] to cam[c] for all c in
    # use.

    # Create a list of cameras that are in use, get rid of duplicates and as we're finding shits from shift_channel_cam
    # to all other cams, don't need to include shift_cam
    use_cams = list(set([cam[i] for i in nbp_basic.use_channels]).symmetric_difference(set([cam[shift_channel]])))

    # Now we will pick a representative channel for each camera:
    # Create an empty list for the sample channels, we'll populate this in the next for loop
    sample_channels = []
    # For each camera we choose a channel which will give us a sample image.
    # According to ground truth data, the best images are those where laser and camera freqs are similar
    for i in range(len(use_cams)):
        # potential_channels is just the preimage of the camera function intersected with use_channels
        potential_channels = [j for j in nbp_basic.use_channels if cam[j] == use_cams[i]]
        # The channel we choose will minimise |laser_freq - camera_freq|
        delta_freq = [abs(cam[j]-laser[j]) for j in potential_channels]
        sample_channels.append(potential_channels[np.argmin(delta_freq)])

    # Load in all sample volumes to an array called sample_image. Sample_image[i] refers to the camera use_cams[i]
    sample_image = utils.nd2.get_raw_images(nbp_basic, nbp_file, [sample_tile], [sample_round], sample_channels,
                                            list(np.arange(nbp_basic.nz//2 - 10, nbp_basic.nz//2 + 10)))[0, 0]

    # Initialise filtered_sample_image
    filtered_sample_image = np.zeros(sample_image.shape)
    # Filter each image with a Sobel filter to improve registration
    for i in range(sample_image.shape[0]):
        filtered_sample_image[i] = filters.sobel(sample_image[i])
    # Delete sample_image to save memory
    del sample_image

    # Store the cam_shifts in this array, which will then be used to populate cam shifts by channel
    cam_shift = np.zeros((len(use_cams), 3), dtype=int)

    # Now for each sample image, detect the shift taking cam[shift_channel] to cam[c]
    for i in range(filtered_sample_image.shape[0]):
        cam_shift[i], _, _ = registration.phase_cross_correlation(filtered_sample_image[i], shift_channel_image)

    # Final part is to populate the shift array. Loop through all channels in use and update them accordingly
    for c in nbp_basic.use_channels:
        # Nothing to do for channels from shift_channel_cam, as there is no shift to itself
        if cam[c] != cam[shift_channel]:
            # First get the index of the camera of channel c, as listed in use_cams
            camera_index = use_cams.index(cam[c])
            # Now to find the shift from anchor cam to channel c cam, we add the shift from anchor cam to shift cam, 
            # and then from shift cam to channel c cam
            shift[c] = cam_shift[camera_index]

    # Reformat shift array from n_channels x 3 to n_tiles x n_rounds x n_channels x 3 for consistency with other code
    # First copy the array for each round
    shift = np.repeat(shift[np.newaxis, :, :], nbp_basic.n_rounds, axis=0)
    # Now copy the array for each tile
    shift = np.repeat(shift[np.newaxis, :, :, :], nbp_basic.n_tiles, axis=0)

    return shift


# nb = Notebook('C:/Users/Reilly/Desktop/Sample Notebooks/Izzie/notebook_single_tile.npz')
# shift = register_cameras(nb.basic_info, nb.file_names, nb.get_config()['register_initial'])
