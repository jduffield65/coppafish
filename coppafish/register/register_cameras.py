import numpy as np
from coppafish import utils
from coppafish.setup import NotebookPage
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel


def register_cameras(nbp_basic: NotebookPage, nbp_file: NotebookPage, config: dict):
    """
    Initial shifts are computed based on a single shift_channel. This channel can be found in
    nb.register_initial.shift_channel. When the cameras are offset from one another, the initial shift should be
    different for channels inherited from different cameras than that belonging to the shift_channel.
    For this reason, we find shifts between cameras too.

    Args:
        nbp_basic: basic info page of Notebook we're performing registration on (Notebook Page)
        nbp_file: file names page of Notebook we're performing registration on (Notebook Page)
        config: dictionary containing register_initial info (dict)

    Returns:
        shift: (n_tiles x n_rounds x n_channels x 3) array containing camera shifts from the anchor channel to each
        channel's camera (np.ndarray)
    """

    # Initialise recurring variables
    cam = nbp_basic.channel_camera
    laser = nbp_basic.channel_laser
    anchor_channel = nbp_basic.anchor_channel
    shift_channel = config['shift_channel']
    ac_sc_shift = [0, 0, 0]
    sample_tile = min(nbp_basic.use_tiles)
    sample_round = min([i for i in nbp_basic.use_rounds if i > 0])
    # Load and filter anchor image
    anchor_image = sobel(utils.nd2.get_raw_images(nbp_basic, nbp_file, [sample_tile], [sample_round],
                                            [anchor_channel], list(np.arange(nbp_basic.nz // 2 - 10,
                                                                             nbp_basic.nz // 2 + 10)))[0, 0, 0])
    # Initialise the variable that we will return as well
    shift = np.zeros((nbp_basic.n_channels, 3), dtype=int)

    # Program works in 2 stages, first we find the camera shift from cam[anchor_channel] to the cam[shift_channel],
    # then we find the shift between cam[shift_channel] and all cameras in cam[use_channels]

    # Stage 1:
    # First check if there is a shift_channel specified. If not, then it will be the same as the anchor so stage 1
    # can be averted, and we will use the value [0, 0, 0] for ac_sc_shift
    if shift_channel is not None:
        # If a shift channel has been specified and does not derive from the anchor camera, then we must compute cam
        # shifts
        if cam[shift_channel] != cam[anchor_channel]:
            # Now we find the shift between cam[shift_channel] and cam[anchor_channel]
            # Load and filter shift channel image
            shift_channel_image = sobel(utils.nd2.get_raw_images(nbp_basic, nbp_file, [sample_tile], [sample_round],
                                                                 [shift_channel], list(np.arange(nbp_basic.nz // 2 - 10,
                                                                 nbp_basic.nz // 2 + 10)))[0, 0, 0])
            # compute this shift by using a phase cross correlation algorithm
            ac_sc_shift, _, _ = phase_cross_correlation(shift_channel_image, anchor_image)
        else:
            shift_channel_image = anchor_image
    else:
        shift_channel_image = anchor_image

    # Stage 2:
    # Now we find the shifts from shift_channel_camera to each camera
    # Create a set of cameras that are in use
    use_cams = list(set([cam[i] for i in nbp_basic.use_channels]))

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

    # Load in all volumes to an array called sample_image. Sample_image[i] refers to the camera non_sc_cam[i]
    sample_image = utils.nd2.get_raw_images(nbp_basic, nbp_file, [sample_tile], [sample_round], sample_channels,
                                            list(np.arange(nbp_basic.nz//2 - 10, nbp_basic.nz//2 + 10)))[0, 0]

    # Initialise filtered_sample_image
    filtered_sample_image = np.zeros(sample_image.shape)
    # Filter each image with a Sobel filter to improve registration
    for i in range(sample_image.shape[0]):
        filtered_sample_image[i] = sobel(sample_image[i])
    # Delete sample_image to save memory
    del sample_image

    # Store the shifts of these sample images in an array called sample_image_shift. Row i corresponds to the shift from
    # sc_cam to use_cams[i]
    sample_image_shift = np.zeros((len(use_cams), 3), dtype=int)

    # Now for each sample image, detect the shift taking cam[shift_channel] to cam[c]
    for i in range(filtered_sample_image.shape[0]):
        sample_image_shift[i], _, _ = phase_cross_correlation(filtered_sample_image[i], shift_channel_image)

    # Final part of stage 2 is to populate the shift array. Loop through all channels in use and update them accordingly
    for c in nbp_basic.use_channels:
        # First get the index of the camera of channel c, as listed in use_cams
        camera_index = use_cams.index(cam[c])
        # Now to find the shift from anchor cam to channel c cam, we add the shift from anchor cam to shift cam, and
        # then from shift cam to channel c cam
        shift[c] = ac_sc_shift + sample_image_shift[camera_index]

    # Reformat shift array from n_channels x 3 to n_tiles x n_rounds x n_channels x 3 for consistency with other code
    # First copy the array for each round
    shift = np.repeat(shift[np.newaxis, :, :], nbp_basic.n_rounds, axis=0)
    # Now copy the array for each tile
    shift = np.repeat(shift[np.newaxis, :, :, :], nbp_basic.n_tiles, axis=0)

    return shift
