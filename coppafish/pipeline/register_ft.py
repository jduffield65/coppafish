import numpy as np
from tqdm import tqdm
from coppafish.setup import NotebookPage, Notebook
from coppafish.utils.raw import load
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation as pcc


def shift(array: np.ndarray, offset: np.ndarray, constant_values=0):
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


def register_ft(nbp_basic: NotebookPage, nbp_file: NotebookPage, nbp_register_initial: NotebookPage):
    """
    Compute inter channel shifts and scales. Do this with a Fourier Shift algorithm.

    Args:
        nbp_basic: (Notebook page) basic info page of the notebook
        nbp_file: (Notebook page) file names page of the notebook
        nbp_register_initial: (Notebook page) register_initial page of the notebook

    Returns:
        nbp: (Notebook page) Register page of the notebook

    """
    # Create a notebook page for the register section
    nbp = NotebookPage("register")
    # Add initial shift
    nbp.initial_shift = nbp_register_initial.shift

    # Initialise commonly used variables
    shift_channel = nbp_register_initial.shift_channel
    use_channels, n_channels = nbp_basic.use_channels, nbp_basic.n_channels
    use_tiles, n_tiles = nbp_basic.use_tiles, nbp_basic.n_tiles
    use_rounds, n_rounds = nbp_basic.use_rounds, nbp_basic.n_rounds
    tile_sz = nbp_basic.tile_sz
    tilepos = nbp_basic.tilepos_yx
    z_scale = nbp_basic.pixel_size_z/nbp_basic.pixel_size_xy
    z_expansion_factor = nbp_register_initial.z_expansion_factor
    initial_shift = nbp_register_initial.shift
    cam_shift = nbp_register_initial.cam_shift.astype(int)
    alpha = 0.3     # This determines how much of the boundary we use in the top, bottom, left and right shift

    # The shifts for each tile and channel, calculated at the top, bottom, left and right will be kept here
    left_shift = np.zeros((n_tiles, n_channels, 3))
    right_shift = np.zeros((n_tiles, n_channels, 3))
    bottom_shift = np.zeros((n_tiles, n_channels, 3))
    top_shift = np.zeros((n_tiles, n_channels, 3))
    shift_correction = np.zeros((n_tiles, n_channels, 3))
    scale_correction = np.ones((n_tiles, n_channels, 2))

    with tqdm(total=n_tiles * len(use_channels)) as pbar:
        pbar.set_description(f"Fourier Registration algorithm for inter channel transforms")
        for t in use_tiles:
            # Since we're computing inter channel shifts, we need a channel to shift from. As the register initial stage
            # shifts the anchor image to the shift channel of each round, and we expect deviations from the shift
            # channel to other channels to be constant across rounds, it makes sense to compute the shifts from the
            # shift channel to the imaging channels.
            # Sobel filter these to improve registration
            reference_image_raw = sobel(load(nbp_file, nbp_basic, r=3, t=t, c=shift_channel))
            for c in use_channels:
                pbar.set_postfix({'tile': f'{t}', 'channel': f'{c}'})
                # Correct for camera offsets so that the shift between channels is not influenced by this systematic
                # shift (this is already accounted for in register initial, and not what this stage is trying to fix)
                channel_image_raw = shift(sobel(load(nbp_file, nbp_basic, r=3, t=t, c=c)), -cam_shift[t, 3, c])
                # Now we'll do the registration on each tile. We do the registration on each end of the image in x and
                # in y. Finding the difference in shifts will allow us to determine the scale as well!
                left_shift[t, c], _, _ = pcc(channel_image_raw[:, :int(tile_sz * alpha)],
                                       reference_image_raw[:, :int(tile_sz * alpha)], upsample_factor=10)
                right_shift[t, c], _, _ = pcc(channel_image_raw[:, int(tile_sz * (1 - alpha)):],
                                        reference_image_raw[:, int(tile_sz * (1 - alpha)):], upsample_factor=10)
                bottom_shift[t, c], _, _ = pcc(channel_image_raw[:int(tile_sz * alpha), :],
                                         reference_image_raw[:int(tile_sz * alpha), :], upsample_factor=10)
                top_shift[t, c], _, _ = pcc(channel_image_raw[int(tile_sz * (1 - alpha)):, :],
                                      reference_image_raw[int(tile_sz * (1 - alpha)):, :], upsample_factor=10)
                print('Top Shift: ', top_shift, '\n', 'Bottom Shift: ', bottom_shift, '\n', 'Left Shift: ', left_shift,
                      '\n', 'Right Shift: ', right_shift)
                # Use these to find scales
                scale_correction[t, c, 0] = 1 + (top_shift[t, c, 0] - bottom_shift[t, c, 0]) / ((1 - alpha) * tile_sz)
                scale_correction[t, c, 1] = 1 + (right_shift[t, c, 1] - left_shift[t, c, 1]) / ((1 - alpha) * tile_sz)
                # Use to find shift
                shift_correction[t, c] = np.median(np.vstack((left_shift[t, c], right_shift[t, c], bottom_shift[t, c],
                                                              top_shift[t, c])), axis=0)
                print('Shift Correction: ', shift_correction[t, c], '\n', 'Scale Correction: ', scale_correction[t, c])
                pbar.update(1)

    # Now regularise if these transforms are too different from neighbouring tiles. Since shifts computed are medians,
    # these are robust to errors, but the scales are not. So regularise for poorly scaled tiles
    for t in use_tiles:
        for c in use_channels:
            neighbours = np.sum(abs(tilepos-tilepos[t]), axis=1) == 1
            y_scale_neighb, x_scale_neighb = scale_correction[neighbours, c].T
            # Now we alter outlier scales. To define an outlier scale, we check if the avg distance from tile t's scale
            # to the neighbours scale is larger than the average distance of the neighbouring scales to each other (ie:
            # neighbouring scales standard deviation)
            if np.std(y_scale_neighb) < 10 * np.mean(abs(y_scale_neighb-scale_correction[t, c, 0])):
                scale_correction[t, c, 0] = np.median(y_scale_neighb)
            if np.std(x_scale_neighb) < 10 * np.mean(abs(x_scale_neighb-scale_correction[t, c, 1])):
                scale_correction[t, c, 1] = np.median(x_scale_neighb)

    # Finally, put this all together in a transform array which has dimensions (n_tiles x n_rounds x n_channels x 4 x 3)
    transform = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    for t in use_tiles:
        for r in use_rounds:
            for c in use_channels:
                transform[t, r, c, 0, 0] = scale_correction[t, c, 0]
                transform[t, r, c, 1, 1] = scale_correction[t, c, 1]
                transform[t, r, c, 2, 2] = z_expansion_factor[r]
                transform[t, r, c, 3, :] = (shift_correction[t, c] + initial_shift[t, r, c]) * [1, 1, z_scale]

    nbp.transform = transform

    return nbp
