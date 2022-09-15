import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage import exposure
from skimage.transform import warp_polar, rotate
from skimage.filters import window
from skimage.registration import phase_cross_correlation
from coppafish.setup.notebook import NotebookPage
from coppafish.setup.notebook import Notebook


def process_image(image: np.ndarray, z_planes: np.ndarray, gamma: int, y: int, x: int, length: int):
    """ This function takes in an image and filters it to make it easier for the rotation detection to get results
    Args:
        image: image to be filtered
        z_planes: The z_planes which we want to look at, if this has been omitted then the image is 2D
        gamma: The power which we will raise every pixel of the image to. Gamma = 1 is no change, Higher values of Gamma
        result in much starker contrast.
        y: y coord of bottom left corner of the square we are considering
        x: x coord of bottom left corner of the square we are considering
        length: side length of square we are considering

    Returns:
        image: Filtered image
    """

    if z_planes is not None:
        # First, collapse some z-planes
        flattened_img = np.zeros((image.shape[1], image.shape[2]))
        for i in z_planes:
            flattened_img += image[i]
        image = flattened_img

    # Crop the image
    image = image[y:y+length, x:x+length]
    # Next, make sure the contrast is well-adjusted
    image = exposure.equalize_hist(image)
    # Rescale so max = 1
    max = np.array(image).max()
    image = image/max
    # Invert the image
    image = np.ones(image.shape)-image
    # Now apply the gamma transformation
    image = image ** gamma
    # Apply Hann Window to Images 1 and 2
    image = image * (window('hann', image.shape) ** 0.1)

    return image


def manual_shift(image1: np.ndarray, image2: np.ndarray):
    """This function takes in 2 images and allows the user to select 3 points on each, then returns the mean shift
    between corresponding points"""
    # Plot image 1 and image 2 side by side
    # plot 1
    plt.subplot(121)
    plt.imshow(image1, cmap=plt.cm.gray)
    # plot 2:
    plt.subplot(122)
    plt.imshow(image2, cmap=plt.cm.gray)
    plt.show()
    plt.pause(0.0001)

    # Now require 6 inputs from user, 3 for each image and give a timer of 2 mins
    ref_points_1 = np.array(plt.ginput(3, 60))
    # plot 1
    plt.subplot(121)
    plt.scatter(ref_points_1[:, 0], ref_points_1[:, 1], s=100, c='red')
    plt.show()
    plt.pause(0.0001)

    ref_points_2 = np.array(plt.ginput(3, 60))
    # plot 2:
    plt.subplot(122)
    plt.scatter(ref_points_2[:, 0], ref_points_2[:, 1], s=100, c='red')
    plt.pause(0.0001)
    plt.show()

    # Average across these shifts
    shift = np.mean(ref_points_2-ref_points_1)

    return shift, ref_points_1, ref_points_2


def detect_rotation(ref: np.ndarray, extra: np.ndarray):

    """
    Function which takes in 2 images which are rotated and translated with respect to one another and returns the
    rotation angle in degrees between them.
    Args:
        ref: reference image from the full notebook
        extra: new image from the partial notebook
    Returns: Anticlockwise angle which, upon application to ref, yields extra.
    """
    # work with shifted FFT log-magnitudes
    ref_ft = np.log2(np.abs(fftshift(fft2(ref))))
    extra_ft = np.log2(np.abs(fftshift(fft2(extra))))

    # Plot image 1 and image 2 side by side
    # plt.subplot(1, 2, 1)
    # plt.imshow(ref_ft, cmap=plt.cm.gray)
    # plot 2:
    # plt.subplot(1, 2, 2)
    # plt.imshow(extra_ft, cmap=plt.cm.gray)
    # plt.show()

    # Create log-polar transformed FFT mag images and register
    shape = ref_ft.shape
    radius = shape[0] // 4  # only take lower frequencies
    warped_ref_ft = warp_polar(ref_ft, radius=radius, scaling='log')
    warped_extra_ft = warp_polar(extra_ft, radius=radius, scaling='log')

    # Plot image 1 and image 2 side by side
    # plt.subplot(1, 2, 1)
    # plt.imshow(warped_ref_ft, cmap=plt.cm.gray)
    # plot 2:
    # plt.subplot(1, 2, 2)
    # plt.imshow(warped_extra_ft, cmap=plt.cm.gray)
    # plt.show()

    warped_ref_ft = warped_ref_ft[:shape[0] // 2, :]  # only use half of FFT
    warped_extra_ft = warped_extra_ft[:shape[0] // 2, :]
    shifts, error, phasediff = phase_cross_correlation(warped_ref_ft,
                                                       warped_extra_ft,
                                                       upsample_factor=10,
                                                       normalization=None)

    # Use translation parameters to calculate rotation parameter
    shift_angle = shifts[0]

    return shift_angle, error


def patch_together(config: dict, nbp_basic: NotebookPage, tile_origin: np.ndarray, z_planes: np.ndarray):
    """This function creates a large stitched image (npy array) from the reference round of a notebook by adding tiles
    in ascending order to an image.

    Args:
        config: configuration file. Uses the following
            tile_dir: Physical location of tiles as npy files
        nbp_basic: Basic_info Page of Notebook. Uses the following
            tilepos_yx: array showing where different tiles are relative to one another
            tile_sz: Side length of the square yx coords
        tile_origin: global coordinate origin of bottom left hand corner of tile
        z_planes: The z_planes we wish to focus stack.
    """
    num_z = len(z_planes)
    tile_origin = tile_origin.astype(int)

    # These are just upper bounds for the height (axis 0) and width (axis 1) of our image
    image_height = np.max(tile_origin[:, 0]) + nbp_basic.tile_sz
    image_width = np.max(tile_origin[:, 1]) + nbp_basic.tile_sz

    # This will be the background we place our tiles on
    patchwork = np.zeros((image_height, image_width))

    # Next, loop through the tiles, laying them onto the image. To avoid double laying no pixels are stored on top
    # of pixels which are already present.
    anchor_channel = nbp_basic.anchor_channel
    for t in nbp_basic.use_tiles:
        # Load in z_shift for this tile
        z_shift = tile_origin[t, 2]
        # Get the tile directory as text
        tile_dir = config.get('file_names').get('tile_dir') + '/anchor_t' + str(t) + 'c' + str(anchor_channel) + '.npy'
        # Load in the whole image (np array of dimension total_z_planes * tile_sz * tile_sz)
        img = np.load(tile_dir)
        # Since we want to consider only z_planes given, we must load in the z_planes which will be shifted to those
        # positions
        z_planes_use = (z_planes - z_shift * np.ones((1, len(z_planes)))).astype(int)
        # Next, average across al z_planes we want to look at
        img = img[range(np.min(z_planes_use), np.max(z_planes_use) + 1), :, :]
        img = np.sum(img, 0) / num_z

        # Now that the tile is loaded in, we can start to overlay it
        # Start by extracting x,y coords of origin
        tile_origin_y = int(tile_origin[t, 0])
        tile_origin_x = int(tile_origin[t, 1])
        # Loop over each tile and if the corresponding position on the patchwork is empty, then place the pixel from
        # img down here
        for i in range(nbp_basic.tile_sz):
            for j in range(nbp_basic.tile_sz):
                if patchwork[i + tile_origin_y, j + tile_origin_x] == 0:
                    patchwork[i + tile_origin_y, j + tile_origin_x] = img[i, j]

    # Next we remove all padding from the image
    # We set a threshold of how many tiles of consecutive padding we see before we call it padding
    threshold = int(nbp_basic.tile_sz / 5)
    # Next we find the border thicknesses. Border is a 4-element array where border[0] is right border and the following
    # indices proceed anticlockwise around the image
    border = np.zeros(4).astype(dtype=int)

    # Find border thicknesses:
    # Start at the right of the image. Then loop over leftwards until min of each col > 0
    min_val = 0
    while min_val == 0:
        min_val = np.min(patchwork[threshold:-threshold, -border[0]])
        border[0] = border[0] + 1
    # Start at the left of the image. Then loop over rightwards until min of each col > 0
    min_val = 0
    while min_val == 0:
        min_val = np.min(patchwork[threshold:-threshold, border[2]])
        border[2] = border[2] + 1
    # Start at the top of the image. Then loop over downwards until min of each col > 0
    min_val = 0
    while min_val == 0:
        min_val = np.min(patchwork[threshold:-threshold, -border[1]])
        border[1] = border[1] + 1
    # Start at the bottom of the image. Then loop over upwards until min of each col > 0
    min_val = 0
    while min_val == 0:
        min_val = np.min(patchwork[threshold:-threshold, border[3]])
        border[3] = border[3] + 1

    # Next, crop these borders in our final image
    patchwork = patchwork[border[1]:-border[3], border[2]:-border[0]]
    return patchwork

# nb = Notebook('C://Users/Reilly/Desktop/Sample Notebooks/Anne/new_notebook.npz')
# image = patch_together(nb.get_config(), nb.basic_info, nb.stitch.tile_origin, [24, 25, 26])
# plt.imshow(image, cmap=plt.cm.gray)
