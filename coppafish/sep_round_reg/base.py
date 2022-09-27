import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage import exposure
from skimage.transform import warp_polar, rotate
from skimage.filters import window
from skimage.registration import phase_cross_correlation
from coppafish.setup.notebook import NotebookPage
import napari
from scipy import interpolate
from coppafish.setup.notebook import Notebook
from skimage import data
from skimage.color import rgb2gray
import scipy.ndimage as snd

matplotlib.use('QtAgg')
matplotlib.pyplot.style.use('dark_background')


def process_image(image: np.ndarray, gamma: int, y: int, x: int, length: int):
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

    # Crop the image
    image = image[:, y:y + length, x:x + length]
    # Next, make sure the contrast is well-adjusted
    image = exposure.equalize_hist(image)
    # Rescale so max = 1
    max = np.array(image).max()
    image = image / max
    # Invert the image
    image = np.ones(image.shape) - image
    # Now apply the gamma transformation
    image = image ** gamma
    # Apply Hann Window to Images 1 and 2
    # for i in range(image.shape[0]):
    #     image[i] = image[i] * (window('hann', image[i].shape) ** 0.1)

    return image


def manual_shift(image1: np.ndarray, image2: np.ndarray):
    """
    This function takes in 2 images and allows the user to select 3 points on eac using Matplotlib then returns the
    mean shift between corresponding points
    """
    # Plot image 1 and image 2 side by side
    # plot 1
    plt.subplot(121)
    plt.imshow(image1, cmap=plt.cm.gray)
    # plot 2:
    plt.subplot(122)
    plt.imshow(image2, cmap=plt.cm.gray)
    plt.pause(0.01)

    # Now require 6 inputs from user, 3 for each image and give a timer of 2 mins
    ref_points_1 = np.array(plt.ginput(3, 60))
    # plot 1
    plt.subplot(121)
    plt.scatter(ref_points_1[:, 0], ref_points_1[:, 1], s=100, c='red')
    plt.pause(0.01)

    ref_points_2 = np.array(plt.ginput(3, 60))
    # plot 2:
    plt.subplot(122)
    plt.scatter(ref_points_2[:, 0], ref_points_2[:, 1], s=100, c='red')
    plt.pause(0.01)

    # Average across these shifts
    ref_points_1 = np.flip(ref_points_1)
    ref_points_2 = np.flip(ref_points_2)
    shift = np.mean(ref_points_2 - ref_points_1, axis=0, dtype='int')

    return shift, ref_points_1, ref_points_2


def manual_shift2(im1: np.ndarray, im2: np.ndarray):
    """
    This function takes in 2 images and allows the user to select points on each using Napari and return the mean diff
    of this set of points. These points are the basis of the rigid transform detection later. This is currently not
    working.
    """
    # Firstly, we need to make sure that the histograms match between images. This is a fancy way of saying that we'll
    # ensure they have the same amount of pixels of each brightness in [0,1]
    im2 = exposure.match_histograms(im2, im1)

    # Create Napari viewer
    viewer = napari.Viewer()

    # Add image 1 and image 2 as image layers in Napari
    viewer.add_image(im1, blending='additive', colormap='bop orange')
    viewer.add_image(im2, blending='additive', colormap='bop blue')

    # Add point layers for ref points in image 1 and image 2
    viewer.add_points(name='img1_ref')
    viewer.add_points(name='img2_ref')

    # Run Napari
    napari.run()

    # Read input points
    ref_points1 = viewer.layers['img1_ref'].data
    ref_points2 = viewer.layers['img2_ref'].data

    shift = np.mean(ref_points2 - ref_points1, axis=0, dtype=int)

    return shift, ref_points1, ref_points2


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
    radius = shape[0] // 2  # only take lower frequencies
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
    warped_ref_ft[np.isnan(warped_extra_ft)] = 0
    warped_extra_ft[np.isnan(warped_extra_ft)] = 0

    shifts = phase_cross_correlation(warped_ref_ft, warped_extra_ft, upsample_factor=100,
                                     reference_mask=np.ones(warped_ref_ft.shape, dtype=int) - np.isnan(warped_ref_ft),
                                     moving_mask=np.ones(warped_ref_ft.shape, dtype=int) - np.isnan(warped_extra_ft),
                                     normalization=None)

    # Use translation parameters to calculate rotation parameter
    shift_angle = shifts[0]

    return shift_angle


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


def simple_z_interp(image: np.ndarray):
    # Since z-pixels are 3 times as large as xy, we interpolate
    num_z = image.shape[0]
    z = np.arange(0, num_z, 1)
    new_image = np.zeros((len(z), image.shape[1] // 3, image.shape[2] // 3))
    for i in range(new_image.shape[1]):
        for j in range(new_image.shape[2]):
            f = interpolate.interp1d(z, image[:, 3 * i, 3 * j])
            new_image[:, i, j] = f(z)
    return new_image


def overlay_3d(im1: np.ndarray, im2=np.ndarray):
    # Since z-pixels are 3 times as large as xy, we interpolate
    im1 = simple_z_interp(im1, 3)
    im2 = simple_z_interp(im2, 3)

    for i in range(im1.shape[0]):
        im1[i] = im1[i] * (window('hann', im1[i].shape) ** 0.1)
    for i in range(im2.shape[0]):
        im2[i] = im2[i] * (window('hann', im2[i].shape) ** 0.1)

    # Create Napari viewer
    viewer = napari.Viewer()

    # Add image 1 and image 2 as image layers in Napari
    viewer.add_image(im1, blending='additive', colormap='bop orange')
    viewer.add_image(im2, blending='additive', colormap='bop blue')

    napari.run()

    return im1, im2


def shift(array, offset, constant_values=0):
    """Returns copy of array shifted by offset, with fill using constant."""
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


def populate(small_image: np.ndarray, large_image: np.ndarray, starting_point: np.ndarray):
    """
    Function which places a 3d image (small_image) into a large 3d image (large_image) without double covering.
    Args:
        small_image: smaller image which will be placed into available region of large_image (np.ndarray)
        large_image: large_image which we are placing smaller image inside (np.ndarray)
        starting_point: bottom left corner in large_image where we want to place small_image
    Returns:
        large_image: large_image with smaller image placed inside appropriately (np.ndarray)
    """
    # First we compute the starting and ending z, y and x coord. First deal with the case that the coordinates
    # are out of the image boundary
    starting_point = np.maximum(starting_point, [0, 0, 0])
    ending_point = np.minimum(starting_point + small_image.shape, large_image.shape)

    # Now deal with the case where there is already something inside the desired region which we would like to avoid
    # double covering. Search for first x,y coord after which there are only zeros in the desired region of large_image
    region = large_image[starting_point[0]:ending_point[0], starting_point[1]:ending_point[1],
             starting_point[2]:ending_point[2]]
    positive_indices = np.argwhere(region)
    if np.min(positive_indices.shape) > 0:
        y_shift = np.max(positive_indices[:, 1])
        x_shift = np.max(positive_indices[:, 0])
    else:
        y_shift = 0
        x_shift = 0
    # Finally, we can populate the large image with the small image. We must disregard the x_shift and y_shift first
    # x and y entries though.
    z_len = ending_point[0] - starting_point[0]
    y_len = ending_point[1] - starting_point[1]
    x_len = ending_point[2] - starting_point[2]
    # Next we crop the small image
    small_image = small_image[:z_len, y_shift:y_shift + y_len, x_shift:x_shift + x_len]

    # Next we place small image inside large image
    large_image[starting_point[0]:ending_point[0], starting_point[1]:ending_point[1], starting_point[2]:ending_point[2]] \
        = small_image

    return large_image

# nb = Notebook('C://Users/Reilly/Desktop/Sample Notebooks/Anne/new_notebook.npz')
# image = patch_together(nb.get_config(), nb.basic_info, nb.stitch.tile_origin, [24, 25, 26])
# plt.imshow(image, cmap=plt.cm.gray)
# astro = rgb2gray(data.astronaut())
# astro_new = shift(astro, [10, 20])
# shift, error, phase = phase_cross_correlation(astro, astro_new)
# astro_new = rotate(astro_new, 7)
# shift, rp1, rp2 = manual_shift2(astro, astro_new)
# plt.show()
# print(shift)
# astro_new = shift(astro_new, shift)
# rgb_overlay = np.zeros((512, 512, 3))
# rgb_overlay[:, :, 0] = astro[:512, :512]
# rgb_overlay[:, :, 2] = astro_new[:512, :512]
# plt.imshow(rgb_overlay)
# plt.show()
# dapi_partial = np.load('C:/Users/Reilly/Desktop/Sample Notebooks/Christina/Sep Round/Sep/dapi_image.npz')
# dapi_partial = dapi_partial.f.arr_0
# dapi_full = np.load('C:/Users/Reilly/Desktop/Sample Notebooks/Christina/Sep Round/Full/dapi_image.npz')
# dapi_full = dapi_full.f.arr_0
# dapi_full = dapi_full[25]
# dapi_partial = dapi_partial[25]
# brain = data.brain()
# overlay_3d(brain, brain)
