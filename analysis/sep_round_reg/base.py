import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage import exposure
from skimage.transform import warp_polar, rotate
from skimage.filters import window, gaussian
from skimage.registration import phase_cross_correlation
from coppafish.setup.notebook import NotebookPage
import napari
from scipy import interpolate
from coppafish.setup.notebook import Notebook
from skimage import data
from skimage.color import rgb2gray


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
    # There are 2 complications here:
    # 1.) Starting point may be out of range
    # 2.) Starting point may already be filled in
    # In both cases we must update the starting point and specify the new position we will start reading the small
    # image from.
    # CASE 1.
    # First update starting point, then find shift
    starting_point_old = starting_point
    starting_point = np.maximum(starting_point, [0, 0, 0], dtype=int)
    # Now if the old starting point was outside of the boundary of the image, we must only lay down the part of
    # small_image that should be included. So define shift to be the difference between starting points and we will
    # only start reading small image from z_coord shift[0], y_coord shift[1] and x_coord shift[2]
    shift1 = starting_point - starting_point_old
    # Now define the ending_point. This prevents the small image from spilling over at the end.
    ending_point = np.minimum(starting_point + small_image.shape, large_image.shape)

    # CASE 2.
    # First find shift, then update starting point.
    # Search for first x,y coord after which there are only zeros in the desired region of large_image
    shift2 = np.zeros(3)
    region = large_image[starting_point[0]:ending_point[0], starting_point[1]:ending_point[1],
             starting_point[2]:ending_point[2]]
    # pos_indices is 3 row by num_positive_indices cols
    zero_indices = np.argwhere(region == 0)
    # if this is nonempty then for y_shift, we look for the largest pos y index, x_shift is largest pos x index,
    # don't bother doing this for z_index though.
    if np.min(zero_indices.shape) > 0:
        shift2[1] = np.min(zero_indices[:, 1]).astype(int)
        shift2[2] = np.min(zero_indices[:, 2]).astype(int)

    # Now find total shift (sum of shift1 and shift 2) and update starting point. This concludes case 2.
    shift_total = np.array(shift1 + shift2, dtype=int)
    starting_point = np.array(starting_point + shift2, dtype=int)

    # Now we have the information that we need and we can populate the large image with the small image.
    # We must disregard the 0:shift[0] z entries, 0:shift[1] y entries, 0:shift[2] x entries of small_image.
    small_image = small_image[shift_total[0]:, shift_total[1]:, shift_total[2]:]
    # This still may spill over at the edge, so crop it at the end to ensure this doesn't happen
    if starting_point[0] + small_image.shape[0] > large_image.shape[0]:
        small_image = small_image[:(large_image.shape[0] - starting_point[0]), :, :]
    if starting_point[1] + small_image.shape[1] > large_image.shape[1]:
        small_image = small_image[:, :(large_image.shape[1] - starting_point[1]), :]
    if starting_point[2] + small_image.shape[2] > large_image.shape[2]:
        small_image = small_image[:, :, :(large_image.shape[2] - starting_point[2])]

    # Now taper the small image at the edges, this helps reduce harsh lines
    small_image = edge_taper(small_image)

    # Next we place small image inside large image. Since starting point has been updated, dimensions should match
    large_image[starting_point[0]:starting_point[0] + small_image.shape[0],
    starting_point[1]:starting_point[1] + small_image.shape[1],
    starting_point[2]:starting_point[2] + small_image.shape[2]] = small_image

    return large_image


def populate2(small_image: np.ndarray, large_image: np.ndarray, starting_point: np.ndarray):
    """
    Function which places a 3d image (small_image) into a large 3d image (large_image) without double covering.
    This function computes way too many checks and is very slow.
    Args:
        small_image: smaller image which will be placed into available region of large_image (np.ndarray)
        large_image: large_image which we are placing smaller image inside (np.ndarray)
        starting_point: bottom left corner in large_image where we want to place small_image
    Returns:
        large_image: large_image with smaller image placed inside appropriately (np.ndarray)
    """

    for i in range(small_image.shape[0]):
        for j in range(small_image.shape[1]):
            for k in range(small_image.shape[2]):
                global_coord = starting_point + [i, j, k]
                if (0 <= global_coord[0] < large_image.shape[0] and 0 <= global_coord[1] < large_image.shape[1] and
                        0 <= global_coord[2] < large_image.shape[2]):
                    if large_image[global_coord[0], global_coord[1], global_coord[2]] == 0:
                        large_image[global_coord[0], global_coord[1], global_coord[2]] = small_image[i, j, k]

    return large_image


def populate3(new_tile: np.ndarray, working_canvas: np.ndarray, ref_image: np.ndarray, starting_point: np.ndarray,
              padding: np.ndarray):
    """
    This function populates the large image without double covering.

    Args:
        new_tile: small image to be added
        ref_image: padded binary image which will be multiplied by the new_tile to get rid of redundant info
        working_canvas: current stage
        starting_point: coord of top left corner of new_tile
        padding: This is a 3d vector which specifies how much padding we want in z, y, x

    Returns:
        working_canvas: updated working canvas
        ref_image: updated ref_image
    """
    # Ensure padding is ndarray and not list
    padding = np.array(padding)
    # First, we will use the ref_image, which is 0 outside the range of the image, 0 wherever a previous tile has
    # been laid down and 1 otherwise. We will multiply this elementwise with an array of the same size which contains
    # 0's everywhere except where we would like to place our new_tile, where it takes the value of the tile.
    # Create large padded image of 0s except where we'd like our new tile. In those places store the value of the new_
    # tile
    image = np.zeros(ref_image.shape + 2*padding, dtype=int)
    image[padding[0] + starting_point[0]: padding[0] + starting_point[0] + new_tile.shape[0],
        padding[1] + starting_point[1]: padding[1] + starting_point[1] + new_tile.shape[1],
        padding[2] + starting_point[2]: padding[2] + starting_point[2] + new_tile.shape[2]] = new_tile
    # Now pad our ref_image
    ref_image_padded = np.zeros(ref_image.shape + 2*padding, dtype=bool)
    ref_image_padded[padding[0]:padding[0]+ref_image.shape[0],
                    padding[1]:padding[1]+ref_image.shape[1],
                    padding[2]:padding[2]+ref_image.shape[2]] = ref_image
    # Now multiply these 2 together elementwise. This will yield something which is 0 outside the boundary or
    # there is already a value at that point. Otherwise it takes the value of the new_tile in its desired position
    image = ref_image_padded * image
    # Now that the only nonzero elements of image are where the new_tile is, we can add this to our working_canvas,
    # which is 0 everywhere other than where we have already laid down the image.
    # First pad the working canvas
    working_canvas_padded = np.zeros(ref_image.shape + 2*padding)
    working_canvas_padded[padding[0]:padding[0]+ref_image.shape[0],
                    padding[1]:padding[1]+ref_image.shape[1],
                    padding[2]:padding[2]+ref_image.shape[2]] = working_canvas
    working_canvas_padded += image
    # Now update our reference_image to have 0s in the new places we have added a tile. Create an indicator array which
    # takes value 1 when image = 0 and 0 otherwise.
    indicator = image == 0
    # To get the new ref_image, multiply these elementwise. This is 0 wherever the ref_image is and 0 wherever the new
    # tile is that we've laid down and 1 otherwise. Since this is padded, it's also zero in the padding region.
    ref_image_padded = ref_image_padded * indicator
    # now remove padding on both ref_image and working_canvas
    working_canvas = working_canvas_padded[padding[0]:padding[0]+ref_image.shape[0],
                    padding[1]:padding[1]+ref_image.shape[1],
                    padding[2]:padding[2]+ref_image.shape[2]]
    ref_image = ref_image_padded[padding[0]:padding[0]+ref_image.shape[0],
                    padding[1]:padding[1]+ref_image.shape[1],
                    padding[2]:padding[2]+ref_image.shape[2]]

    return working_canvas, ref_image


def edge_taper(image: np.ndarray):
    """
    Blurring with a gradient towards the edges.
    Args:
        image: 3D Image whose edges we want blurred. Coords are in (z,y,x) format. (np.ndarray)
    Returns:
        image: 3D image with blurred edges. Coords are in (z,y,x) format. (np.ndarray)
    """
    # Initialise the blurred_image
    blurred_image = np.zeros(image.shape)
    # Since we'd like to taper the edges, we need to make a function that interpolates the image from 0 at the edges to
    # 1 in the middle
    z_len = image.shape[0]
    y_len = image.shape[1]
    x_len = image.shape[2]
    dy = y_len // 5
    dx = x_len // 5
    # Initialise x and y masks, these will be repeated so won't have this size for much longer
    y_mask = np.zeros(y_len)
    x_mask = np.zeros(x_len)
    for i in range(y_len):
        if i <= dy:
            y_mask[i] = i/dy
        elif dy < i <= y_len - dy:
            y_mask[i] = 1
        else:
            y_mask[i] = -(i-y_len)/dy
    y_mask = np.repeat(y_mask[:, np.newaxis], x_len, axis=1)
    for i in range(x_len):
        if i <= dx:
            x_mask[i] = i/dx
        elif dx < i <= x_len - dx:
            x_mask[i] = 1
        else:
            x_mask[i] = -(i-x_len)/dx
    x_mask = np.repeat(x_mask[np.newaxis, :], y_len, axis=0)

    original_mask = x_mask * y_mask
    blur_mask = 1 - original_mask
    blur_mask_binary = np.ceil(blur_mask)

    # everything so far has been done in yx, so now we must apply to each z-plane
    for i in range(z_len):
        blurred_image[i] = gaussian(image[i]*blur_mask_binary, 5)
        image[i] = image[i] * original_mask + blurred_image[i] * blur_mask

    return image


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


def manual_shift(im1: np.ndarray, im2: np.ndarray):
    """
    This function takes in 2 images and allows the user to select points on each using Napari and return the mean diff
    of this set of points. These points are the basis of the rigid transform detection later. This is currently not
    working.
    Args:
        im1: image 1 (np.ndarray)
        im2:  image 2 (np.ndarray)

    Returns:
        shift: mean diff between ref_points2 - ref_points1
        ref_points1: manually selected coords of features from image 1
        ref_points2: manually selected coords of features from image 2
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

    shift = np.mean(ref_points1 - ref_points2, axis=0, dtype=int)

    return shift, ref_points1, ref_points2


def detect_rotation(ref: np.ndarray, extra: np.ndarray):
    """
    Function which takes in 2 2D images which are rotated and translated with respect to one another and returns the
    rotation angle in degrees between them.
    Args:
        ref: reference image from the full notebook
        extra: new image from the partial notebook
    Returns:
        angle: Anticlockwise angle which, upon application to ref, yields extra.
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
    """
    Function to interpolate z-planes as these are z-pixels are 3 times larger than xy pixels.
    Args:
        image: 3d image whose z planes are to be interpolated (np.ndarray)
    Returns:
        new_image: image with 3 times as many z_planes, interpolated
    """
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
    """
    Function to view subsection of image1 and image2 overlayed in 3D.
    Args:
        im1:
        im2:

    Returns:
        ima
    """
    # Since z-pixels are 3 times as large as xy, we interpolate
    im1 = simple_z_interp(im1)
    im2 = simple_z_interp(im2)

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

# nb = Notebook('C://Users/Reilly/Desktop/Sample Notebooks/Anne/new_notebook.npz')
# image = patch_together(nb.get_config(), nb.basic_info, nb.stitch.tile_origin, [24, 25, 26])
# plt.imshow(image, cmap=plt.cm.gray)
# astro = rgb2gray(data.astronaut())
# astro_new = shift(astro, [10, 20])
# shift, error, phase = phase_cross_correlation(astro, astro_new)
# astro_new = rotate(astro_new, 7)
# shift, rp1, rp2 = manual_shift(astro, astro_new)
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
# astro = rgb2gray(data.astronaut())
# astro = astro[np.newaxis, :, :]
# moon = data.moon()
# moon = moon[np.newaxis, :, :]
# new_moon = edge_taper(moon)
# print("Hello Freaks")
