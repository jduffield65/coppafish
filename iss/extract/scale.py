import numpy as np
from iss.extract.base import get_nd2_tile_ind
from iss.utils.morphology import convolve_2d
from iss import utils


def select_tile(tilepos_yx, use_tiles):
    """
    selects tile in use_tiles closest to centre.

    :param tilepos_yx: integer numpy array [n_tiles, 2]
        tiff tile positions (index 0 refers to [0,0])
    :param use_tiles: integer list [n_use_tiles]
    :return: integer
    """
    mean_yx = np.round(np.mean(tilepos_yx, 0))
    nearest_t = np.linalg.norm(tilepos_yx[use_tiles] - mean_yx, axis=1).argmin()
    return use_tiles[nearest_t]


def get_nd2_index(images, fov, channel, z):
    """

    :param images: ND2Reader object with fov, channel, z as index order.
    :param fov: integer. nd2 tile index, index -1 refers to tile at yx = [0,0]
    :param channel: integer. channel index
    :param z: integer. z-plane index
    :return: integer. index of desired plane in nd2 object
    """
    start_index = fov * images.sizes['c'] * images.sizes['z'] + channel * images.sizes['z']
    return start_index + z


def get_z_plane(images, fov, use_channels, use_z):
    """
    Finds z plane and channel that has maximum pixel value for given tile

    :param images: ND2Reader object with fov, channel, z as index order.
    :param fov: integer. nd2 tile index, index 0 refers to tile at yx = [MaxY,MaxX]
    :param use_channels: integer list. channels to consider
    :param use_z: integer list. z-planes to consider
    :return:
        max_channel: integer, channel to which image with max pixel value corresponds.
        max_z: integer, z-plane to which image with max pixel value corresponds.
        image: integer numpy array [tile_sz x tile_sz]: corresponding image.
    """
    image_max = np.zeros((len(use_channels), len(use_z)))
    for i in range(len(use_channels)):
        image_max[i, :] = np.max(np.max(utils.nd2.get_image(images, fov, i, use_z), axis=0), axis=0)
        # images[get_nd2_index(images, fov, use_channels[j], use_z[i])].max()
    max_channel = use_channels[np.max(image_max, axis=1).argmax()]
    max_z = use_z[np.max(image_max, axis=0).argmax()]
    return max_channel, max_z, utils.nd2.get_image(images, fov, max_channel, max_z)


def get_scale(im_file, tilepos_yx_tiff, tilepos_yx_nd2, use_tiles, use_channels, use_z, scale_norm, filter_kernel):
    """
    convolves the image for tile t, channel c, z-plane z with filter_kernel
    then gets the multiplier to apply to filtered nd2 images by dividing scale_norm by the max value of this
    filtered image

    :param im_file: string, file path of nd2 file
    :param tilepos_yx_tiff: numpy array[n_tiles x 2]
        [i,:] contains YX position of tile with tiff index i.
        index 0 refers to YX = [0,0]
    :param tilepos_yx_nd2: numpy array[n_tiles x 2]
        [i,:] contains YX position of tile with nd2 fov index i.
        index 0 refers to YX = [MaxY,MaxX]
    :param use_tiles: integer list. tiff tile indices to consider when finding tile if t is None
    :param use_channels: integer list. channels to consider when finding channel if c is None
    :param use_z: integer list. z-planes to consider when finding z_plane if z is None
    :param scale_norm: integer
    :param filter_kernel: numpy float array. Kernel to convolve nd2 data with to produce tiff tiles
    :return:
        t: integer, tiff tile index (index 0 refers to tilepos_yx['tiff']=[0,0]) scale found from.
        c: integer, channel scale found from.
        z: integer, z-plane scale found from.
        scale: float, multiplier to apply to filtered nd2 images before saving as tiff so full tiff uint16
               range occupied.
    """
    # tile to get scale from is central tile
    t = select_tile(tilepos_yx_tiff, use_tiles)
    images = utils.nd2.load(im_file)
    # find z-plane with max pixel across all channels of tile t
    c, z, image = get_z_plane(images, get_nd2_tile_ind(t, tilepos_yx_nd2, tilepos_yx_tiff), use_channels, use_z)
    # convolve_2d image in same way we convolve_2d before saving tiff files
    im_filtered = convolve_2d(image, filter_kernel)
    scale = scale_norm / im_filtered.max()
    return t, c, z, float(scale)
