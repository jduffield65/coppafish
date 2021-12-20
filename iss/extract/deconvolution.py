import numpy as np
from .. import utils
from ..find_spots.base import detect_spots, check_neighbour_intensity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
from . import scale
from .base import get_nd2_tile_ind


def get_isolated_points(spot_yx, isolation_dist):
    """
    get the isolated points in a point cloud as those whose neighbour is far.

    :param spot_yx: numpy integer array [n_peaks x image.ndim]
        yx or yxz location of spots found in image.
    :param isolation_dist: float
        spots are isolated if nearest neighbour is further away than this.
    :return: numpy boolean array [n_peaks]
    """
    nbrs = NearestNeighbors(n_neighbors=2).fit(spot_yx)
    distances, _ = nbrs.kneighbors(spot_yx)
    return distances[:, 1] > isolation_dist


def get_spot_images(image, spot_yx, shape):
    """
    builds an image around each spot of size given by shape and returns array containing all of these.

    :param image: numpy array [nY x nX (x nZ)]
        image spots were found on.
    :param spot_yx: numpy integer array [n_peaks x image.ndim]
        yx or yxz location of spots found.
    :param shape: list or numpy integer array giving size in y, x (and z) directions.
        desired size of image for each spot.
    :return: numpy array [n_peaks x y_shape x x_shape (x z_shape)]
    """
    if min(np.array(shape) % 2) == 0:
        raise ValueError(f"Require shape to be odd in each dimension but given shape was {shape}.")
    mid_index = np.ceil(np.array(shape)/2).astype(int) - 1  # index in spot_images where max intensity is for each spot.
    spot_images = np.empty((spot_yx.shape[0], *shape))
    spot_images[:] = np.nan  # set to nan if spot image goes out of bounds of image.
    max_image_index = np.array(image.shape)
    for s in tqdm(range(spot_yx.shape[0])):
        min_pos = np.clip((spot_yx[s] - mid_index), 0, max_image_index)
        max_pos = np.clip((spot_yx[s] + mid_index + 1), 0, max_image_index)
        spot_images_min_index = mid_index - (spot_yx[s] - min_pos)
        spot_images_max_index = mid_index + (max_pos - spot_yx[s])
        if len(shape) == 2:
            small_im = image[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1]]
            spot_images[s, spot_images_min_index[0]:spot_images_max_index[0],
                        spot_images_min_index[1]:spot_images_max_index[1]] = small_im
        elif len(shape) == 3:
            small_im = image[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], min_pos[2]:max_pos[2]]
            spot_images[s, spot_images_min_index[0]:spot_images_max_index[0],
                        spot_images_min_index[1]:spot_images_max_index[1],
                        spot_images_min_index[2]:spot_images_max_index[2]] = small_im
    return spot_images


def get_average_spot_image(spot_images, av_type='mean', symmetry=None, annulus_width=1.0):
    """
    given an array of spot images, this returns the average spot image.

    :param spot_images: numpy array [n_peaks x y_shape x x_shape (x z_shape)]
        array containing small images surrounding the n_peaks spots.
    :param av_type: 'mean' or 'median' indicating which average to use. optional.
        default: 'mean'
    :param symmetry: optional, default: None.
        None - just finds mean at every pixel.
        'quadrant_2d' - assumes each quadrant of each z-plane expected to look the same so concatenates these.
        'annulus' - assumes each z-plane is circularly symmetric about central pixel.
                    I.e. only finds only pixel value from all pixels a certain distance from centre.
    :param annulus_width: float, optional. default: 1.0
        if symmetry is 'annulus', this specifies how big an annulus to use, within which we expect all
        pixel values to be the same.
    :return: numpy array [y_shape x x_shape (x z_shape)]
    """
    if av_type == 'mean':
        av_func = lambda x, axis: np.nanmean(x, axis)
    elif av_type == 'median':
        av_func = lambda x, axis: np.nanmedian(x, axis)
    else:
        raise ValueError(f"av_type must be 'mean' or 'median' but value given was {av_type}")

    mid_index = np.ceil(np.array(spot_images.shape[1:]) / 2).astype(int) - 1

    if symmetry is None:
        av_image = av_func(spot_images, 0)
    elif symmetry == "quadrant_2d":
        # rotate all quadrants so spot is at bottom right corner
        quad1 = spot_images[:, 0:mid_index[0]+1, 0:mid_index[1]+1]
        quad2 = np.rot90(spot_images[:, 0:mid_index[0]+1, mid_index[1]:], 1, axes=(1, 2))
        quad3 = np.rot90(spot_images[:, mid_index[0]:, mid_index[1]:], 2, axes=(1, 2))
        quad4 = np.rot90(spot_images[:, mid_index[0]:, 0:mid_index[1]+1], 3, axes=(1, 2))
        all_quads = np.concatenate((quad1, quad2, quad3, quad4))
        av_quad = av_func(all_quads, 0)
        if spot_images.ndim == 4:
            av_image = np.pad(av_quad, [[0, mid_index[0]+1], [0, mid_index[1]+1], [0, 0]], 'symmetric')
        else:
            av_image = np.pad(av_quad, [[0, mid_index[0]+1], [0, mid_index[1]+1]], 'symmetric')
        # remove repeated central column and row
        av_image = np.delete(av_image, mid_index[0] + 1, axis=0)
        av_image = np.delete(av_image, mid_index[1] + 1, axis=1)
    elif symmetry == "annulus_2d":
        X, Y = np.meshgrid(np.arange(spot_images.shape[1]) - mid_index[0],
                           np.arange(spot_images.shape[2]) - mid_index[1])
        d = np.sqrt(X ** 2 + Y ** 2)
        annulus_bins = np.arange(0, d.max(), annulus_width)
        # find which bin each pixel should contribute to.
        bin_index = np.abs(np.expand_dims(d, 2) - annulus_bins).argmin(axis=2)
        av_image = np.zeros_like(spot_images[0])
        for i in range(annulus_bins.size):
            current_bin = bin_index == i
            av_image[current_bin] = av_func(spot_images[:, current_bin], (0, 1))
    else:
        raise ValueError(f"symmetry must be None, 'quadrant_2d' or 'annulus_2d' but value given was {symmetry}")
    return av_image


def plot_psf(psf, n_columns=2, log=False):
    """
    plot psf as a series of panels for each z-plane.

    :param psf: numpy array [y_shape x x_shape (x z_shape)]
    :param n_columns: number of columns to have in subplots.
    :param log: whether to take log10 of psf before plotting
    """
    n_rows = np.ceil(psf.shape[2]/n_columns).astype(int)
    fig, axs = plt.subplots(n_rows, n_columns, sharex='all', sharey='all')
    fig.set_figheight(n_rows*3)
    fig.set_figwidth((n_columns+1)*3)
    z = 0
    if log:
        small = min(psf[psf>0])/10000
        psf = np.log10(psf+small)
    caxis_min = np.percentile(psf, 1)
    caxis_max = psf.max()
    for i in range(n_columns):
        for j in range(n_rows):
            if z < psf.shape[2]:
                im = axs[j, i].imshow(psf[:, :, z], vmin=caxis_min, vmax=caxis_max)
                axs[j, i].set_title(f"z = {z}", fontsize=12)
                axs[j, i].xaxis.set_visible(False)
                axs[j, i].yaxis.set_visible(False)
                z += 1
            else:
               fig.delaxes(axs[j, i])
    plt.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()


def psf_pad(psf, image_shape):
    """
    pads psf with zeros so has same dimensions as image

    :param psf: numpy float array [y_diameter x x_diameter x z_diameter]
    :param image_shape: numpy integer array [y, x, z] number of pixels of padded image
    :return: numpy float array same size as image with psf centered on middle pixel.
    """
    # must pad with ceil first so that ifftshift puts central pixel to (0,0,0).
    pre_pad = np.ceil((np.array(image_shape)-np.array(psf.shape))/2).astype(int)
    post_pad = np.floor((np.array(image_shape)-np.array(psf.shape))/2).astype(int)
    return np.pad(psf, [(pre_pad[i], post_pad[i]) for i in range(len(pre_pad))])


def get_psf_spots(im_file, tilepos_yx_tiff, tilepos_yx_nd2, use_tiles, channel, use_z, radius_xy, radius_z,
                  min_spots, intensity_thresh, intensity_auto_param, isolation_dist, shape):
    """
    finds spot_shapes about spots found in raw data, average of these then used for psf.

    :param im_file: string, file path of reference round nd2 file
    :param tilepos_yx_tiff: numpy array[n_tiles x 2]
        [i,:] contains YX position of tile with tiff index i.
        index 0 refers to YX = [0,0]
    :param tilepos_yx_nd2: numpy array[n_tiles x 2]
        [i,:] contains YX position of tile with nd2 fov index i.
        index 0 refers to YX = [MaxY,MaxX]
    :param use_tiles: integer list. tiff tile indices used in experiment.
    :param channel: integer, reference channel.
    :param use_z: integer list. z-planes used in experiment.
    :param radius_xy: integer
        radius of dilation structuring element in xy plane (approximately spot radius)
    :param radius_z: integer
        radius of dilation structuring element in z direction (approximately spot radius)
    :param min_spots: integer, minimum number of spots required to determine average shape from. Typical: 300
    :param intensity_thresh: maybe_float, spots are local maxima in image with pixel value > intensity_thresh.
        if intensity_thresh is None, will automatically compute it from mid z-plane of first tile.
    :param intensity_auto_param: float, if intensity_thresh is automatically computed, it is done using this.
    :param isolation_dist: float, spots are isolated if nearest neighbour is further away than this.
    :param shape: list, desired size of image about each spot [y_diameter, x_diameter, z_diameter].
    :return:
        spot_images: numpy integer array [n_spots x y_diameter x x_diameter x z_diameter]
        intensity_thresh: float, only different to input if input was None.
        tiles_used: list, tiles used to get spots.
    """
    n_spots = 0
    images = utils.nd2.load(im_file)
    spot_images = np.zeros((0, shape[0], shape[1], shape[2]), dtype=int)
    tiles_used = []
    while n_spots < min_spots:
        t = scale.select_tile(tilepos_yx_tiff, use_tiles)  # choose tile closet to centre
        im = utils.nd2.get_image(images, get_nd2_tile_ind(t, tilepos_yx_nd2, tilepos_yx_tiff), channel, use_z)
        mid_z = np.ceil(im.shape[2] / 2).astype(int)
        median_im = np.median(im[:, :, mid_z])
        if intensity_thresh is None:
            intensity_thresh = median_im + np.median(np.abs(im[:, :, mid_z] - median_im)) * intensity_auto_param
        elif intensity_thresh <= median_im or intensity_thresh >= np.iinfo(np.uint16).max:
            raise utils.errors.OutOfBoundsError("intensity_thresh", intensity_thresh, median_im,
                                                np.iinfo(np.uint16).max)
        spot_yxz, _ = detect_spots(im, intensity_thresh, radius_xy, radius_z)
        # check fall off in intensity not too large
        not_single_pixel = check_neighbour_intensity(im, spot_yxz, median_im)
        isolated = get_isolated_points(spot_yxz, isolation_dist)
        spot_yxz = spot_yxz[np.logical_and(isolated, not_single_pixel), :]
        if n_spots == 0 and np.shape(spot_yxz)[0] < min_spots/4:
            # raise error on first tile if looks like we are going to use more than 4 tiles
            raise ValueError(f"\nFirst tile, {t}, only found {np.shape(spot_yxz)[0]} spots."
                             f"\nMaybe consider lowering intensity_thresh from current value of {intensity_thresh}.")
        spot_images = np.append(spot_images, get_spot_images(im, spot_yxz, shape), axis=0)
        n_spots = np.shape(spot_images)[0]
        use_tiles = np.setdiff1d(use_tiles, t)
        tiles_used.append(t)
        if len(use_tiles) == 0 and n_spots < min_spots:
            raise ValueError(f"\nRequired min_spots = {min_spots}, but only found {n_spots}.\n"
                             f"Maybe consider lowering intensity_thresh from current value of {intensity_thresh}.")
    return spot_images, intensity_thresh.astype(float), tiles_used


def get_psf(spot_images, annulus_width):
    """
    this gets psf, which is average image of spot from individual images of spots.

    :param spot_images: numpy integer array [n_spots x y_diameter x x_diameter x z_diameter]
    :param annulus_width: float, in each z_plane, this specifies how big an annulus to use,
        within which we expect all pixel values to be the same.
    :return: numpy float array [y_diameter x x_diameter x z_diameter]
    """
    # normalise each z plane of each spot image first so each has median of 0 and max of 1.
    # Found that this works well as taper psf anyway, which gives reduced intensity as move away from centre.
    spot_images = spot_images - np.expand_dims(np.nanmedian(spot_images, axis=[1, 2]), [1, 2])
    spot_images = spot_images / np.expand_dims(np.nanmax(spot_images, axis=(1, 2)), [1, 2])
    psf = get_average_spot_image(spot_images, 'median', 'annulus_2d', annulus_width)
    # normalise psf so min is 0 and max is 1.
    psf = psf - psf.min()
    psf = psf / psf.max()
    return psf


def get_wiener_filter(psf, image_shape, constant):
    """
    this tapers the psf so goes to 0 at edges and then computes wiener filter from it

    :param psf: numpy float array [y_diameter x x_diameter x z_diameter]
    :param image_shape: numpy integer array, indicates the shape of the tiles to be convolved after padding.
        [n_im_y, n_im_x, n_im_z]
    :param constant: float, constant used in wiener filter
    :return: numpy complex128 array [n_im_y x n_im_x x n_im_z]
    """
    # taper psf so smoothly goes to 0 at each edge.
    psf = psf * np.hanning(psf.shape[0]).reshape(-1, 1, 1) * np.hanning(psf.shape[1]).reshape(1, -1, 1) * \
          np.hanning(psf.shape[2]).reshape(1, 1, -1)
    psf = psf_pad(psf, image_shape)
    psf_ft = np.fft.fftn(np.fft.ifftshift(psf))
    return np.conj(psf_ft) / np.real((psf_ft * np.conj(psf_ft) + constant))


def wiener_deconvolve(image, im_pad_shape, filter):
    """
    this pads image so goes to median value of image at each edge. Then deconvolves using wiener filter.

    :param image: numpy integer array [n_im_y, n_im_x, n_im_z]. image to be deconvolved
    :param im_pad_shape: list [n_pad_y, n_pad_x, n_pad_z]. how much to pad image in y, x, z directions.
    :param filter: numpy complex128 array [n_im_y+2*n_pad_y, n_im_x+2*n_pad_x, n_im_z+2*n_pad_z].
        wiener filter to use.
    :return: numpy integer array [n_im_y, n_im_x, n_im_z]
    """
    im_max = image.max()
    im_min = image.min()
    im_av = np.median(image[:, :, 0])
    image = np.pad(image, [(im_pad_shape[i], im_pad_shape[i]) for i in range(len(im_pad_shape))], 'linear_ramp',
                   end_values=[(im_av, im_av)] * 3)
    im_deconvolved = np.real(np.fft.ifftn(np.fft.fftn(image) * filter))
    im_deconvolved = im_deconvolved[im_pad_shape[0]:-im_pad_shape[0], im_pad_shape[1]:-im_pad_shape[1],
                                    im_pad_shape[2]:-im_pad_shape[2]]
    # set min and max so it covers same range as input image
    im_deconvolved = im_deconvolved - im_deconvolved.min()
    return np.round(im_deconvolved * (im_max-im_min) / im_deconvolved.max() + im_min).astype(int)
