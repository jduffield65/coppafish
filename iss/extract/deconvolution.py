import numpy as np
from .. import utils
from ..find_spots.base import detect_spots, check_neighbour_intensity, get_isolated_points
from tqdm import tqdm
import matplotlib.pyplot as plt
from . import scale
from ..utils.nd2 import get_nd2_tile_ind
from ..setup import NotebookPage
from typing import List, Union, Optional, Tuple
import warnings


def get_spot_images(image: np.ndarray, spot_yxz: np.ndarray, shape: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Builds an image around each spot of size given by shape and returns array containing all of these.

    Args:
        image: ```float [nY x nX (x nZ)]```.
            Image that spots were found on.
        spot_yxz: ```int [n_peaks x image.ndim]```.
            yx or yxz location of spots found.
        shape: ```int [image.ndim]```
            ```[y_shape, x_shape, (z_shape)]```: Desired size of image for each spot in each direction.

    Returns:
        ```float [n_peaks x y_shape x x_shape (x z_shape)]```. ```[s]``` is the small image surrounding spot ```s```.
    """
    if min(np.array(shape) % 2) == 0:
        raise ValueError(f"Require shape to be odd in each dimension but given shape was {shape}.")
    mid_index = np.ceil(np.array(shape) / 2).astype(
        int) - 1  # index in spot_images where max intensity is for each spot.
    spot_images = np.empty((spot_yxz.shape[0], *shape))
    spot_images[:] = np.nan  # set to nan if spot image goes out of bounds of image.
    max_image_index = np.array(image.shape)
    n_spots = spot_yxz.shape[0]
    no_verbose = n_spots < 6000 / len(shape)  # show progress bar with lots of pixels.
    with tqdm(total=n_spots, disable=no_verbose) as pbar:
        pbar.set_description("Loading in spot images from tiff files")
        for s in range(n_spots):
            min_pos = np.clip((spot_yxz[s] - mid_index), 0, max_image_index)
            max_pos = np.clip((spot_yxz[s] + mid_index + 1), 0, max_image_index)
            spot_images_min_index = mid_index - (spot_yxz[s] - min_pos)
            spot_images_max_index = mid_index + (max_pos - spot_yxz[s])
            if len(shape) == 2:
                small_im = image[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1]]
                spot_images[s, spot_images_min_index[0]:spot_images_max_index[0],
                spot_images_min_index[1]:spot_images_max_index[1]] = small_im
            elif len(shape) == 3:
                small_im = image[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], min_pos[2]:max_pos[2]]
                spot_images[s, spot_images_min_index[0]:spot_images_max_index[0],
                spot_images_min_index[1]:spot_images_max_index[1],
                spot_images_min_index[2]:spot_images_max_index[2]] = small_im
            pbar.update(1)
    pbar.close()
    return spot_images


def get_average_spot_image(spot_images: np.ndarray, av_type: str = 'mean', symmetry: Optional[str] = None,
                           annulus_width: float = 1.0) -> np.ndarray:
    """
    Given an array of spot images, this returns the average spot image.

    Args:
        spot_images: ```float [n_peaks x y_shape x x_shape (x z_shape)]```.
            ```spot_images[s]``` is the small image surrounding spot ```s```.
            Any nan values will be ignored when computing the average spot image.
        av_type: Optional, one of the following indicating which average to use:

            - ```'mean'```
            - ```'median'```
        symmetry: Optional, one of the following:

            - ```None``` - Just finds mean at every pixel.
            - ```'quadrant_2d'``` - Assumes each quadrant of each z-plane expected to look the same so concatenates
                these.
            - ```'annulus_2d'``` - assumes each z-plane is circularly symmetric about central pixel.
                I.e. only finds only pixel value from all pixels a certain distance from centre.
            - ```'annulus_3d'``` - Same as ```'annulus_2d'```, except now z-planes are symmetric about the mid-plane.
                I.e. `av_image[:,:,mid-i] = av_image[:,:,mid+i]` for all `i`.
        annulus_width: If ```symmetry = 'annulus'```, this specifies how big an annulus to use,
            within which we expect all pixel values to be the same.

    Returns:
        ```float [y_shape x x_shape (x z_shape)]```. Average small image about a spot.
    """
    # avoid nan in average because some spot_images may have nans because the image ran out of bounds of the tile.
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
        quad1 = spot_images[:, 0:mid_index[0] + 1, 0:mid_index[1] + 1]
        quad2 = np.rot90(spot_images[:, 0:mid_index[0] + 1, mid_index[1]:], 1, axes=(1, 2))
        quad3 = np.rot90(spot_images[:, mid_index[0]:, mid_index[1]:], 2, axes=(1, 2))
        quad4 = np.rot90(spot_images[:, mid_index[0]:, 0:mid_index[1] + 1], 3, axes=(1, 2))
        all_quads = np.concatenate((quad1, quad2, quad3, quad4))
        av_quad = av_func(all_quads, 0)
        if spot_images.ndim == 4:
            av_image = np.pad(av_quad, [[0, mid_index[0] + 1], [0, mid_index[1] + 1], [0, 0]], 'symmetric')
        else:
            av_image = np.pad(av_quad, [[0, mid_index[0] + 1], [0, mid_index[1] + 1]], 'symmetric')
        # remove repeated central column and row
        av_image = np.delete(av_image, mid_index[0] + 1, axis=0)
        av_image = np.delete(av_image, mid_index[1] + 1, axis=1)
    elif symmetry == "annulus_2d" or symmetry == "annulus_3d":
        X, Y = np.meshgrid(np.arange(spot_images.shape[1]) - mid_index[0],
                           np.arange(spot_images.shape[2]) - mid_index[1])
        d = np.sqrt(X ** 2 + Y ** 2)
        annulus_bins = np.arange(0, d.max(), annulus_width)
        # find which bin each pixel should contribute to.
        bin_index = np.abs(np.expand_dims(d, 2) - annulus_bins).argmin(axis=2)
        av_image = np.zeros_like(spot_images[0])
        if symmetry == "annulus_3d":
            if spot_images.ndim != 4:
                raise ValueError("Must give 3D images with symmetry = 'annulus_3d'")
            n_z = spot_images.shape[3]
            if n_z % 2 == 0:
                raise ValueError("Must have odd number of z-planes with symmetry = 'annulus_3d'")
            # ensure each z-plane has unique set of indices so can average each separately.
            bin_index = np.tile(np.expand_dims(bin_index, 2), [1, 1, n_z])
            for i in range(mid_index[2]):
                current_max_index = bin_index[:, :, mid_index[2] - i].max()
                bin_index[:, :, mid_index[2] - i - 1] = bin_index[:, :, mid_index[2]] + current_max_index + 1
                bin_index[:, :, mid_index[2] + i + 1] = bin_index[:, :, mid_index[2] - i - 1]
        for i in np.unique(bin_index):
            current_bin = bin_index == i
            av_image[current_bin] = av_func(spot_images[:, current_bin], (0, 1))
    else:
        raise ValueError(f"symmetry must be None, 'quadrant_2d', 'annulus_2d' or 'annulus_3d' but value given was "
                         f"{symmetry}")

    if symmetry is not None:
        is_odd = (np.array(spot_images.shape[1:3]) % 2).astype(bool)
        if not is_odd.all():
            warnings.warn(f"spot_images shape is {av_image.shape} which is even in some dimensions."
                          f"\nThis means centre of symmetry will be off-centre.")
    return av_image


def plot_psf(psf: np.ndarray, n_columns: int = 2, log: bool = False):
    """
    Plot psf as a series of panels for each z-plane.

    Args:
        psf: ```float [y_shape x x_shape (x z_shape)]```. Point Spread Function to plot.
        n_columns: Number of columns to have in subplots.
        log: Whether to take log10 of psf before plotting
    """
    n_rows = np.ceil(psf.shape[2] / n_columns).astype(int)
    fig, axs = plt.subplots(n_rows, n_columns, sharex='all', sharey='all')
    fig.set_figheight(n_rows * 3)
    fig.set_figwidth((n_columns + 1) * 3)
    z = 0
    if log:
        small = min(psf[psf > 0]) / 10000
        psf = np.log10(psf + small)
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


def psf_pad(psf: np.ndarray, image_shape: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Pads psf with zeros so has same dimensions as image

    Args:
        psf: ```float [y_shape x x_shape (x z_shape)]```.
            Point Spread Function with same shape as small image about each spot.
        image_shape: ```int [psf.ndim]```.
            Number of pixels in ```[y, x, (z)]``` direction of padded image.

    Returns:
        ```float [image_shape[0] x image_shape[1] (x image_shape[2])]```.
        Array same size as image with psf centered on middle pixel.
    """
    # must pad with ceil first so that ifftshift puts central pixel to (0,0,0).
    pre_pad = np.ceil((np.array(image_shape) - np.array(psf.shape)) / 2).astype(int)
    post_pad = np.floor((np.array(image_shape) - np.array(psf.shape)) / 2).astype(int)
    return np.pad(psf, [(pre_pad[i], post_pad[i]) for i in range(len(pre_pad))])


def get_psf_spots(nbp_file: NotebookPage, nbp_basic: NotebookPage, round: int,
                  use_tiles: List[int], channel: int, use_z: List[int], radius_xy: int, radius_z: int, min_spots: int,
                  intensity_thresh: Optional[float], intensity_auto_param: float, isolation_dist: float,
                  shape: List[int]) -> Tuple[np.ndarray, float, List[int]]:
    """
    Finds spot_shapes about spots found in raw data, average of these then used for psf.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        round: Reference round to get spots from to determine psf.
            This should be the anchor round (last round) if using.
        use_tiles: ```int [n_use_tiles]```.
            tiff tile indices used in experiment.
        channel: Reference channel to get spots from to determine psf.
        use_z: ```int [n_z]```. Z-planes used in the experiment.
        radius_xy: Radius of dilation structuring element in xy plane (approximately spot radius).
        radius_z: Radius of dilation structuring element in z direction (approximately spot radius)
        min_spots: Minimum number of spots required to determine average shape from. Typical: 300
        intensity_thresh: Spots are local maxima in image with ```pixel value > intensity_thresh```.
            if ```intensity_thresh = None```, will automatically compute it from mid z-plane of first tile.
        intensity_auto_param: If ```intensity_thresh = None``` so is automatically computed, it is done using this.
        isolation_dist: Spots are isolated if nearest neighbour is further away than this.
        shape: ```int [y_diameter, x_diameter, z_diameter]```. Desired size of image about each spot.

    Returns:
        - ```spot_images``` - ```int [n_spots x y_diameter x x_diameter x z_diameter]```.
            ```spot_images[s]``` is the small image surrounding spot ```s```.
        - ```intensity_thresh``` - ```float```. Only different from input if input was ```None```.
        - ```tiles_used``` - ```int [n_tiles_used]```. Tiles the spots were found on.
    """
    n_spots = 0
    spot_images = np.zeros((0, shape[0], shape[1], shape[2]), dtype=int)
    tiles_used = []
    while n_spots < min_spots:
        t = scale.central_tile(nbp_basic.tilepos_yx, use_tiles)  # choose tile closet to centre
        im = utils.raw.load(nbp_file, nbp_basic, None, round, channel, use_z)
        mid_z = np.ceil(im.shape[2] / 2).astype(int)
        median_im = np.median(im[:, :, mid_z])
        if intensity_thresh is None:
            intensity_thresh = median_im + np.median(np.abs(im[:, :, mid_z] - median_im)) * intensity_auto_param
        elif intensity_thresh <= median_im or intensity_thresh >= np.iinfo(np.uint16).max:
            raise utils.errors.OutOfBoundsError("intensity_thresh", intensity_thresh, median_im,
                                                np.iinfo(np.uint16).max)
        spot_yxz, _ = detect_spots(im, intensity_thresh, radius_xy, radius_z, True)
        # check fall off in intensity not too large
        not_single_pixel = check_neighbour_intensity(im, spot_yxz, median_im)
        isolated = get_isolated_points(spot_yxz, isolation_dist)
        spot_yxz = spot_yxz[np.logical_and(isolated, not_single_pixel), :]
        if n_spots == 0 and np.shape(spot_yxz)[0] < min_spots / 4:
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


def get_psf(spot_images: np.ndarray, annulus_width: float) -> np.ndarray:
    """
    This gets psf, which is average image of spot from individual images of spots.
    It is normalised so min value is 0 and max value is 1.

    Args:
        spot_images: ```int [n_spots x y_diameter x x_diameter x z_diameter]```.
            ```spot_images[s]``` is the small image surrounding spot ```s```.
        annulus_width: Within each z-plane, this specifies how big an annulus to use,
            within which we expect all pixel values to be the same.

    Returns:
        ```float [y_diameter x x_diameter x z_diameter]```.
            Average small image about a spot. Normalised so min is 0 and max is 1.
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


def get_wiener_filter(psf: np.ndarray, image_shape: Union[np.ndarray, List[int]], constant: float) -> np.ndarray:
    """
    This tapers the psf so goes to 0 at edges and then computes wiener filter from it.

    Args:
        psf: ```float [y_diameter x x_diameter x z_diameter]```.
            Average small image about a spot. Normalised so min is 0 and max is 1.
        image_shape: ```int [n_im_y, n_im_x, n_im_z]```.
            Indicates the shape of the image to be convolved after padding.
        constant: Constant used in wiener filter.

    Returns:
        ```complex128 [n_im_y x n_im_x x n_im_z]```. Wiener filter of same size as image.
    """
    # taper psf so smoothly goes to 0 at each edge.
    psf = psf * np.hanning(psf.shape[0]).reshape(-1, 1, 1) * np.hanning(psf.shape[1]).reshape(1, -1, 1) * \
          np.hanning(psf.shape[2]).reshape(1, 1, -1)
    psf = psf_pad(psf, image_shape)
    psf_ft = np.fft.fftn(np.fft.ifftshift(psf))
    return np.conj(psf_ft) / np.real((psf_ft * np.conj(psf_ft) + constant))


def wiener_deconvolve(image: np.ndarray, im_pad_shape: List[int], filter: np.ndarray) -> np.ndarray:
    """
    This pads ```image``` so goes to median value of ```image``` at each edge. Then deconvolves using wiener filter.

    Args:
        image: ```int [n_im_y x n_im_x x n_im_z]```.
            Image to be deconvolved.
        im_pad_shape: ```int [n_pad_y, n_pad_x, n_pad_z]```.
            How much to pad image in ```[y, x, z]``` directions.
        filter: ```complex128 [n_im_y+2*n_pad_y, n_im_x+2*n_pad_x, n_im_z+2*n_pad_z]```.
            Wiener filter to use.

    Returns:
        ```int [n_im_y x n_im_x x n_im_z]```. Deconvolved image.
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
    return np.round(im_deconvolved * (im_max - im_min) / im_deconvolved.max() + im_min).astype(int)
