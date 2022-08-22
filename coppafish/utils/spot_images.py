import warnings
from typing import Union, List, Optional
import numpy as np
from tqdm import tqdm


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
