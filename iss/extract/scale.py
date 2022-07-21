import numpy as np
from .base import get_nd2_tile_ind
from .. import utils
from typing import List, Tuple, Optional
import nd2


def select_tile(tilepos_yx: np.ndarray, use_tiles: List[int]) -> int:
    """
    Selects tile in use_tiles closest to centre.

    Args:
        tilepos_yx: ```int [n_tiles x 2]```.
            tiff tile positions (index ```0``` refers to ```[0,0]```).
        use_tiles: ```int [n_use_tiles]```.
            Tiles used in the experiment.

    Returns:
        tile in ```use_tiles``` closest to centre.
    """
    mean_yx = np.round(np.mean(tilepos_yx, 0))
    nearest_t = np.linalg.norm(tilepos_yx[use_tiles] - mean_yx, axis=1).argmin()
    return use_tiles[nearest_t]


def get_nd2_index(images: nd2.ND2File, fov: int, channel: int, z: int) -> int:
    """
    Gets index of desired plane in nd2 file.

    Args:
        images: ND2Reader object with ```fov```, ```channel```, ```z``` as index order.
        fov: nd2 tile index, index ```-1``` refers to tile at ```yx = [0,0]```.
        channel: Channel index.
        z: Z-plane index.

    Returns:
        Index of desired plane in nd2 file.
    """
    start_index = fov * images.sizes['c'] * images.sizes['z'] + channel * images.sizes['z']
    return start_index + z


def get_z_plane(images: nd2.ND2File, fov: int, use_channels: List[int], use_z: List[int]) -> \
        Tuple[int, int, np.ndarray]:
    """
    Finds z plane and channel that has maximum pixel value for given tile.

    Args:
        images: ND2Reader object with ```fov```, ```channel```, ```z``` as index order.
        fov: nd2 tile index, index ```-1``` refers to tile at ```yx = [0,0]```.
        use_channels: ```int [n_use_channels]```.
            Channels to consider.
        use_z: ```int [n_z]```.
            Z-planes to consider.

    Returns:
        - ```max_channel``` - ```int```.
            Channel to which image with max pixel value corresponds.
        - ```max_z``` - ```int```.
            Z-plane to which image with max pixel value corresponds.
        - ```image``` - ```int [tile_sz x tile_sz]```.
            Corresponding image.
    """
    image_max = np.zeros((len(use_channels), len(use_z)))
    for i in range(len(use_channels)):
        image_max[i, :] = np.max(np.max(utils.nd2.get_image(images, fov, use_channels[i], use_z), axis=0), axis=0)
        # images[get_nd2_index(images, fov, use_channels[j], use_z[i])].max()
    max_channel = use_channels[np.max(image_max, axis=1).argmax()]
    max_z = use_z[np.max(image_max, axis=0).argmax()]
    return max_channel, max_z, utils.nd2.get_image(images, fov, max_channel, max_z)


def get_scale(im_file: str, tilepos_yx_npy: np.ndarray, tilepos_yx_nd2: np.ndarray, use_tiles: List[int],
              use_channels: List[int], use_z: List[int], scale_norm: int,
              filter_kernel: np.ndarray, smooth_kernel: Optional[np.ndarray] = None) -> Tuple[int, int, int, float]:
    """
    Convolves the image for tile ```t```, channel ```c```, z-plane ```z``` with ```filter_kernel```
    then gets the multiplier to apply to filtered nd2 images by dividing ```scale_norm``` by the max value of this
    filtered image.

    Args:
        im_file: File path of nd2 file
        tilepos_yx_npy: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with npy index ```i```. index 0 refers to ```YX = [MaxY,MaxX]```.
        tilepos_yx_nd2: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with nd2 index ```i```. index 0 refers to ```YX = [0,0]```.
        use_tiles: ```int [n_use_tiles]```.
            tiff tile indices to consider when finding tile.
        use_channels: ```int [n_use_channels]```.
            Channels to consider when finding channel.
        use_z: ```int [n_z]```.
            Z-planes to consider when finding z_plane.
        scale_norm: Desired maximum pixel value of npy images. Typical: ```40000```.
        filter_kernel: ```float [ny_kernel x nx_kernel]```.
            Kernel to convolve nd2 data with to produce npy tiles. Typical shape: ```[13 x 13]```.
        smooth_kernel: ```float [ny_smooth x nx_smooth]```.
            2D kernel to smooth filtered image with npy with. Typical shape: ```[3 x 3]```.
            If None, no smoothing is applied

    Returns:
        - ```t``` - ```int```.
            npy tile index (index ```0``` refers to ```tilepos_yx_npy=[MaxY, MaxX]```) scale found from.
        - ```c``` - ```int```.
            Channel scale found from.
        - ```z``` - ```int```.
            Z-plane scale found from.
        - ```scale``` - ```float```.
            Multiplier to apply to filtered nd2 images before saving as npy so full npy ```uint16``` range occupied.
    """
    # tile to get scale from is central tile
    t = select_tile(tilepos_yx_npy, use_tiles)
    images = utils.nd2.load(im_file)
    # find z-plane with max pixel across all channels of tile t
    c, z, image = get_z_plane(images, get_nd2_tile_ind(t, tilepos_yx_nd2, tilepos_yx_npy), use_channels, use_z)
    # convolve_2d image in same way we convolve_2d before saving tiff files
    im_filtered = utils.morphology.convolve_2d(image, filter_kernel)
    if smooth_kernel is not None:
        im_filtered = utils.morphology.imfilter(im_filtered, smooth_kernel, oa=False)
    scale = scale_norm / im_filtered.max()
    return t, c, z, float(scale)
