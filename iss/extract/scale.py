import numpy as np
from ..utils.nd2 import get_nd2_tile_ind
from .. import utils
from ..setup import NotebookPage
from typing import List, Tuple, Optional
import nd2


def central_tile(tilepos_yx: np.ndarray, use_tiles: List[int]) -> int:
    """
    returns tile in use_tiles closest to centre.

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


def get_z_plane(nbp_file: NotebookPage, nbp_basic: NotebookPage, r: int, t: int, use_channels: List[int],
                use_z: List[int]) -> Tuple[int, int, np.ndarray]:
    """
    Finds z plane and channel that has maximum pixel value for given round and tile.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        r: Round to consider.
        t: npy tile index (index ```0``` refers to ```tilepos_yx_npy=[MaxY, MaxX]```) to find z-plane from.
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
    round_dask_array = utils.raw.load(nbp_file, nbp_basic, r=r)
    image_max = np.zeros((len(use_channels), len(use_z)))
    for i in range(len(use_channels)):
        image_max[i, :] = np.max(np.max(utils.raw.load(nbp_file, nbp_basic, round_dask_array, r,
                                                       t, use_channels[i], use_z), axis=0), axis=0)
    max_channel = use_channels[np.max(image_max, axis=1).argmax()]
    max_z = use_z[np.max(image_max, axis=0).argmax()]
    return max_channel, max_z, utils.raw.load(nbp_file, nbp_basic, round_dask_array, r, t, max_channel, max_z)


def get_scale(nbp_file: NotebookPage, nbp_basic: NotebookPage, r: int, use_tiles: List[int],
              use_channels: List[int], use_z: List[int], scale_norm: int,
              filter_kernel: np.ndarray, smooth_kernel: Optional[np.ndarray] = None) -> Tuple[int, int, int, float]:
    """
    Convolves the image for tile ```t```, channel ```c```, z-plane ```z``` with ```filter_kernel```
    then gets the multiplier to apply to filtered nd2 images by dividing ```scale_norm``` by the max value of this
    filtered image.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        r: Round to get `scale` from.
            Should be 0 to determine `scale` and the anchor round (last round) to determine `scale_anchor`.
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
    t = central_tile(nbp_basic.tilepos_yx, use_tiles)
    # find z-plane with max pixel across all channels of tile t
    c, z, image = get_z_plane(nbp_file, nbp_basic, r, t, use_channels, use_z)
    # convolve_2d image in same way we convolve_2d before saving tiff files
    im_filtered = utils.morphology.convolve_2d(image, filter_kernel)
    if smooth_kernel is not None:
        im_filtered = utils.morphology.imfilter(im_filtered, smooth_kernel, oa=False)
    scale = scale_norm / im_filtered.max()
    return t, c, z, float(scale)
