from scipy.spatial import KDTree
from .. import utils
import numpy as np
from typing import Optional


def spot_yxz(spot_details: np.ndarray, tile: int, round: int, channel: int, spot_no: np.ndarray) -> np.ndarray:
    """
    Function which gets yxz positions of spots on a particular ```tile```, ```round```, ```
    channel``` from ```spot_details``` in find_spots notebook page. Initially this just cropped spot_details array
    with 7 columns and n_spots rows. Now spot_details is just an n_spots * 3 column array and tile, round and channel
    are computed from number of spots on each t,r,c

    Args:
        spot_details: ```int16 [n_spots x 3]```.
            ```spot_details[s]``` is ```[tile, round, channel, isolated, y, x, z]``` of spot ```s```.
        tile: Tile of desired spots.
        round: Round of desired spots.
        channel: Channel of desired spots.
        spot_no: num_tile * num_rounds * num_channels array containing num_spots on each [t,r,c]

    Returns:
        - ```spot_yxz``` - ```int16 [n_trc_spots x 3]```.
            yxz coordinates of spots on chosen ```tile```, ```round``` and ```channel```.

    """
    #     Function which gets yxz positions of spots on a particular ```tile```, ```round```,
    #     ```channel``` from ```spot_details``` in find_spots notebook page.

    # spots are read in by looping over rounds, channels, then tiles we need to sum up to but not including the number
    # of spots in all rounds before r, then sum round r with all tiles up to (but not incl) tile t, then sum round r,
    # tile t and all channels up to (but not including) channel c. This gives number of spots found before [t,r,c] ie:
    # start index. To get end_index, just add number of spots on [t,r,c]

    start_index = np.sum(spot_no[:tile, :, :]) + np.sum(spot_no[tile, :round, :]) + np.sum(spot_no[tile, round, :channel])
    end_index = start_index + spot_no[tile, round, channel]

    use = range(start_index, end_index)

    return spot_details[use]


def spot_isolated(isolated_spots: np.ndarray, tile: int, ref_round: int, ref_channel: int, spot_no: np.ndarray) \
        -> np.ndarray:
    """
    Exactly same rational as spot_yxz but now return isolated status of spots in t,r,c

    Args:
        isolated_spots: ```int16 [n_ref_spots x 3]```.
            ```isolated_spots[s]``` is ```true ``` if spot ```s``` is isolated, ```false``` o/w.
        tile: Tile of desired spots.

        spot_no: num_tile * num_rounds * num_channels array containing num_spots on each [t,r,c]

    Returns:
        - ```isolated_spots``` - ```bool [n_ref_spots on this channel * 1]```.
            Isolated status of each reference spot on this tile.

    """
    #     Function which gets yxz positions of spots on a particular ```tile```, ```round```,
    #     ```channel``` from ```spot_details``` in find_spots notebook page.

    # spots are read in by looping over rounds, channels, then tiles we need to sum up to but not including the number
    # of spots in all rounds before r, then sum round r with all tiles up to (but not incl) tile t, then sum round r,
    # tile t and all channels up to (but not including) channel c. This gives number of spots found before [t,r,c] ie:
    # start index. To get end_index, just add number of spots on [t,r,c]

    start_index = np.sum(spot_no[:tile, ref_round, ref_channel])
    end_index = start_index + spot_no[tile, ref_round, ref_channel]

    use = range(start_index, end_index)

    return isolated_spots[use]


def get_isolated(image: np.ndarray, spot_yxz: np.ndarray, thresh: float, radius_inner: float, radius_xy: float,
                 radius_z: Optional[float] = None) -> np.ndarray:
    """
    Determines whether each spot in ```spot_yxz``` is isolated by getting the value of image after annular filtering
    at each location in ```spot_yxz```.

    Args:
        image: ```float [n_y x n_x x n_z]```.
            image spots were found on.
        spot_yxz: ```int [n_peaks x image.ndim]```.
            yx or yxz location of spots found.
            If axis 1 dimension is more than ```image.ndim```, only first ```image.ndim``` dimensions used
            i.e. if supply yxz, with 2d image, only yx position used.
        thresh: Spots are isolated if annulus filtered image at spot location less than this.
        radius_inner: Inner radius of annulus filtering kernel within which values are all zero.
        radius_xy: Outer radius of annulus filtering kernel in xy direction.
        radius_z: Outer radius of annulus filtering kernel in z direction.
            If ```None```, 2D filter is used.

    Returns:
        ```bool [n_peaks]```.
            Whether each spot is isolated or not.

    """
    se = utils.strel.annulus(radius_inner, radius_xy, radius_z)
    # With just coords, takes about 3s for 50 z-planes.
    isolated = utils.morphology.imfilter_coords(image, se, spot_yxz, padding=0, corr_or_conv='corr') / np.sum(se)
    return isolated < thresh


def check_neighbour_intensity(image: np.ndarray, spot_yxz: np.ndarray, thresh: float = 0) -> np.ndarray:
    """
    Checks whether a neighbouring pixel to those indicated in ```spot_yxz``` has intensity less than ```thresh```.
    The idea is that if pixel has very low intensity right next to it, it is probably a spurious spot.

    Args:
        image: ```float [n_y x n_x x n_z]```.
            image spots were found on.
        spot_yxz: ```int [n_peaks x image.ndim]```.
            yx or yxz location of spots found.
            If axis 1 dimension is more than ```image.ndim```, only first ```image.ndim``` dimensions used
            i.e. if supply yxz, with 2d image, only yx position used.
        thresh: Spots are indicated as ```False``` if intensity at neighbour to spot location is less than this.

    Returns:
        ```float [n_peaks]```.
            ```True``` if no neighbours below thresh.
    """
    if image.ndim == 3:
        transforms = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    elif image.ndim == 2:
        transforms = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    else:
        raise ValueError(f"image has to have two or three dimensions but given image has {image.ndim} dimensions.")
    keep = np.zeros((spot_yxz.shape[0], len(transforms)), dtype=bool)
    for i, t in enumerate(transforms):
        mod_spot_yx = spot_yxz + t
        for j in range(image.ndim):
            mod_spot_yx[:, j] = np.clip(mod_spot_yx[:, j], 0, image.shape[j] - 1)
        keep[:, i] = image[tuple([mod_spot_yx[:, j] for j in range(image.ndim)])] > thresh
    return keep.min(axis=1)


def get_isolated_points(spot_yxz: np.ndarray, isolation_dist: float) -> np.ndarray:
    """
    Get the isolated points in a point cloud as those whose neighbour is far.

    Args:
        spot_yxz: ```int [n_peaks x image.ndim]```.
            yx or yxz location of spots found in image.
        isolation_dist: Spots are isolated if nearest neighbour is further away than this.

    Returns:
        ```bool [n_peaks]```. ```True``` for points far from any other point in ```spot_yx```.

    """
    tree = KDTree(spot_yxz)
    # for distances more than isolation_dist, distances will be set to infinity i.e. will be > isolation_dist.
    distances = tree.query(spot_yxz, k=[2], distance_upper_bound=isolation_dist)[0].squeeze()
    return distances > isolation_dist
