from scipy.spatial import KDTree
from .. import utils
import numpy as np
from typing import Optional, Tuple, Union


def spot_yxz(spot_details: np.ndarray, tile: int, round: int, channel: int,
             return_isolated: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Function which gets yxz positions (and whether isolated) of spots on a particular ```tile```, ```round```, ```
    channel``` from ```spot_details``` in find_spots notebook page.

    Args:
        spot_details: ```int16 [n_spots x 7]```.
            ```spot_details[s]``` is ```[tile, round, channel, isolated, y, x, z]``` of spot ```s```.
        tile: Tile of desired spots.
        round: Round of desired spots.
        channel: Channel of desired spots.
        return_isolated: Whether to return isolated status of each spot.

    Returns:
        - ```spot_yxz``` - ```int16 [n_trc_spots x 3]```.
            yxz coordinates of spots on chosen ```tile```, ```round``` and ```channel```.
        - ```spot_isolated``` - ```bool [n_trc_spots]``` (Only returned if ```return_isolated = True```).
            Isolated status (```1``` if isolated, ```0``` if not) of the spots.
    """
    #     Function which gets yxz positions (and whether isolated) of spots on a particular ```tile```, ```round```,
    #     ```channel``` from ```spot_details``` in find_spots notebook page.
    use = np.all((spot_details[:, 0] == tile, spot_details[:, 1] == round, spot_details[:, 2] == channel), axis=0)
    if return_isolated:
        return spot_details[use, 4:], spot_details[use, 3]
    else:
        return spot_details[use, 4:]


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
            mod_spot_yx[:, j] = np.clip(mod_spot_yx[:, j], 0, image.shape[j]-1)
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
