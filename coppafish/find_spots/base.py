from scipy.spatial import KDTree
from .. import utils
from ..setup import NotebookPage
import numpy as np
import os
from typing import Optional


def spot_yxz(local_yxz: np.ndarray, tile: int, round: int, channel: int, spot_no: np.ndarray) -> np.ndarray:
    """
    Function which gets yxz positions of spots on a particular ```tile```, ```round```, ```
    channel``` from ```local_yxz``` in find_spots notebook page. Initially this just cropped spot_details array
    with 7 columns and n_spots rows. Now spot_details is just an n_spots * 3 column array and tile, round and channel
    are computed from number of spots on each t,r,c

    Args:
        local_yxz: ```int16 [n_spots x 3]```.
            ```local_yxz[s]``` is ```[tile, round, channel, isolated, y, x, z]``` of spot ```s```.
        tile: Tile of desired spots.
        round: Round of desired spots.
        channel: Channel of desired spots.
        spot_no: num_tile * num_rounds * num_channels array containing num_spots on each [t,r,c]

    Returns:
        - ```spot_yxz``` - ```int16 [n_trc_spots x 3]```.
            yxz coordinates of spots on chosen ```tile```, ```round``` and ```channel```.

    """
    #     Function which gets yxz positions of spots on a particular ```tile```, ```round```,
    #     ```channel``` from ```local_yxz``` in find_spots notebook page.

    # spots are read in by looping over rounds, channels, then tiles we need to sum up to but not including the number
    # of spots in all rounds before r, then sum round r with all tiles up to (but not incl) tile t, then sum round r,
    # tile t and all channels up to (but not including) channel c. This gives number of spots found before [t,r,c] ie:
    # start index. To get end_index, just add number of spots on [t,r,c]

    start_index = np.sum(spot_no[:tile, :, :]) + np.sum(spot_no[tile, :round, :]) + np.sum(spot_no[tile, round, :channel])
    end_index = start_index + spot_no[tile, round, channel]

    use = range(start_index, end_index)

    return local_yxz[use]


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


def load_spot_info(nbp_file: NotebookPage, nbp_basic: NotebookPage) -> dict:
    """
    Loads spot info from spot_info_dir. If spot_info_dir does not exist, returns dict with empty lists.

    Args:
        nbp_file: File names notebook page.
        nbp_basic: Basic info notebook page.

    Returns:
        spot_info: Dictionary with following 4 keys
        * spot_yxz: [n_spots x 3] int array of yxz positions of spots
        * spot_no: [n_tiles x n_rounds x n_channels] int array of number of spots in each tile, round, channel
        * isolated: [n_anchor_spots] bool array indicating whether each anchor spot is isolated
        * completed: [n_tiles x n_rounds x n_channels] bool array indicating whether spot finding has been completed
    """
    if os.path.isfile(nbp_file.spot_details_info):
        raw = np.load(nbp_file.spot_details_info, allow_pickle=True)
        spot_info = {'spot_yxz': raw.f.arr_0, 'spot_no': raw.f.arr_1, 'isolated': raw.f.arr_2,
                     'completed': raw.f.arr_3}
    else:
        spot_info = {'spot_yxz': np.zeros((0, 3), dtype=np.int32),
                     'spot_no': np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
                                          nbp_basic.n_channels), dtype=np.uint16),
                     'isolated': np.zeros((0), dtype=bool),
                     'completed': np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
                                            nbp_basic.n_channels), dtype=bool)}
    return spot_info


def filter_intense_spots(local_yxz: np.ndarray, spot_intensity: np.ndarray, n_z: int,
                         max_spots: int = 500) -> np.ndarray:
    """
    Filters spots by intensity. For each z plane, keeps only the top max_spots spots.
    Args:
        local_yxz: [n_spots x 3] int array of yxz positions of spots
        spot_intensity: [n_spots] float array of spot intensities
        max_spots: int indicating maximum number of spots to keep per z plane
        n_z: int indicating number of z planes

    Returns:
        local_yxz: [n_spots_keep x 3] int array of yxz positions of spots
    """
    keep = np.ones(local_yxz.shape[0], dtype=bool)
    # Loop over each z plane and keep only the top max_spots spots
    for z in range(n_z):
        # If the number of spots on this z-plane is > max_spots (500 by default for 3D) then we
        # set the intensity threshold to the 500th most intense spot and take the top 500 values
        if np.sum(local_yxz[:, 2] == z) > max_spots:
            intensity_thresh = np.sort(spot_intensity[local_yxz[:, 2] == z])[-max_spots]
            keep[np.logical_and(local_yxz[:, 2] == z, spot_intensity < intensity_thresh)] = False

    return local_yxz[keep]