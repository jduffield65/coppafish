from scipy.spatial import KDTree
from .. import utils
import numpy as np
from typing import Optional, Tuple, Union
from ..setup import Notebook
import warnings


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


def check_n_spots(nb: Notebook):
    """
    This checks that a decent number of spots are detected on:

    * Each channel across all rounds and tiles.
    * Each tile across all rounds and channels.
    * Each round across all tile and channels.

    An error will be raised if any of these conditions are violated.

    `config['find_spots']['n_spots_warn_fraction']` and `config['find_spots']['n_spots_error_fraction']`
    are the parameters which determine if warnings/errors will be raised.

    Args:
        nb: *Notebook* containing `find_spots` page.

    """
    # TODO: show example of what error looks like in the docs
    config = nb.get_config()['find_spots']
    if nb.basic_info.is_3d:
        n_spots_warn = config['n_spots_warn_fraction'] * config['max_spots_3d'] * nb.basic_info.nz
    else:
        n_spots_warn = config['n_spots_warn_fraction'] * config['max_spots_2d']
    n_spots_warn = int(np.ceil(n_spots_warn))
    use_tiles = np.asarray(nb.basic_info.use_tiles)
    use_rounds = np.asarray(nb.basic_info.use_rounds)  # don't consider anchor in this analysis
    use_channels = np.asarray(nb.basic_info.use_channels)
    spot_no = nb.find_spots.spot_no[np.ix_(use_tiles, use_rounds, use_channels)]
    error_message = ""

    # Consider bad channels first as most likely to have consistently low spot counts in a channel
    n_images = len(use_tiles) * len(use_rounds)
    n_images_error = int(np.floor(n_images * config['n_spots_error_fraction']))
    n_bad_images = np.zeros(len(use_channels), dtype=int)
    for c in range(len(use_channels)):
        bad_images = np.vstack(np.where(spot_no[:, :, c] < n_spots_warn)).T
        n_bad_images[c] = bad_images.shape[0]
        if n_bad_images[c] > 0:
            bad_images[:, 0] = use_tiles[bad_images[:, 0]]
            bad_images[:, 1] = use_rounds[bad_images[:, 1]]
            warnings.warn(f"\nChannel {use_channels[c]} - {n_bad_images[c]} tiles/rounds with n_spots < {n_spots_warn}:"
                          f"\n{bad_images}")

    fail_inds = np.where(n_bad_images >= n_images_error)[0]
    if len(fail_inds) > 0:
        error_message = error_message + f"\nChannels that failed: {use_channels[fail_inds]}\n" \
                                        f"This is because out of {n_images} tiles/rounds, these channels had " \
                                        f"respectively:\n{n_bad_images[fail_inds]}\ntiles/rounds with " \
                                        f"n_spots < {n_spots_warn}. These are all more than the error threshold of " \
                                        f"{n_images_error}.\nConsider removing these from use_channels."
        # don't consider failed channels for subsequent warnings/errors
        use_channels = np.setdiff1d(use_channels, use_channels[fail_inds])
        spot_no = nb.find_spots.spot_no[np.ix_(use_tiles, use_rounds, use_channels)]

    # Consider bad tiles next as second most likely to have consistently low spot counts in a tile
    n_images = len(use_channels) * len(use_rounds)
    n_images_error = int(np.floor(n_images * config['n_spots_error_fraction']))
    n_bad_images = np.zeros(len(use_tiles), dtype=int)
    for t in range(len(use_tiles)):
        bad_images = np.vstack(np.where(spot_no[t] < n_spots_warn)).T
        n_bad_images[t] = bad_images.shape[0]
    fail_inds = np.where(n_bad_images >= n_images_error)[0]
    if len(fail_inds) > 0:
        error_message = error_message + f"\nTiles that failed: {use_tiles[fail_inds]}\n" \
                                        f"This is because out of {n_images} rounds/channels, these tiles had " \
                                        f"respectively:\n{n_bad_images[fail_inds]}\nrounds/channels with " \
                                        f"n_spots < {n_spots_warn}. These are all more than the error threshold of " \
                                        f"{n_images_error}.\nConsider removing these from use_tiles."
        # don't consider failed channels for subsequent warnings/errors
        use_tiles = np.setdiff1d(use_tiles, use_tiles[fail_inds])
        spot_no = nb.find_spots.spot_no[np.ix_(use_tiles, use_rounds, use_channels)]

    # Consider bad rounds last as least likely to have consistently low spot counts in a round
    n_images = len(use_channels) * len(use_tiles)
    n_images_error = int(np.floor(n_images * config['n_spots_error_fraction']))
    n_bad_images = np.zeros(len(use_rounds), dtype=int)
    for r in range(len(use_rounds)):
        bad_images = np.vstack(np.where(spot_no[:, r] < n_spots_warn)).T
        n_bad_images[r] = bad_images.shape[0]
    fail_inds = np.where(n_bad_images >= n_images_error)[0]
    if len(fail_inds) > 0:
        error_message = error_message + f"\nRounds that failed: {use_rounds[fail_inds]}\n" \
                                        f"This is because out of {n_images} tiles/channels, these tiles had " \
                                        f"respectively:\n{n_bad_images[fail_inds]}\ntiles/channels with " \
                                        f"n_spots < {n_spots_warn}. These are all more than the error threshold of " \
                                        f"{n_images_error}.\nConsider removing these from use_rounds."

    # Consider anchor
    if nb.basic_info.use_anchor:
        spot_no = nb.find_spots.spot_no[use_tiles, nb.basic_info.anchor_round, nb.basic_info.anchor_channel]
        n_images = len(use_tiles)
        n_images_error = int(np.floor(n_images * config['n_spots_error_fraction']))
        bad_images = np.where(spot_no < n_spots_warn)[0]
        n_bad_images = len(bad_images)
        if n_bad_images > 0:
            bad_images = use_tiles[bad_images]
            warnings.warn(
                f"\nAnchor - {n_bad_images} tiles with n_spots < {n_spots_warn}:\n"
                f"{bad_images}")

        if n_bad_images >= n_images_error:
            error_message = error_message + f"\nAnchor - tiles {bad_images} all had n_spots < {n_spots_warn}. " \
                                            f"{n_bad_images}/{n_images} tiles failed which is more than the " \
                                            f"error threshold of {n_images_error}.\n" \
                                            f"Consider removing these tiles from use_tiles."

    if len(error_message) > 0:
        error_message = error_message + f"\nThe function iss.plot.view_find_spots may be useful for investigating " \
                                        f"why the above tiles/rounds/channels had so few spots detected."
        raise ValueError(error_message)
