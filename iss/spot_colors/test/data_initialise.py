from typing import Optional, List
import numpy as np
from ...setup import get_tile_file_names, NotebookPage
from ...utils.npy import save_tile


def get_notebook_pages(tile_dir: str, is_3d: bool, tile_sz: np.ndarray, z_scale: float,
                       n_rounds = 7, n_channels = 7, use_channels: Optional[List] = None):
    """
    Returns notebook pages with required parameters.
    Args:
        tile_dir:
        is_3d:
        tile_sz:
        z_scale:
        use_channels:

    Returns:

    """
    nbp_basic = NotebookPage('basic_info')
    nbp_basic.is_3d = is_3d
    nbp_basic.anchor_round = n_rounds
    nbp_basic.anchor_channel = 1
    nbp_basic.dapi_channel = 0
    nbp_basic.n_rounds = n_rounds
    nbp_basic.use_rounds = np.arange(n_rounds)
    nbp_basic.n_channels = n_channels
    if use_channels is None:
        nbp_basic.use_channels = np.arange(n_channels)
    else:
        nbp_basic.use_channels = np.clip(use_channels, 0, n_channels-1)
    nbp_basic.n_tiles = 1
    nbp_basic.use_tiles = np.array([0])
    nbp_basic.tilepos_yx = np.array([0, 0])
    nbp_basic.tilepos_yx_nd2 = np.array([0, 0])
    nbp_basic.tile_pixel_value_shift = 15000
    nbp_basic.tile_sz = tile_sz[0]
    if len(tile_sz) == 2:
        if not is_3d:
            tile_sz = np.append(tile_sz, 1)
        else:
            raise ValueError('TileSz should be 3D.')
    if not is_3d:
        tile_sz[2] = 1
    nbp_basic.tile_centre = (tile_sz - 1) / 2
    nbp_basic.nz = tile_sz[2]
    # need pixel_size to compute z_scale in get_spot_colors
    nbp_basic.pixel_size_z = 1
    nbp_basic.pixel_size_xy = 1 / z_scale

    nbp_file = NotebookPage('file_names')
    nbp_file.tile_dir = tile_dir
    nbp_file.round = ['round0', 'round1', 'round2', 'round3', 'round4', 'round5', 'round6']
    nbp_file.anchor = ['anchor']
    if is_3d:
        n_channel_npy = nbp_basic.n_channels
    else:
        n_channel_npy = 0
    nbp_file.tile = get_tile_file_names(tile_dir, nbp_file.round + nbp_file.anchor, nbp_basic.n_tiles, n_channel_npy)
    return nbp_file, nbp_basic


def single_random_tile(nbp_basic: NotebookPage, r: int, c: int, tile_sz: np.ndarray):
    """
    Makes a random image that corresponds to a given round and channel.

    Args:
        nbp_basic:
        r:
        c:
        tile_sz:

    Returns:

    """
    if r == nbp_basic.anchor_round:
        use_channels = [val for val in [nbp_basic.dapi_channel, nbp_basic.anchor_channel] if val is not None]
    else:
        use_channels = nbp_basic.use_channels

    if not np.isin(c, use_channels):
        # if not using channel, set to all zeros.
        image = np.zeros(tile_sz, dtype=np.uint16)
    elif r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel:
        # dapi image is already uint16 and has no shift
        image = np.random.randint(0, np.iinfo(np.uint16).max, tile_sz, dtype=np.uint16)
    else:
        # for spot images, need to shift zero and thus make int32. Will shift to uint16 before saving.
        # -nbp_basic.tile_pixel_value_shift is the invalid value hence the +1 so min actual value can't be the
        # invalid value.
        image = np.random.randint(-nbp_basic.tile_pixel_value_shift + 1,
                                  np.iinfo(np.uint16).max - nbp_basic.tile_pixel_value_shift, tile_sz, dtype=np.int32)
    return image


def make_random_tiles(nbp_file: NotebookPage, nbp_basic: NotebookPage, t: int,
                      use_rounds: List[int], tile_sz: np.ndarray):
    """
    Save random tiles to tile directory.

    Args:
        nbp_file: Contains tile
        nbp_basic: Contains anchor_round, anchor_channel, tile_pixel_value_shift, dapi_channel, use_channels, is_3d,
            nz, tile_sz
        t: Tile to make tiffs of all rounds/channels
        use_rounds: Which rounds to make tiffs of.
        tile_sz: YXZ size of tiffs to make.

    Returns:

    """
    if not nbp_basic.is_3d:
        tile_sz = tile_sz[:2]
    for r in use_rounds:
        if not nbp_basic.is_3d:
            all_channel_image = np.zeros((nbp_basic.n_channels, nbp_basic.tile_sz, nbp_basic.tile_sz), dtype=np.int32)
        for c in np.arange(nbp_basic.n_channels):
            image = single_random_tile(nbp_basic, r, c, tile_sz)
            if nbp_basic.is_3d:
                save_tile(nbp_file, nbp_basic, image, t, r, c)
            else:
                all_channel_image[c] = image
        if not nbp_basic.is_3d:
            save_tile(nbp_file, nbp_basic, all_channel_image, t, r)


def get_random_transforms(nbp_basic: NotebookPage, tile_sz: np.ndarray, z_scale: float):
    """
    Get a random transform for each round, channel which is close to identity and with a shift which is less
    than a third of the tile size.

    Args:
        nbp_basic:
        tile_sz:
        z_scale:

    Returns:

    """
    ndim = len(tile_sz)
    if ndim == 3 and tile_sz[2] == 1:
        ndim = 2
    transforms = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels, 4, 3))
    if ndim == 2:
        transforms[:, :, :, 2, 2] = 1
    for t in nbp_basic.use_tiles:
        for r in nbp_basic.use_rounds:
            for c in nbp_basic.use_channels:
                transforms[t, r, c, :ndim, :ndim] = np.eye(ndim) + np.random.uniform(-1e-3, 1e-3, (ndim, ndim))
                for i in range(ndim):
                    transforms[t, r, c, 3, i] = np.random.uniform(-tile_sz[i] / 3, tile_sz[i] / 3)
                    if i == 2:
                        # put z shift in units of yx pixels.
                        transforms[t, r, c, 3, i] = transforms[t, r, c, 3, i] * z_scale
    return transforms
