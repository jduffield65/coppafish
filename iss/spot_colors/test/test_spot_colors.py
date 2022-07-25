import tempfile
import unittest
import numpy as np
from ...utils.npy import save_tile
from ...setup.notebook import NotebookPage
from ...setup.tile_details import get_tile_file_names
from ..base import apply_transform, get_spot_colors
from .. import base_optimised as optimised
from typing import List, Optional
import jax.numpy as jnp


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


class TestSpotColors(unittest.TestCase):
    """
    Test whether spot colors read in using jax is the same as with numpy.
    """
    MinYX = 50
    MaxYX = 300
    MinZ = 3
    MaxZ = 12
    MinSpots = 100
    MaxSpots = 1000
    MinRounds = 3
    Z_Scale = 5.6

    def all_test(self, is_3d: bool, single_z: bool=False):
        tile_sz = np.zeros(3, dtype=int)
        tile_sz[:2] = np.random.randint(self.MinYX, self.MaxYX)
        if is_3d:
            tile_sz[2] = np.random.randint(self.MinZ, self.MaxZ)
        else:
            tile_sz[2] = 1
        n_spots = np.random.randint(self.MinSpots, self.MaxSpots)
        spot_yxz = np.zeros((n_spots, 3), dtype=int)
        for i in np.where(tile_sz > 1)[0]:
            if single_z and i == 2:
                spot_yxz[:, i] = np.random.randint(0, tile_sz[i] - 1)
            else:
                spot_yxz[:, i] = np.random.randint(0, tile_sz[i] - 1, n_spots)

        with tempfile.TemporaryDirectory() as tile_dir:
            nbp_file, nbp_basic = get_notebook_pages(tile_dir, is_3d, tile_sz, self.Z_Scale)
            invalid_value = -nbp_basic.tile_pixel_value_shift
            n_use_rounds = np.random.randint(self.MinRounds, nbp_basic.n_rounds)
            n_use_channels = np.random.randint(self.MinRounds, nbp_basic.n_channels)
            use_rounds = np.sort(np.random.choice(np.arange(nbp_basic.n_rounds), n_use_rounds, False))
            use_channels = np.sort(np.random.choice(np.arange(nbp_basic.n_channels), n_use_channels, False))
            transforms = get_random_transforms(nbp_basic, tile_sz, self.Z_Scale)
            t = 0
            # spot_no, round, channel which have different yxz_transform between jax and python due to rounding.
            transform_src_diff = np.zeros((0, 3), dtype=int)
            for r in use_rounds:
                for c in use_channels:
                    yxz_transform = apply_transform(spot_yxz, transforms[t, r, c], nbp_basic.tile_centre, self.Z_Scale,
                                                    tile_sz)[0]
                    yxz_transform_jax = np.asarray(
                        optimised.apply_transform(jnp.array(spot_yxz), jnp.array(transforms[t, r, c]),
                                                  jnp.array(nbp_basic.tile_centre), self.Z_Scale, jnp.asarray(tile_sz))[0])
                    diff = yxz_transform_jax - yxz_transform
                    # tolerance of 1 as expect difference due to rounding error will be just 1.
                    # expect rounding error because jax is float32 while python is float64.
                    self.assertTrue(np.max(np.abs(diff)) <= 1)
                    if np.max(np.abs(diff)) > 0:
                        # Record which spots/rounds/channels have differing transforms.
                        wrong_spot_no = np.where(diff != 0)[0]
                        n_wrong = len(wrong_spot_no)
                        src_wrong = np.append(wrong_spot_no[:, np.newaxis], np.tile(np.array([r, c]), (n_wrong, 1)),
                                              axis=1)
                        transform_src_diff = np.append(transform_src_diff, src_wrong, axis=0)
            make_random_tiles(nbp_file, nbp_basic, t, use_rounds, tile_sz)
            spot_colors = np.full((n_spots, nbp_basic.n_rounds, nbp_basic.n_channels), invalid_value, dtype=np.int32)
            spot_colors[np.ix_(np.arange(n_spots), use_rounds, use_channels)] = \
                get_spot_colors(spot_yxz, t, transforms, nbp_file, nbp_basic, use_rounds, use_channels)
            spot_colors_jax = np.full((n_spots, nbp_basic.n_rounds, nbp_basic.n_channels), invalid_value,
                                      dtype=np.int32)
            spot_colors_jax[np.ix_(np.arange(n_spots), use_rounds, use_channels)] = \
                optimised.get_spot_colors(jnp.array(spot_yxz), t, jnp.array(transforms), nbp_file, nbp_basic,
                                          use_rounds, use_channels)
            diff = spot_colors - spot_colors_jax
            n_wrong_colors = transform_src_diff.shape[0]
            if n_wrong_colors > 0:
                # If some transforms differ by one pixel, colors will differ too.
                expected_wrong_diff = diff[transform_src_diff[:, 0], transform_src_diff[:, 1], transform_src_diff[:, 2]]
                # Can still get same color if transform was out of bounds so got invalid. Correct for this.
                valid = spot_colors_jax[transform_src_diff[:, 0], transform_src_diff[:, 1],
                                        transform_src_diff[:, 2]] != invalid_value
                valid = np.logical_and(valid, spot_colors[transform_src_diff[:, 0], transform_src_diff[:, 1],
                                                          transform_src_diff[:, 2]] != invalid_value)
                n_wrong_colors = np.sum(valid)
                self.assertTrue(np.sum(expected_wrong_diff[valid] != 0) == n_wrong_colors)
            self.assertTrue(np.sum(diff != 0) == n_wrong_colors)

    def test_2d(self):
        self.all_test(False)

    def test_3d(self):
        self.all_test(True)

    def test_3d_single_z(self):
        # Quite often run case of 3d pipeline but all spots on same z-plane. Check this works.
        self.all_test(True, True)
