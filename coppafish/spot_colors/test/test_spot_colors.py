import tempfile
import unittest
import numpy as np
from .data_initialise import get_notebook_pages, make_random_tiles, get_random_transforms
from ..base import apply_transform, get_spot_colors
from .. import base_optimised as optimised
import jax.numpy as jnp


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
