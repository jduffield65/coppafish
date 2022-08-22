import tempfile
import unittest
import numpy as np
from ... import utils
from ...spot_colors.test import get_notebook_pages, single_random_tile


class TestNPY(unittest.TestCase):
    MinYX = 50
    MaxYX = 300
    MinZ = 5
    MaxZ = 12
    MinSpots = 100
    MaxSpots = 1000
    N_Rounds = 7
    Anchor_Round = 7
    N_Channels = 7
    Z_Scale = 6.3
    t = 0
    tol = 0

    def all_test(self, r: int, is_3d: bool = True, use_channels=None):
        tile_sz = np.zeros(3, dtype=int)
        tile_sz[:2] = np.random.randint(self.MinYX, self.MaxYX)
        if is_3d:
            tile_sz[2] = np.random.randint(self.MinZ, self.MaxZ)
        else:
            tile_sz[2] = 1
        with tempfile.TemporaryDirectory() as tile_dir:
            nbp_file, nbp_basic = get_notebook_pages(tile_dir, is_3d, tile_sz, self.Z_Scale, self.N_Rounds,
                                                     self.N_Channels, use_channels)
            if r == nbp_basic.anchor_round:
                use_channels = [val for val in [nbp_basic.dapi_channel, nbp_basic.anchor_channel] if val
                                is not None]
            else:
                use_channels = nbp_basic.use_channels
            invalid_value = -nbp_basic.tile_pixel_value_shift
            if not is_3d:
                tile_sz = tile_sz[:2]
                image = np.zeros((nbp_basic.n_channels,) + tuple(tile_sz), dtype=np.int32)
                for c in range(nbp_basic.n_channels):
                    image[c] = single_random_tile(nbp_basic, r, c, tile_sz)
                utils.npy.save_tile(nbp_file, nbp_basic, image.copy(), self.t, r)
                loaded_image = np.zeros_like(image, dtype=np.int32)
                for c in range(nbp_basic.n_channels):
                    loaded_image[c] = utils.npy.load_tile(nbp_file, nbp_basic, self.t, r, c)
                diff = image - loaded_image
                diff = diff[loaded_image != invalid_value]
                self.assertTrue(np.abs(diff).max() <= self.tol)
                # check only get invalid value for un-used channels.
                n_invalid = np.sum(loaded_image == invalid_value, axis=(1, 2))
                self.assertTrue((n_invalid[use_channels] == 0).all())
                un_used_channels = np.setdiff1d(np.arange(nbp_basic.n_channels), use_channels)
                if len(un_used_channels) > 0:
                    self.assertTrue((n_invalid[un_used_channels] == np.prod(image[0].shape)).all())
            else:
                for c in range(nbp_basic.n_channels):
                    image = single_random_tile(nbp_basic, r, c, tile_sz)
                    utils.npy.save_tile(nbp_file, nbp_basic, image.copy(), self.t, r, c)
                    loaded_image = utils.npy.load_tile(nbp_file, nbp_basic, self.t, r, c)
                    diff = image - loaded_image
                    diff = diff[loaded_image != invalid_value]
                    self.assertTrue(np.abs(diff).max() <= self.tol)
                    if np.isin(c, use_channels):
                        # no invalid value if use channel
                        self.assertTrue((np.sum(loaded_image == invalid_value) == 0).all())

    def load_subset_all(self, r, channel, is_3d, tile_sz, yxz, apply_shift):
        try:
            with tempfile.TemporaryDirectory() as tile_dir:
                nbp_file, nbp_basic = get_notebook_pages(tile_dir, is_3d, tile_sz, self.Z_Scale, self.N_Rounds,
                                                         self.N_Channels)
                if not is_3d:
                    tile_sz = tile_sz[:2]
                    image = np.zeros((nbp_basic.n_channels,) + tuple(tile_sz), dtype=np.int32)
                    for c in range(nbp_basic.n_channels):
                        image[c] = single_random_tile(nbp_basic, r, c, tile_sz)
                    utils.npy.save_tile(nbp_file, nbp_basic, image.copy(), self.t, r)
                    if isinstance(yxz, list):
                        coord_index = np.ix_(yxz[0], yxz[1])
                    else:
                        coord_index = tuple(np.asarray(yxz[:, i]) for i in range(2))
                    im_subset = image[channel][coord_index]
                else:
                    image = single_random_tile(nbp_basic, r, channel, tile_sz)
                    utils.npy.save_tile(nbp_file, nbp_basic, image.copy(), self.t, r, channel)
                    if yxz[0] is None:
                        # Case where want to load in entire z-plane
                        im_subset = image[:, :, yxz[2]]
                        if isinstance(yxz[2], list):
                            im_subset = im_subset[:, :, 0]
                    else:
                        if isinstance(yxz, list):
                            coord_index = np.ix_(yxz[0], yxz[1], yxz[2])
                            im_subset = image[coord_index]
                            # check shape correct
                            self.assertTrue(im_subset.shape == tuple([len(val) for val in yxz]))
                        else:
                            coord_index = tuple(np.asarray(yxz[:, i]) for i in range(3))
                            im_subset = image[coord_index]
                            # check shape correct
                            self.assertTrue(im_subset.shape == yxz.shape[:1])
                if not apply_shift:
                    im_subset = im_subset + nbp_basic.tile_pixel_value_shift
                im_subset_load = utils.npy.load_tile(nbp_file, nbp_basic, self.t, r, channel, yxz, apply_shift)
                diff = im_subset - im_subset_load
                self.assertTrue(np.abs(diff).max() <= self.tol)
        except (PermissionError, NotADirectoryError) as e:
            # In windows, can get a permission error here when trying to access temporary directory
            self.assertTrue(True)

    def test_imaging_2d(self):
        # Test can load in a whole imaging round/channel image
        self.all_test(np.random.randint(self.N_Rounds), False)

    def test_imaging_3d(self):
        # Test can load in a whole imaging round/channel image
        self.all_test(np.random.randint(self.N_Rounds), True)

    def test_anchor_2d(self):
        # Test can load in a whole anchor round/channel and dapi round/channel image
        self.all_test(self.Anchor_Round, False)

    def test_anchor_3d(self):
        # Test can load in a whole anchor round/channel and dapi round/channel image
        self.all_test(self.Anchor_Round, True)

    def test_unused_channel_2d(self):
        # Test can load in a channel which was not used.
        n_unused = np.random.randint(3,6)
        self.all_test(np.random.randint(self.N_Rounds), False,
                      np.sort(np.random.choice(np.arange(self.N_Channels), self.N_Channels-n_unused, False)))

    def test_unused_channel_3d(self):
        # Test can load in a channel which was not used.
        n_unused = np.random.randint(3, 6)
        self.all_test(np.random.randint(self.N_Rounds), True,
                      np.sort(np.random.choice(np.arange(self.N_Channels), self.N_Channels - n_unused, False)))

    def test_single_z_3d(self):
        # Test can load in a single z plane of an imaging round/channel image
        tile_sz = np.zeros(3, dtype=int)
        tile_sz[:2] = np.random.randint(self.MinYX, self.MaxYX)
        tile_sz[2] = np.random.randint(self.MinZ, self.MaxZ)
        self.load_subset_all(np.random.randint(self.N_Rounds), np.random.randint(self.N_Channels), True,
                             tile_sz, [None, None, np.random.randint(tile_sz[2])], np.random.randint(2, dtype=bool))

    def test_multi_z_3d(self):
        # Test can load in multiple z planes of an imaging round/channel image
        tile_sz = np.zeros(3, dtype=int)
        tile_sz[:2] = np.random.randint(self.MinYX, self.MaxYX)
        tile_sz[2] = np.random.randint(self.MinZ, self.MaxZ)
        n_z_planes = np.random.randint(1, tile_sz[2])
        self.load_subset_all(np.random.randint(self.N_Rounds), np.random.randint(self.N_Channels), True,
                             tile_sz, [None, None, np.random.randint(0, tile_sz[2], n_z_planes)],
                             np.random.randint(2, dtype=bool))

    def test_subset_2d(self):
        # Test can load in a subset of an imaging round/channel image
        tile_sz = np.zeros(3, dtype=int)
        tile_sz[:2] = np.random.randint(self.MinYX, self.MaxYX)
        tile_sz[2] = 1
        y = np.sort(np.random.choice(np.arange(tile_sz[0]), np.random.randint(5, tile_sz[0] - 10), False))
        x = np.sort(np.random.choice(np.arange(tile_sz[1]), np.random.randint(5, tile_sz[1] - 10), False))
        self.load_subset_all(np.random.randint(self.N_Rounds), np.random.randint(self.N_Channels), False,
                             tile_sz, [y, x], np.random.randint(2, dtype=bool))

    def test_subset_3d(self):
        # Test can load in a subset of an imaging round/channel image
        tile_sz = np.zeros(3, dtype=int)
        tile_sz[:2] = np.random.randint(self.MinYX, self.MaxYX)
        tile_sz[2] = np.random.randint(self.MinZ, self.MaxZ)
        y = np.sort(np.random.choice(np.arange(tile_sz[0]), np.random.randint(5, tile_sz[0] - 10), False))
        x = np.sort(np.random.choice(np.arange(tile_sz[1]), np.random.randint(5, tile_sz[1] - 10), False))
        z = np.sort(np.random.choice(np.arange(tile_sz[2]), np.random.randint(2, tile_sz[2] - 1), False))
        self.load_subset_all(np.random.randint(self.N_Rounds), np.random.randint(self.N_Channels), True,
                             tile_sz, [y, x, z], np.random.randint(2, dtype=bool))

    def test_pixel_values_2d(self):
        # Test can load in the pixel values at specific coordinates of an imaging round/channel image
        tile_sz = np.zeros(3, dtype=int)
        tile_sz[:2] = np.random.randint(self.MinYX, self.MaxYX)
        tile_sz[2] = 1
        n_spots = np.random.randint(self.MinSpots, self.MaxSpots)
        spot_yxz = np.zeros((n_spots, 2), dtype=int)
        for i in range(2):
            spot_yxz[:, i] = np.random.randint(0, tile_sz[i] - 1, n_spots)
        self.load_subset_all(np.random.randint(self.N_Rounds), np.random.randint(self.N_Channels), False,
                             tile_sz, spot_yxz, np.random.randint(2, dtype=bool))

    def test_pixel_values_3d(self):
        # Test can load in the pixel values at specific coordinates of an imaging round/channel image
        tile_sz = np.zeros(3, dtype=int)
        tile_sz[:2] = np.random.randint(self.MinYX, self.MaxYX)
        tile_sz[2] = np.random.randint(self.MinZ, self.MaxZ)
        n_spots = np.random.randint(self.MinSpots, self.MaxSpots)
        spot_yxz = np.zeros((n_spots, 3), dtype=int)
        for i in range(3):
            spot_yxz[:, i] = np.random.randint(0, tile_sz[i] - 1, n_spots)
        self.load_subset_all(np.random.randint(self.N_Rounds), np.random.randint(self.N_Channels), True,
                             tile_sz, spot_yxz, np.random.randint(2, dtype=bool))
