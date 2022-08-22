import unittest
import numpy as np
from ...find_spots.test.random_spot import random_spot_yx, remove_spots, add_noise
from ..shift import compute_shift
from ...utils.base import round_any


class TestShift(unittest.TestCase):
    shift_spacing_yx = 5
    shift_spacing_z = 2
    shift_score_thresh = 2
    min_score = None
    min_score_min_dist = 11
    min_score_max_dist = 20
    min_score_multiplier = 3   # This probably should be less for actual pipeline, about 1.5.
    # For multi-widen, multiplier to be larger here, because few spots so if shift is wrong get very low score
    # but if correct, get very high score.
    min_score_multiplier_multi_widen = 6
    max_noise = 3
    nz_collapse = 30
    tol = 2

    @staticmethod
    def get_spots(max_noise, dimensions=2):
        """
        Makes some random spots, shifts them by random transform and applies noise to shifted spots.

        :param max_noise: noise is added to transformed spots. This indicates maximum amplitude of noise.
            expected to be around 5. Lower value makes it easier.
        :param dimensions: 2 or 3 indicating number of coordinates for each spot.
            If 2, will set z_coordinate to 0 for all spots.
        :return:
            spot_yx: numpy integer array [n_spots x 3]
            transform_yx: numpy integer array [n_spots x 3]
            transform: numpy integer array [transform_y, transform_x (, transform_z)]
        """
        # TODO: comment test functions better
        n_spots = np.random.randint(600, 2000)
        if dimensions == 2:
            spot_yx, n_spots = random_spot_yx(n_spots, 2047, 2047)
            transform = np.random.choice(range(-100, 100), 2)
        elif dimensions == 3:
            spot_yx, n_spots = random_spot_yx(n_spots, 2047, 2047, 29)
            transform = np.random.choice(range(-100, 100), 3)
        else:
            raise ValueError(f"dimensions given was {dimensions} but should be 2 or 3")
        transform_yx = spot_yx + transform
        noise_amplitude = np.random.randint(1, max_noise)
        transform_yx = add_noise(transform_yx, noise_amplitude, True)
        if dimensions == 2:
            spot_yx = np.pad(spot_yx, [(0, 0), (0, 1)])
            transform_yx = np.pad(transform_yx, [(0, 0), (0, 1)])
        return spot_yx, transform_yx, transform

    def get_random_shift_searches(self, transform):
        """
        gets a random selection of shifts in y, x, z which does contain the actual transform given.

        :param transform: numpy integer array [2 or 3,]
        :return:
            y_search: numpy integer array
            x_search: numpy integer array
            z_search: numpy integer array, this is always [0] if given transform only has 2 values.
        """
        y_min = round_any(transform[0] - np.random.randint(5, 18) * self.shift_spacing_yx,
                          self.shift_spacing_yx, 'floor')
        y_max = round_any(transform[0] + np.random.randint(5, 18) * self.shift_spacing_yx,
                          self.shift_spacing_yx, 'ceil') + 1
        y_search = np.arange(y_min, y_max, self.shift_spacing_yx)
        x_min = round_any(transform[1] - np.random.randint(5, 18) * self.shift_spacing_yx,
                          self.shift_spacing_yx, 'floor')
        x_max = round_any(transform[1] + np.random.randint(5, 18) * self.shift_spacing_yx,
                          self.shift_spacing_yx, 'ceil') + 1
        x_search = np.arange(x_min, x_max, self.shift_spacing_yx)
        if transform.shape[0] == 3:
            z_min = round_any(transform[2] - np.random.randint(5, 18) * self.shift_spacing_z,
                              self.shift_spacing_z, 'floor')
            z_max = round_any(transform[2] + np.random.randint(5, 18) * self.shift_spacing_z,
                              self.shift_spacing_z, 'ceil') + 1
            z_search = np.arange(z_min, z_max, self.shift_spacing_z)
            return y_search, x_search, z_search
        else:
            return y_search, x_search, np.arange(1)

    def all_test(self, dimensions, remove=None, widen=0, z_scale=1, multiple_widen=False):
        """

        :param dimensions: 2 or 3 whether to have yx or yxz spot coordinates.
        :param remove:
            'base': a random number of spots will be removed from the base set.
            'transform': a random number of spots will be removed from the transformed set.
            'both': a random number of spots will be removed from the base and transformed set.
            default is None meaning will have 1:1 correspondence between base and transformed spots
                (equal number of both).
        :param widen: if shift not found in initial shift search, will extend shift range by this many values in each
            direction.
        :param multiple_widen: if True, starting shift_range will be quite far off from including actual transform.
            This is to test the while loop.
        :param z_scale: how much to scale z pixel values to make them same units as xy.
        """
        spot_yxz, transform_yxz, actual_transform = self.get_spots(self.max_noise, dimensions)
        y_search, x_search, z_search = self.get_random_shift_searches(actual_transform)
        z_widen = 0
        max_shift_range = [500, 500, 0]
        if widen > 0:
            # Set search range outside actual_transform so widen is tested.
            y_search = y_search[y_search > actual_transform[0] + widen/2 * self.shift_spacing_yx]
            x_search = x_search[x_search < actual_transform[1] - widen/2 * self.shift_spacing_yx]
            if multiple_widen:
                y_search = y_search + widen * self.shift_spacing_yx * np.random.randint(6)
                x_search = x_search - widen * self.shift_spacing_yx * np.random.randint(6)
            if dimensions == 3:
                z_search = z_search[z_search < actual_transform[2] - widen/2 * self.shift_spacing_z]
                z_widen = widen
                max_shift_range[2] = int(np.ptp(z_search) + 1)  # only allow one widening in z.
        if remove == 'base' or 'both':
            spot_yxz = remove_spots(spot_yxz, np.random.randint(spot_yxz.shape[0] / 8, spot_yxz.shape[0] / 2))
        if remove == 'transform' or 'both':
            transform_yxz = remove_spots(transform_yxz, np.random.randint(transform_yxz.shape[0] / 8,
                                                                          transform_yxz.shape[0] / 2))
        if multiple_widen:
            min_score_multiplier = self.min_score_multiplier_multi_widen
        else:
            min_score_multiplier = self.min_score_multiplier
        if dimensions == 2:
            actual_transform = np.pad(actual_transform, (0, 1))
            nz_collapse = None
        else:
            nz_collapse = self.nz_collapse
        found_transform, score, score_thresh, debug_info = \
            compute_shift(spot_yxz, transform_yxz, self.min_score, min_score_multiplier, self.min_score_min_dist,
                          self.min_score_max_dist, self.shift_score_thresh, y_search, x_search, None,
                          [widen, widen, z_widen], max_shift_range, z_scale, nz_collapse, self.shift_spacing_z)
        diff = actual_transform.astype(int) - found_transform.astype(int)
        self.assertTrue(np.abs(diff).max() <= self.tol)

    def test_2d(self):
        self.all_test(2)

    def test_2d_remove1(self):
        self.all_test(2, 'base')

    def test_2d_remove2(self):
        self.all_test(2, 'transform')

    def test_2d_remove3(self):
        self.all_test(2, 'both')

    def test_2d_widen(self):
        self.all_test(2, 'both', 5)

    def test_2d_multiple_widen(self):
        self.all_test(2, 'both', 5, multiple_widen=True)

    def test_3d(self):
        self.all_test(3)

    def test_3d_remove1(self):
        self.all_test(3, 'base')

    def test_3d_remove2(self):
        # I am using small scales as to not affect score much but to
        # ensure function can deal with scale parameter.
        # For actual data, z_scale is around 6.
        self.all_test(3, 'transform', z_scale=1.000036899402)

    def test_3d_remove3(self):
        self.all_test(3, 'both')

    def test_3d_widen(self):
        self.all_test(3, 'both', 5, 3.6)

    def test_3d_multiple_widen(self):
        self.all_test(3, 'both', 5, 1.005567259, True)
