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
    min_score_auto_param = 5
    max_noise = 3
    # TODO: test with remove from each spot_yx and both.
    # TODO: test with scaling dimension in z direction.

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
            spot_yx = random_spot_yx(n_spots, 2047, 2047)
            transform = np.random.choice(range(-100, 100), 2)
        elif dimensions == 3:
            spot_yx = random_spot_yx(n_spots, 2047, 2047, 29)
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

    def test_2d(self):
        spot_yx, transform_yx, actual_transform = self.get_spots(self.max_noise, 2)
        y_search, x_search, z_search = self.get_random_shift_searches(actual_transform)
        actual_transform = np.pad(actual_transform, (0, 1))
        found_transform, score, score_thresh = compute_shift(spot_yx, transform_yx, self.min_score,
                                                             self.min_score_auto_param, self.shift_score_thresh,
                                                             y_search, x_search, z_search)
        diff = actual_transform.astype(int) - found_transform.astype(int)
        self.assertTrue(np.abs(diff).max() <= 0)

