import unittest
import numpy as np
from iss.find_spots.test.random_spot import random_spot_yx, remove_spots, add_noise
from iss.stitch.shift import get_best_shift
from iss.utils.base import round_any


class TestShift(unittest.TestCase):
    shift_spacing_yx = 5
    shift_spacing_z = 2
    shift_score_thresh = 2
    # TODO: test with remove from each spot_yx and both.
    # TODO: test with scaling dimension in z direction.

    @staticmethod
    def get_spots(dimensions=2):
        # TODO: comment test functions better
        n_spots = np.random.randint(600, 2000)
        if dimensions == 2:
            spot_yx = random_spot_yx(n_spots, 2047, 2047)
            transform = np.random.choice(range(-100, 100), 2)
        elif dimensions == 3:
            spot_yx = random_spot_yx(n_spots, 2047, 2047, 29)
            transform = np.random.choice(range(-100, 100), 3)
        transform_yx = spot_yx + transform
        noise_amplitude = np.random.randint(1, 8)
        transform_yx = add_noise(transform_yx, noise_amplitude, True)
        return spot_yx, transform_yx, transform

    def get_random_shift_searches(self, transform):
        """

        :param transform: numpy integer array [2 or 3,]
        :return:
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
            return y_search, x_search

    def test_2d(self):
        spot_yx, transform_yx, actual_transform = self.get_spots(2)
        y_search, x_search = self.get_random_shift_searches(actual_transform)
        found_transform, score, score_median, score_iqr = get_best_shift(spot_yx, transform_yx, self.shift_score_thresh,
                                                                         y_search, x_search)
        diff = actual_transform.astype(int) - found_transform.astype(int)
        self.assertTrue(np.abs(diff).max() <= 0)



