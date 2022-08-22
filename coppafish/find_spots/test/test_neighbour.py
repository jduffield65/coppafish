import unittest
import os
import numpy as np
from ..base import check_neighbour_intensity
from math import ceil
from .random_spot import random_spot_yx, find_isolated_spots


class TestNeighbour(unittest.TestCase):
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10
    transforms_2d = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    transforms_3d = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

    @staticmethod
    def get_initial_info(dimensions=2):
        """
        get array of random size with all pixel values above thresh.

        :param dimensions: whether to make 2 or 3 dimensional image.
        :return:
        """
        n_spots = np.random.randint(20, 400)
        thresh = np.random.uniform(-50, 50)
        if dimensions == 2:
            array = np.random.randint(ceil(thresh), 10000, (np.random.randint(n_spots, 1000),
                                                            np.random.randint(n_spots, 1000)))
        elif dimensions == 3:
            array = np.random.randint(ceil(thresh), 10000, (np.random.randint(n_spots, 500),
                                                            np.random.randint(n_spots, 500),
                                                            np.random.randint(2, 29)))
        return array, n_spots, thresh

    @staticmethod
    def get_spots(array, n_spots, edges=False):
        """
        get yx location of spots in array.
        All will not be at any edge if edges is False.
        All will be at one edge if edges is True.

        :param array:
        :param n_spots:
        :param edges:
        :return:
        """
        if array.ndim == 2:
            max_z = None
        else:
            max_z = array.shape[2] - 1
        spot_yx, n_spots = random_spot_yx(n_spots, array.shape[0] - 3, array.shape[1] - 3, max_z, min_y=1, min_x=1, min_z=1)
        if edges:
            # make all spots be at one edge
            dim_to_make_edge = np.random.randint(0, array.ndim, n_spots)
            dim_direction = np.random.choice([0, -1], n_spots)
            spot_yx[tuple([np.arange(n_spots), dim_to_make_edge])] = dim_direction
            for j in range(array.ndim):
                spot_yx[spot_yx[:, j] == -1, j] = array.shape[j]-1
        return spot_yx

    def apply_shifted_spots(self, array, spot_yx, thresh):
        """
        updates array with neighbouring pixels to some of spot_yx having value below thresh
        so should be flagged as negative by check_neighbour_intensity function.

        :param array:
        :param spot_yx:
        :param thresh:
        :return:
        """
        # get position of shifted spots to set below thresh
        n_spots = spot_yx.shape[0]
        if array.ndim == 3:
            transforms = self.transforms_3d
        else:
            transforms = self.transforms_2d
        n_fail_spots = np.random.randint(5, ceil(0.75 * n_spots))
        fail_spot_index = np.random.randint(0, n_spots-1, n_fail_spots)
        fail_transform_index = np.random.randint(0, transforms.shape[0], n_fail_spots)
        fail_mod_yx = spot_yx[fail_spot_index, :] + transforms[fail_transform_index, :]

        # only keep shifted spots which don't interfere with other spots.
        keep_isolated = find_isolated_spots(spot_yx, fail_mod_yx)
        # only keep shifts within dimensions of array
        keep_dims_below = np.min(fail_mod_yx >= [0*i for i in range(array.ndim)], axis=1)
        keep_dims_above = np.min(fail_mod_yx < [array.shape[i] for i in range(array.ndim)], axis=1)
        keep_final = np.logical_and.reduce([keep_isolated, keep_dims_below, keep_dims_above])

        # change value of array at those locations where shifted spot is still valid.
        fail_mod_yx = fail_mod_yx[keep_final, :]
        fail_value = np.random.uniform(thresh-100, thresh-1, fail_mod_yx.shape[0])
        array[tuple([fail_mod_yx[:, j] for j in range(array.ndim)])] = fail_value
        expected_answer = np.ones((n_spots,), dtype=bool)
        expected_answer[fail_spot_index[keep_final]] = False
        return array, expected_answer, fail_mod_yx

    def all_test(self, dimensions=2, edges=False):
        """
        bridge function to perform relevant tests

        :param dimensions:
        :param edges:
        :return:
        """
        array, n_spots, thresh = self.get_initial_info(dimensions)
        spot_yx = self.get_spots(array, n_spots, edges)
        array, expected_answer, fail_mod_yx = self.apply_shifted_spots(array, spot_yx, thresh)
        actual_answer = check_neighbour_intensity(array, spot_yx, thresh)
        diff = actual_answer.astype(int) - expected_answer.astype(int)
        self.assertTrue(np.abs(diff).max() <= 0)

    def test_2d(self):
        self.all_test(dimensions=2, edges=False)

    def test_2d_edges(self):
        self.all_test(dimensions=2, edges=True)

    def test_3d(self):
        self.all_test(dimensions=3, edges=False)

    def test_3d_edges(self):
        self.all_test(dimensions=3, edges=True)


if __name__ == '__main__':
    unittest.main()
