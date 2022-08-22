import unittest
import numpy as np
from ..base import strip_hack


class TestStripHack(unittest.TestCase):
    tol = 0

    @staticmethod
    def get_answer(array):
        """
        gets initial array to give to strip_hack as well as expected answer.

        :param array: 2d or 3d numpy integer array
        :return:
            array: input array for strip_hack function
            array_answer: expected answer from strip_hack
            columns_to_change: columns set to a constant in array
        """
        # get some columns to set to a single value across all rows and z-planes
        rng = np.random.default_rng()
        n_col_change = np.random.randint(array.shape[1]-1)  # always have at least one column not changed
        columns_to_change = rng.choice(array.shape[1], size=n_col_change, replace=False)
        columns_to_change.sort()
        if array.ndim == 3:
            array[:, columns_to_change] = np.random.randint(0, 10000, (1, n_col_change, 1))
        else:
            array[:, columns_to_change] = np.random.randint(0, 10000, (1, n_col_change))
        const_columns = np.setdiff1d(np.arange(array.shape[1]), columns_to_change)
        # get expected answer by changing these to the nearest column not changed
        array_answer = array.copy()
        for col in columns_to_change:
            nearest_const_col = const_columns[np.argmin(np.abs(const_columns-col))]
            array_answer[:, col] = array_answer[:, nearest_const_col]
        return array, array_answer, columns_to_change

    def all_test(self, array):
        """
        comparison to do for both tests

        :param array: 2d or 3d random numpy integer array
        """
        # get some columns to set to a single value across all rows and z-planes
        func_input, answer, columns_to_change = self.get_answer(array)
        func_result, func_cols = strip_hack(func_input.copy())
        diff = func_result - answer
        self.assertTrue(np.abs(diff).max() <= self.tol)  # check assigned columns in correct way
        # check all columns found
        self.assertTrue(len([i for i in func_cols if i in columns_to_change]) == len(columns_to_change))

    def test_3d(self):
        # get random 3d integer array (nd2 images are integer)
        array = np.random.randint(0, 10000, (np.random.randint(2, 1000), np.random.randint(2, 1000),
                                  np.random.randint(2, 50)))
        self.all_test(array)

    def test_2d(self):
        # get random 2d integer array (nd2 images are integer)
        array = np.random.randint(0, 10000, (np.random.randint(2, 1000), np.random.randint(2, 1000)))
        self.all_test(array)


if __name__ == '__main__':
    unittest.main()
