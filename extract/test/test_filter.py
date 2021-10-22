import unittest
import os
import numpy as np
from extract.filter import hanning_diff, disk_strel
from utils.matlab import load_array
import utils.errors


class TestFilter(unittest.TestCase):
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_hanning_diff(self):
        """
        Check whether hanning filters are same as with MATLAB
        and that sum of filter is 0.

        test files contain:
        r1: inner radius of hanning filter
        r2: outer radius of hanning filter
        h: hanning filter produced by MATLAB
        """
        folder = os.path.join(self.folder, 'hanning')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r1, r2, output_matlab = load_array(test_file, ['r1', 'r2', 'h'])
            output_python = hanning_diff(int(r1), int(r2))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB
            self.assertTrue(np.abs(output_python.sum()) <= self.tol)  # check sum is zero

    def test_disk(self):
        """
        Check whether disk_strel gives the same results as MATLAB strel('disk)

        test_files contain:
        r: radius of filter kernel
        n: 0, 4, 6 or 8
        nhood: filter kernel found by MATLAB
        """
        folder = os.path.join(self.folder, 'disk')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r, n, output_matlab = load_array(test_file, ['r', 'n', 'nhood'])
            output_python = disk_strel(int(r), int(n))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB


if __name__ == '__main__':
    unittest.main()
