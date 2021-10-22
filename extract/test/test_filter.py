import unittest
import os
import numpy as np
from extract.filter import get_filter
from utils.matlab import load_array


class TestFilter(unittest.TestCase):
    """
    Check whether hanning filters are same as with MATLAB
    and that sum of filter is 0.

    test files contain:
    r1: inner radius of hanning filter
    r2: outer radius of hanning filter
    h: hanning filter produced by MATLAB
    """

    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/filter/')
    tol = 1e-10

    def test_get_filter(self):
        for file_name in [s for s in os.listdir(self.folder) if "test" in s]:
            test_file = self.folder + file_name
            r1, r2, output_matlab = load_array(test_file, ['r1', 'r2', 'h'])
            output_python = get_filter(int(r1), int(r2))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB
            self.assertTrue(np.abs(output_python.sum()) <= self.tol)  # check sum is zero


if __name__ == '__main__':
    unittest.main()
