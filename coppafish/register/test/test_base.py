import unittest
import os
import numpy as np
from ..base import find_shift_array


# Define a class in which the tests will run
class TestFindShiftArray(unittest.TestCase):
    """
    Check whether find_shift_array function works the same as MATLAB function.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_find_shift_array(self):
        folder = os.path.join(self.folder, 'find_shift_array')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            shift_matlab, shift_python = matlab.load_array(test_file, ['shift_matlab', 'shift_python'])
            shift_python = find_shift_array(shift_python)
            diff = shift_python - shift_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)