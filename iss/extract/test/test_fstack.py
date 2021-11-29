import unittest
from ..fstack import focus_stack
from ...utils import matlab, errors
import os
import numpy as np


class TestFstack(unittest.TestCase):
    """
    Load in matlab data and test to see if get same output as Matlab did
    folder contains matlab examples in files with "test" prefix.
    Each file contains input image in form of MATLAB cell
    and output as array.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/focus_stack/')
    tol = 1

    def test_focus_stack(self):
        test_files = [s for s in os.listdir(self.folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(self.folder, file_name)
            input_im = np.moveaxis(np.array(matlab.load_cell(test_file, 'input')).squeeze(), 0, 2)
            output_matlab = matlab.load_array(test_file, 'output')
            output_python = focus_stack(input_im)
            diff = output_python.astype(int) - output_matlab.astype(int)
            self.assertTrue(np.abs(diff).max() <= self.tol)


if __name__ == '__main__':
    unittest.main()
