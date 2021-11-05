import unittest
from iss.extract.fstack import focus_stack
from iss.utils.matlab import load_cell, load_array
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
        for file_name in [s for s in os.listdir(self.folder) if "teast" in s]:
            test_file = self.folder + file_name
            input_im = np.moveaxis(np.array(load_cell(test_file, 'input')).squeeze(), 0, 2)
            output_matlab = load_array(test_file, 'output')
            output_python = focus_stack(input_im)
            diff = output_python.astype(int) - output_matlab.astype(int)
            self.assertTrue(np.abs(diff).max() <= self.tol)


if __name__ == '__main__':
    unittest.main()
