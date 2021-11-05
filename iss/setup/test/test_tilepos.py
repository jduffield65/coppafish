import unittest
from utils.matlab import load_array
from setup.tile_details import get_tilepos
import os
import numpy as np


class TestTilePos(unittest.TestCase):
    """
    Check whether nd2 tile indices are the same as got with MATLAB
    Also check whether tiff first and last indices make sense.

    test files contain:
    InputDirectory: where nd2 files are.
    imfile: path to actual nd2 file used.
    xypos: input to get_tilepos function.
    TilePosYX: output of get_tilepos function.
    tile_sz: xy-dimension of tile in pixels.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/tilepos/')
    tol = 0

    def test_get_tilepos(self):
        for file_name in [s for s in os.listdir(self.folder) if "test" in s]:
            test_file = self.folder + file_name
            xy_pos = load_array(test_file, 'xypos')
            output_matlab = load_array(test_file, 'TilePosYX') - 1
            output_python = get_tilepos(xy_pos, int(load_array(test_file, 'tile_sz')))
            diff = output_python['nd2'].astype(int) - output_matlab.astype(int)
            self.assertTrue(np.abs(diff).max() <= self.tol)
            # check first position of tiff_yx is [0,0]
            self.assertTrue(sum(output_python['tiff'][0]) <= self.tol)
            # check last position of tiff_yx is [max_y, max_x]
            self.assertTrue(sum(output_python['tiff'][-1] - np.max(output_python['tiff'], 0)) <= self.tol)


if __name__ == '__main__':
    unittest.main()
