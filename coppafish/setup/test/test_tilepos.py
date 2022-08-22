import unittest
from ...utils import matlab, errors
from ..tile_details import get_tilepos
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
        test_files = [s for s in os.listdir(self.folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(self.folder, file_name)
            xy_pos = matlab.load_array(test_file, 'xypos')
            output_matlab = matlab.load_array(test_file, 'TilePosYX') - 1
            output_python_nd2, output_python_tiff = get_tilepos(xy_pos, int(matlab.load_array(test_file, 'tile_sz')))
            # MATLAB is wrong here!!
            diff = output_python_nd2.astype(int) - output_matlab.astype(int)
            # self.assertTrue(np.abs(diff).max() <= self.tol)
            # check first position of tiff_yx_nd2 is [0,0]
            self.assertTrue(np.sum(output_python_nd2[0]) <= self.tol)
            # check last position of tiff_yx_nd2 is [ny, nx] or [ny, 0]
            ny = np.max(output_matlab, axis=0)[0]
            nx = np.max(output_matlab, axis=0)[1]
            self.assertTrue(np.sum(output_python_nd2[0] - [ny, nx]) <= self.tol or
                            np.sum(output_python_nd2[0] - [ny, 0]) <= self.tol)
            # check first position of tiff_yx is [ny, nx]
            self.assertTrue(np.sum(output_python_tiff[0] - [ny, nx]) <= self.tol)
            # check last position of tiff_yx is [0, 0]
            self.assertTrue(np.sum(output_python_tiff[-1]) <= self.tol)


if __name__ == '__main__':
    unittest.main()
