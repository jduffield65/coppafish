import unittest
import os
import numpy as np
from ..base import get_isolated
from ...utils import matlab, errors, strel


class TestIsolated(unittest.TestCase):
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_get_isolated(self):
        """
        Check whether spots identified as isolated using get_isolated match those identified
        in MATLAB in python_testing/find_spots/isolated_spots script.

        test files contain:
        image: image that spots were found on.
        thresh: spots are isolated if annulus filtered image at spot location less than this.
        r0: inner radius of annulus filtering kernel within which values are all zero.
        rXY: outer radius of annulus filtering kernel in xy direction.
        rZ: outer radius of annulus filtering kernel in z direction (0 if 2d filter used).
        Isolated: boolean indicated whether each spot is isolated.
        """
        folder = os.path.join(self.folder, 'isolated')
        # files have matching names to those in detect_spots examples as same images and spots where used
        # except for files with 'random' in where different PeakYX were used.
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, thresh, r0, r_xy, r_z, peak_yx, output_matlab = \
                matlab.load_array(test_file, ['image', 'thresh', 'r0', 'rXY', 'rZ', 'PeakYX', 'Isolated'])
            if int(r_z) == 0:
                r_z = None
            else:
                r_z = float(r_z)
            output_matlab = output_matlab.flatten().astype(int)
            output_python = get_isolated(image, np.ascontiguousarray(peak_yx.astype(int)-1), float(thresh),
                                         float(r0), float(r_xy), r_z).astype(int)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= 0)  # check match MATLAB


if __name__ == '__main__':
    unittest.main()
