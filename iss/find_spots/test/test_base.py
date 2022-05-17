import unittest
import os
import numpy as np
from ..base import detect_spots, get_isolated
from ...utils import matlab, errors, strel


class TestBase(unittest.TestCase):
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_detect_spots(self):
        """
        Check whether spots found using detect_spots have same location and intensity as those found
        in MATLAB in python_testing/find_spots/detect_spots script.

        test files contain:
        image: image that spots are to be found on.
        intensity_thresh:  spots are local maxima in image with pixel value > intensity_thresh
        r: xy radius of structure element
        r_z: z radius of structure element (0 if 2d filter used)
        remove_duplicates: Whether to only keep one pixel if two or more pixels are
                           local maxima and have same intensity.
        PeakYX: yx or yxz location of spots found.
        PeakIntensity: pixel value of spots found.
        """
        folder = os.path.join(self.folder, 'detect_spots')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, thresh, r_xy, r_z, remove_duplicates, peak_yx_matlab, peak_intensity_matlab = \
                matlab.load_array(test_file, ['image', 'intensity_thresh', 'r', 'r_z',
                                              'remove_duplicates', 'PeakYX','PeakIntensity'])
            peak_intensity_matlab = peak_intensity_matlab.flatten()
            # In MATLAB used disk se not cuboid se so provide se not radii to detect_spots function.
            if int(r_z) == 0:
                se = strel.disk(int(r_xy))
            else:
                se = strel.disk_3d(int(r_xy), int(r_z))
            remove_duplicates = int(remove_duplicates) == 1
            peak_yx_python, peak_intensity_python = detect_spots(image, float(thresh), None,
                                                                 None, remove_duplicates, se)
            # Sort both data sets same way to compare (by intensity and then by y).
            # need to make intensity integer for sort to deal with random shift.
            sorted_arg_python = np.lexsort((peak_yx_python[:, 0], peak_intensity_python.astype(int)))
            sorted_arg_matlab = np.lexsort((peak_yx_matlab[:, 0], peak_intensity_matlab.astype(int)))
            peak_yx_matlab = peak_yx_matlab[sorted_arg_matlab, :] - 1  # to match python indexing
            peak_yx_python = peak_yx_python[sorted_arg_python, :]
            peak_intensity_matlab = peak_intensity_matlab[sorted_arg_matlab]
            peak_intensity_python = peak_intensity_python[sorted_arg_python]

            diff_yx = peak_yx_python - peak_yx_matlab.astype(int)
            diff_intensity = peak_intensity_python - peak_intensity_matlab
            if "avoid_repeats" in file_name:
                # these tests have neighbouring pixels with same intensity - can accept either pixel
                # hence expect a different position but same intensity.
                tol_yx = 2
            else:
                tol_yx = 1
            self.assertTrue(np.abs(diff_yx).max() <= tol_yx)  # check match MATLAB
            self.assertTrue(np.abs(diff_intensity).max() <= 0.5)  # check match MATLAB
            self.assertTrue(peak_intensity_python.min() > float(thresh))

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
            output_python = get_isolated(image, peak_yx.astype(int)-1, float(thresh),
                                         float(r0), float(r_xy), r_z).astype(int)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= 0)  # check match MATLAB


if __name__ == '__main__':
    unittest.main()
