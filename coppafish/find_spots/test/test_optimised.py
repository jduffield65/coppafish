import unittest
import os
import numpy as np
from ..detect import detect_spots
from ..detect_optimised import detect_spots as detect_spots_optimised
from ...utils import matlab, errors, strel


class TestDetectSpots(unittest.TestCase):
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
            if np.sum(se) < 300:
                # Only test detect_spots with small structuring element case as slow with larger one.
                peak_yx_python2, peak_intensity_python2 = detect_spots_optimised(image, float(thresh), None,
                                                                                 None, remove_duplicates, se)
                sorted_arg_python2 = np.lexsort((peak_yx_python2[:, 0], peak_intensity_python2.astype(int)))
                peak_yx_python2 = peak_yx_python2[sorted_arg_python2, :]
                peak_intensity_python2 = peak_intensity_python2[sorted_arg_python2]
                diff_yx2 = peak_yx_python - peak_yx_python2
                diff_intensity2 = peak_intensity_python - peak_intensity_python2
                self.assertTrue(np.abs(diff_yx2).max() <= tol_yx)  # check match other python
                self.assertTrue(np.abs(diff_intensity2).max() <= 0.5)  # check match other python


if __name__ == '__main__':
    unittest.main()
