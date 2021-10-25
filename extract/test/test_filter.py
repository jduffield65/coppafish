import unittest
import os
import numpy as np
from extract.filter import hanning_diff, disk_strel, filter_imaging, filter_dapi
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

    def test_filter_imaging(self):
        """
        Check whether filter_imaging gives same results as MATLAB:
        I_mod = padarray(image,(size(kernel)-1)/2,'replicate','both');
        image_filtered = convn(I_mod, kernel,'valid');

        test_file contains:
        image: image to filter (no padding)
        kernel: array to convolve image with
        image_filtered: result of MATLAB filtering
        """
        folder = os.path.join(self.folder, 'filter')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = load_array(test_file, ['image', 'kernel', 'image_filtered'])
            output_python = filter_imaging(image, kernel)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    def test_filter_dapi(self):
        folder = os.path.join(self.folder, 'dapi')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            # MATLAB and python differ if kernel has any odd dimensions and is not symmetric
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = load_array(test_file, ['image', 'kernel', 'image_filtered'])
            output_python = filter_dapi(image, kernel)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB



if __name__ == '__main__':
    unittest.main()
