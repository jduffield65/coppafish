import unittest
import os
import numpy as np
from ..morphology import hanning_diff, convolve_2d, top_hat, dilate, imfilter, imfilter_coords, ensure_odd_kernel
from ..strel import disk, disk_3d, annulus, fspecial
from ...utils import matlab, errors


class TestMorphology(unittest.TestCase):
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-9

    def test_hanning_diff(self):
        """
        Check whether hanning filters are same as with MATLAB
        and that sum of convolve_2d is 0.

        test files contain:
        r1: inner radius of hanning convolve_2d
        r2: outer radius of hanning convolve_2d
        h: hanning convolve_2d produced by MATLAB
        """
        folder = os.path.join(self.folder, 'hanning')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r1, r2, output_matlab = matlab.load_array(test_file, ['r1', 'r2', 'h'])
            output_python = hanning_diff(int(r1), int(r2))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB
            self.assertTrue(np.abs(output_python.sum()) <= self.tol)  # check sum is zero

    def test_disk(self):
        """
        Check whether disk_strel gives the same results as MATLAB strel('disk)

        test_files contain:
        r: radius of convolve_2d kernel
        n: 0, 4, 6 or 8
        nhood: convolve_2d kernel found by MATLAB
        """
        folder = os.path.join(self.folder, 'disk')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r, n, output_matlab = matlab.load_array(test_file, ['r', 'n', 'nhood'])
            output_python = disk(int(r), int(n))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    def test_disk_3d(self):
        """
        Check whether 3d structuring element are same as with MATLAB
        function strel3D_2

        test files contain:
        rXY: xy radius of structure element
        rZ: z radius of structure element
        kernel: structuring element produced by MATLAB strel3D_2 function (In iss 3d branch)
        """
        folder = os.path.join(self.folder, 'disk_3d')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r_xy, r_z, output_matlab = matlab.load_array(test_file, ['rXY', 'rZ', 'kernel'])
            # remove zeros at edges for MATLAB version
            output_matlab = output_matlab[:, :, ~np.all(output_matlab == 0, axis=(0, 1))]
            output_matlab = output_matlab[:, ~np.all(output_matlab == 0, axis=(0, 2)), :]
            output_matlab = output_matlab[~np.all(output_matlab == 0, axis=(1, 2)), :, :]
            output_python = disk_3d(int(r_xy), int(r_z))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= 0)  # check match MATLAB

    def test_annulus(self):
        """
        Check whether annulus structuring element matches MATLAB code:
        # 2D
        [xr, yr] = meshgrid(-floor(rXY):floor(rXY),-floor(rXY):floor(rXY));
        Annulus = (xr.^2 + yr.^2)<=rXY.^2 & (xr.^2 + yr.^2) > r0.^2;
        # 3D
        [xr, yr, zr] = meshgrid(-floor(rXY):floor(rXY),-floor(rXY):floor(rXY),-floor(rZ):floor(rZ));
        Annulus = (xr.^2 + yr.^2+zr.^2)<=rXY.^2 & (xr.^2 + yr.^2+zr.^2) > r0.^2;

        test files contain:
        rXY: xy outer radius of structure element
        rZ: z outer radius of structure element (0 if 2D)
        r0: inner radius of structure element, within which it is 0.
        Annulus: structuring element produced by MATLAB
        """
        folder = os.path.join(self.folder, 'annulus')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r_xy, r_z, r0, output_matlab = matlab.load_array(test_file, ['rXY', 'rZ', 'r0', 'Annulus'])
            if r_z == 0:
                output_python = annulus(float(r0), float(r_xy))
            else:
                output_python = annulus(float(r0), float(r_xy), float(r_z))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= 0)  # check match MATLAB

    def test_fspecial_ellipsoid(self):
        folder = os.path.join(self.folder, 'fspecial_ellipsoid')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            se_sz, output_matlab = matlab.load_array(test_file, ['SE_sz', 'SE'])
            se_sz = se_sz[0]
            output_python = fspecial(*tuple(se_sz))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    def test_fspecial_disk(self):
        folder = os.path.join(self.folder, 'fspecial_disk')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            se_sz, output_matlab = matlab.load_array(test_file, ['SE_sz', 'SE'])
            se_sz = se_sz[0]
            output_python = fspecial(*tuple(se_sz))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    def test_convolve_2d(self):
        """
        Check whether filter_imaging gives same results as MATLAB:
        I_mod = padarray(image,(size(kernel)-1)/2,'replicate','both');
        image_filtered = convn(I_mod, kernel,'valid');

        test_file contains:
        image: image to convolve_2d (no padding)
        kernel: array to convolve image with
        image_filtered: result of MATLAB filtering
        """
        folder = os.path.join(self.folder, 'convolve_2d')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = matlab.load_array(test_file, ['image', 'kernel', 'image_filtered'])
            output_python = convolve_2d(image, kernel)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    def test_filter_dapi(self):
        """
        Check whether filter_dapi gives same results as MATLAB:
        image_filtered = imtophat(image, kernel);

        test_file contains:
        image: image to convolve_2d (no padding)
        kernel: array to apply tophat convolve_2d to image with
        image_filtered: result of MATLAB filtering
        """
        folder = os.path.join(self.folder, 'dapi')
        test_files = [s for s in os.listdir(folder) if "test" in s and "even" not in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            # MATLAB and python differ if kernel has any even dimensions and is not symmetric
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = matlab.load_array(test_file, ['image', 'kernel', 'image_filtered'])
            output_python = top_hat(image, kernel)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    @unittest.expectedFailure
    def test_filter_dapi_even(self):
        """
        as above but with even kernels, should fail.
        test_file contains:
        image: image to convolve_2d (no padding)
        kernel: array to apply tophat convolve_2d to image with
        image_filtered: result of MATLAB filtering
        """
        folder = os.path.join(self.folder, 'dapi')
        test_files = [s for s in os.listdir(folder) if "test" in s and "even" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = matlab.load_array(test_file, ['image', 'kernel', 'image_filtered'])
            output_python = top_hat(image, kernel)

    def test_dilate(self):
        """
        Check whether dilate gives same results as MATLAB imdilate function.

        test_file contains:
        image: image to dilate (no padding)
        kernel: structuring element to dilate with
        image_dilated: result of dilation.
        """
        folder = os.path.join(self.folder, 'dilate')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = matlab.load_array(test_file, ['image', 'kernel', 'image_dilated'])
            output_python = dilate(image, kernel.astype(int))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    def test_imfilter(self):
        """
        Check whether imfilter gives same results as MATLAB imfilter function.

        test_file contains:
        image: image to filter (no padding)
        kernel: structuring element to filter with.
        pad: method of padding, if number means constant at each edge.
        conv_or_corr: whether convolution or correlation was used.
        image_filtered: result of filtering.
        """
        tol = 1e-5
        folder = os.path.join(self.folder, 'imfilter')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        # matlab_to_python_pad = {
        # 'symmetric': 'reflect', 'replicate': 'nearest', 'circular': 'wrap'} # for old conv method
        matlab_to_python_pad = {'symmetric': 'symmetric', 'replicate': 'edge', 'circular': 'wrap'}
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, pad, corr_or_conv, kernel, output_matlab = \
                matlab.load_array(test_file, ['image', 'pad', 'conv_or_corr', 'kernel', 'image_filtered'])
            if len(pad.flatten()) > 1:
                pad = ''.join([chr(val) for val in pad.flatten()])
                pad = matlab_to_python_pad[pad]
            else:
                pad = pad.flatten()[0]
            corr_or_conv = ''.join([chr(val) for val in corr_or_conv.flatten()])
            output_python = imfilter(image, kernel, pad, corr_or_conv)
            output_python_no_oa = imfilter(image, kernel, pad, corr_or_conv, oa=False)
            diff = output_python - output_matlab
            diff2 = output_python - output_python_no_oa
            self.assertTrue(np.abs(diff).max() <= tol)  # check match MATLAB
            self.assertTrue(np.abs(diff2).max() <= tol)

    def test_imfilter_coords(self):
        """

        """
        tol = 1e-2  # for computing local maxima: shouldn't matter what it is (keep below 0.01 for int image).

        im_sz = np.random.randint(30, 300, 3)
        kernel_sz = np.random.randint(3, 29, 3)
        for ndims in [2, 3]:
            image = np.random.choice([-1, 0, 1], im_sz[:ndims]).astype(np.int8)
            n_spots = np.random.randint(3, np.clip(np.prod(image.shape)/10, 4, 300).astype(int))
            coords = np.round(np.random.rand(n_spots, ndims) * (im_sz[:ndims]-1)).astype(int)

            if ndims == 3 and bool(np.random.randint(2)):
                # sometimes use a 2D kernel for 3D image
                kernel = np.random.randint(0, 100, kernel_sz[:2])
            else:
                kernel = np.random.randint(0, 100, kernel_sz[:ndims])

            corr_or_conv = np.random.choice(['conv', 'corr'])
            padding = np.random.choice(['symmetric', 'edge', 'wrap', 'constant'])
            if padding == 'constant':
                padding = np.random.randint(-20, 20)

            if ndims == 3 and kernel.ndim==2:
                kernel_filt = np.expand_dims(kernel, 2)
            else:
                kernel_filt = kernel.copy()

            # cython code designed to work with binary kernel.
            kernel_filt = kernel_filt / np.max(kernel)
            kernel_filt = np.round(kernel_filt).astype(int)

            im_filt = imfilter(image, kernel_filt, padding, corr_or_conv)
            im_filt_result = im_filt[tuple([coords[:, j] for j in range(ndims)])]

            jax_result = imfilter_coords(image, kernel_filt, coords, padding, corr_or_conv)
            diff = jax_result - np.round(im_filt_result).astype(int)
            self.assertTrue(np.abs(diff).max() <= tol)  # check match full image filtering


if __name__ == '__main__':
    unittest.main()
