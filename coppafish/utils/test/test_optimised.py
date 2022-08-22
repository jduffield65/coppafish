import unittest
import os
import numpy as np
from ..morphology.filter import imfilter_coords
from ..morphology.filter_optimised import imfilter_coords as imfilter_coords_optimised


class TestOptimisedImfilter(unittest.TestCase):
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-9

    def test_imfilter_coords(self):
        """
        See whether manual convolution at just a few locations produces same result as filtering entire image.
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

            im_filt_result = imfilter_coords(image, kernel_filt, coords, padding, corr_or_conv)
            jax_result = imfilter_coords_optimised(image, kernel_filt, coords, padding, corr_or_conv)
            diff = jax_result - np.round(im_filt_result).astype(int)
            self.assertTrue(np.abs(diff).max() <= tol)  # check match full image filtering


if __name__ == '__main__':
    unittest.main()
