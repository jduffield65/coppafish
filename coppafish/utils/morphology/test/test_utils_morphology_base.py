from coppafish.utils.morphology import ftrans2, hanning_diff, convolve_2d, ensure_odd_kernel, top_hat, dilate

import numpy as np


def test_convolve_2d():
    size_0 = 11
    size_1 = 6
    image_shape = (size_0, size_1)
    rng = np.random.RandomState(7)
    image = rng.rand(size_0, size_1)
    # Simple summing kernel test
    kernel_0 = np.ones(image_shape)
    output_0 = convolve_2d(image, kernel_0)
    assert output_0.shape == image_shape, 'Unexpected output shape'
    assert output_0[size_0//2, size_1//2] == image.sum(), 'Expected kernel to sum all values together'
    assert np.allclose(convolve_2d(image, np.array([0])), 0), 'Expected zeros output'


def test_ensure_odd_kernel():
    rng = np.random.RandomState(13)
    array_odd  = rng.rand(5, 7, 11, 9, 1, 1, 3)
    array_even = rng.rand(2, 7, 11, 10, 1, 1, 3)
    new_shape = (3, 7, 11, 11, 1, 1, 3)
    assert ensure_odd_kernel(array_odd).shape == array_odd.shape
    assert ensure_odd_kernel(array_odd, 'end').shape == array_odd.shape
    output_start = ensure_odd_kernel(array_even)
    output_end   = ensure_odd_kernel(array_even, 'end')
    assert output_start.shape == new_shape, 'Unexpected output shape'
    assert output_end.shape == new_shape, 'Unexpected output shape'
    assert np.allclose(output_start[ 0,:,:, 0], 0), 'Expected zeros as padding'
    assert np.allclose(output_end  [-1,:,:,-1], 0), 'Expected zeros as padding'
    assert np.allclose(output_start[1:,:,:, 1:], array_even), 'Unexpected output'
    assert np.allclose(output_end  [:2,:,:,:10], array_even), 'Unexpected output'
