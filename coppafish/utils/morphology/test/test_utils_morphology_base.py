import numpy as np

from coppafish.utils import morphology


def test_hanning_diff():
    output_0 = morphology.base.hanning_diff(1,2)
    assert output_0.shape == (2*2+1, 2*2+1), 'Unexpected output shape'
    assert np.unravel_index(np.argmax(output_0, axis=None), output_0.shape) == (2,2), \
        'Expected maximum of kernel at centre'
    assert np.allclose(output_0.sum(), 0), 'Expected output to sum to zero'
    output_1 = morphology.base.hanning_diff(10,11)
    for i in range(11):
        if i == 0:
            continue
        assert output_1[i+1,11] > output_1[i,11], \
            'Expected hanning difference to monotonically increase along one axis'


def test_convolve_2d():
    size_0 = 11
    size_1 = 6
    image_shape = (size_0, size_1)
    rng = np.random.RandomState(7)
    image = rng.rand(size_0, size_1)
    # Simple summing kernel test
    kernel_0 = np.ones(image_shape)
    output_0 = morphology.base.convolve_2d(image, kernel_0)
    assert output_0.shape == image_shape, 'Unexpected output shape'
    assert output_0[size_0//2, size_1//2] == image.sum(), 'Expected kernel to sum all values together'
    assert np.allclose(morphology.base.convolve_2d(image, np.array([0])), 0), 'Expected zeros output'


def test_ensure_odd_kernel():
    rng = np.random.RandomState(13)
    array_odd  = rng.rand(5, 7, 11, 9, 1, 1, 3)
    array_even = rng.rand(2, 7, 11, 10, 1, 1, 3)
    new_shape = (3, 7, 11, 11, 1, 1, 3)
    assert morphology.base.ensure_odd_kernel(array_odd).shape == array_odd.shape
    assert morphology.base.ensure_odd_kernel(array_odd, 'end').shape == array_odd.shape
    output_start = morphology.base.ensure_odd_kernel(array_even)
    output_end   = morphology.base.ensure_odd_kernel(array_even, 'end')
    assert output_start.shape == new_shape, 'Unexpected output shape'
    assert output_end.shape == new_shape, 'Unexpected output shape'
    assert np.allclose(output_start[ 0,:,:, 0], 0), 'Expected zeros as padding'
    assert np.allclose(output_end  [-1,:,:,-1], 0), 'Expected zeros as padding'
    assert np.allclose(output_start[1:,:,:, 1:], array_even), 'Unexpected output'
    assert np.allclose(output_end  [:2,:,:,:10], array_even), 'Unexpected output'


# def test_top_hat():
#     rng = np.random.RandomState(52)
#     image_x = 23
#     image_y = 11
#     image_shape  = (image_y, image_x)
#     image_centre = (image_y//2, image_x//2)
#     image_0 = rng.rand(image_y, image_x).astype(np.float64)
#     image_1 = rng.randint(2**16-1, size=image_shape, dtype=np.uint16)
#     kernel_0 = np.ones(image_shape, dtype=np.uint8)
#     kernel_1 = rng.randint(2, size=image_shape, dtype=np.uint8)
#     # Check every combination of kernel and image together
#     images = [image_0, image_1]
#     kernels = [kernel_0, kernel_1]
#     for image in images:
#         for kernel in kernels:
#             kernel_centre = (kernel.shape[0]//2, kernel.shape[1]//2)
#             output = top_hat(image, kernel)
#             expected_centre_pixel = np.multiply(image[
#                 image_centre[0] - kernel_centre[0]:image_centre[0] + kernel_centre[0] + 1, 
#                 image_centre[1] - kernel_centre[1]:image_centre[1] + kernel_centre[1] + 1], kernel).sum()
#             assert np.allclose(output[image_centre], expected_centre_pixel), 'Unexpected central pixel result'


def test_dilate():
    rng = np.random.RandomState(75)
    images = [rng.rand(1, 2, 3, 6, 10) * 2**15, rng.rand(2, 2).astype(np.float32), rng.rand(11).astype(np.float64)]
    kernels = [rng.randint(2, size=(1,2,3,6,5)), rng.randint(2, size=(2, 2)), rng.randint(2, size=(11))]
    for image, kernel in zip(images, kernels):
        output = morphology.base.dilate(image, kernel)
        assert output.shape == image.shape, 'Expected the dilated `image` shape to equal input `image` shape'
        assert not np.allclose(output, image), 'Expected dilated `image` to be different to the input `image`'
