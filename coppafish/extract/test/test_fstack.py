import numpy as np

from coppafish.extract import fstack
from coppafish.register import preprocessing


def test_rgb2gray():
    rng = np.random.RandomState(7)
    images = []
    images.append(rng.rand(11, 14, 3).astype(np.float16))
    images.append(rng.rand(6, 2, 3).astype(np.float32))
    images.append(rng.rand(1, 3, 3).astype(np.float64))
    images.append(rng.randint(2**8-1, size=(3, 5, 3), dtype=np.uint8))
    images.append(rng.randint(2**7-1, size=(3, 5, 3), dtype=np.int8))
    images.append(rng.randint(2**16-1, size=(7, 9, 3), dtype=np.uint16))
    images.append(rng.randint(2**15-1, size=(8, 6, 3), dtype=np.int16))
    images.append(rng.randint(2**32-1, size=(8, 10, 3), dtype=np.uint32))
    images.append(rng.randint(2**31-1, size=(9, 11, 3), dtype=np.int32))
    images.append(rng.randint(2**64-1, size=(8, 18, 3), dtype=np.uint64))
    images.append(rng.randint(2**63-1, size=(8, 18, 3), dtype=np.int64))
    for image in images:
        output = fstack.rgb2gray(image)
        assert output.ndim == 2, 'Expected two dimensional output'
        assert output.shape == image.shape[:2], f'Expected shape {image.shape[:2]}, got {output.shape}'
        assert output.dtype == image.dtype, 'Expected output as the same dtype as input'


def test_im2double():
    rng = np.random.RandomState(28)
    images = []
    images.append(rng.rand(11, 14).astype(np.float16))
    images.append(rng.rand(6, 2).astype(np.float32))
    images.append(rng.rand(1).astype(np.float64))
    images.append(rng.randint(2**8-1, size=(3, 2, 4, 5, 6, 2, 1), dtype=np.uint8))
    images.append(rng.randint(2**7-1, size=(3, 5), dtype=np.int8))
    images.append(rng.randint(2**16-1, size=(7, 9), dtype=np.uint16))
    images.append(rng.randint(2**15-1, size=(8, 6), dtype=np.int16))
    images.append(rng.randint(2**32-1, size=(8, 10), dtype=np.uint32))
    images.append(rng.randint(2**31-1, size=(9, 11), dtype=np.int32))
    images.append(rng.randint(2**64-1, size=(8, 18), dtype=np.uint64))
    images.append(rng.randint(2**63-1, size=(8, 18, 7), dtype=np.int64))
    for image in images:
        output = fstack.im2double(image)
        assert output.ndim == image.ndim, f'Expected {image.ndim} dimensions, got {output.ndim}'
        assert output.shape == image.shape, f'Expected shape {image.shape}, got {output.shape}'
        assert output.dtype == np.float64, f'Expected float64 dtype output, got {output.dtype}'
        assert output.max() <= 1, 'Output maximum must be <= 1'
        assert output.min() >= -1, 'Output minimum must be >= -1'


def test_focus_stack():
    rng = np.random.RandomState(51)
    images = []
    images.append(rng.randint(1, 2**8-1, size=(5, 6, 7, 3), dtype=np.uint8))
    images.append(rng.randint(1, 2**16-1, size=(8, 9, 10), dtype=np.uint16))
    for image in images:
        output_0 = fstack.focus_stack(image, nhsize=rng.randint(np.min(list(image.shape)), dtype=np.int64), 
                                      alpha=rng.rand(), sth=rng.rand()*(2**63-1))
        if image.ndim == 4:
            assert output_0.shape == (image.shape[0], image.shape[1], 3), f'Unexpected output shape'
        else:
            assert output_0.shape == image.shape[:2], f'Unexpected output shape'
        assert output_0.dtype in [np.int_, np.uint8], f'Unexpected output dtype: {output_0.dtype}'
    # Check that the output looks the same even with an x and y offset
    image = rng.randint(1, 2**8-1, size=(25, 26, 27, 3), dtype=np.uint8)
    offset = rng.randint(5, size=(4))
    # No z shift because it is not clear how it will affect the output
    offset[2], offset[3] = 0, 0
    image_shifted = preprocessing.custom_shift(image, offset)
    output = fstack.focus_stack(image)
    output_shifted = fstack.focus_stack(image_shifted)
    assert np.allclose(
        output[7:15,7:15,:], 
        output_shifted[7+offset[0]:15+offset[0],7+offset[1]:15+offset[1],:]
    ), f'Output from focus stack did not shift as expected'
