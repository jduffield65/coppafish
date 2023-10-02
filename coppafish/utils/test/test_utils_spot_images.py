from coppafish.utils.spot_images import get_spot_images, get_average_spot_image

import numpy as np
import math as maths


def test_get_spot_images():
    image_x = 64
    image_y = 128
    image_z = 10
    # Y then X
    shape_2d = [5, 7]
    shape_3d = [9, 3, 3]
    rng = np.random.RandomState(14)
    image_2d = rng.rand(image_y, image_x) * 100 - 40
    image_3d = rng.rand(image_y, image_x, image_z) * 100 - 40
    n_peaks = 10
    spot_yxz_2d = np.zeros((n_peaks, 2), dtype=int)
    spot_yxz_3d = np.zeros((n_peaks, 3), dtype=int)
    # Place spot position information and create expected output
    expected_output_2d = np.empty((n_peaks, shape_2d[0], shape_2d[1]))
    expected_output_3d = np.empty((n_peaks, shape_3d[0], shape_3d[1], shape_3d[2]))
    for i in range(n_peaks):
        y = rng.randint(10, image_y - 10)
        x = rng.randint(10, image_x - 10)
        z = rng.randint(4, image_z - 4)
        spot_yxz_2d[i] = np.array([y, x])
        expected_output_2d[i] = image_2d[
            y - shape_2d[0]//2:y + maths.ceil(shape_2d[0]/2),
            x - shape_2d[1]//2:x + maths.ceil(shape_2d[1]/2),
        ]
        spot_yxz_3d[i] = \
            np.array([y, x, z])
        expected_output_3d[i] = image_3d[
            y - shape_3d[0]//2:y + maths.ceil(shape_3d[0]/2),
            x - shape_3d[1]//2:x + maths.ceil(shape_3d[1]/2),
            z - shape_3d[2]//2:z + maths.ceil(shape_3d[2]/2),
        ]
    output_2d = get_spot_images(image_2d, spot_yxz_2d, shape_2d)
    output_3d = get_spot_images(image_3d, spot_yxz_3d, shape_3d)
    assert np.allclose(output_2d, expected_output_2d), '2d output was not expected'
    assert np.allclose(output_3d, expected_output_3d), '3d output was not expected'
    output_2d = get_spot_images(image_2d, spot_yxz_2d, np.array(shape_2d))
    output_3d = get_spot_images(image_3d, spot_yxz_3d, np.array(shape_3d))
    assert np.allclose(output_2d, expected_output_2d), '2d output was not expected'
    assert np.allclose(output_3d, expected_output_3d), '3d output was not expected'
    # Test the out of bounds mechanic
    assert np.all(np.isnan(get_spot_images(
        np.zeros((5, 3)), np.array([[20, 10], [-10, -100]]), shape=[3, 5]))), \
        'Expected only nan values when out of bounds 2d spot given'
    assert np.all(np.isnan(get_spot_images(
        np.zeros((5, 3, 2)), np.array([[20, 10, 1000], [-10, -100, 1000]]), shape=[1, 5, 3]))), \
        'Expected only nan values when out of bounds 3d spot given'
    assert np.all(np.isnan(get_spot_images(
        np.zeros((5, 3)), np.array([[20, 10], [-10, -100]]), shape=np.array([3, 5])))), \
        'Expected only nan values when out of bounds 2d spot given'
    assert np.all(np.isnan(get_spot_images(
        np.zeros((5, 3, 2)), np.array([[20, 10, 1000], [-10, -100, 1000]]), shape=np.array([1, 5, 3])))), \
        'Expected only nan values when out of bounds 3d spot given'


def test_get_average_spot_image():
    n_peaks = 4
    x_shape = 5
    y_shape = 7
    z_shape = 9
    # Make each spot image each have a constant value
    spot_images_2d = np.zeros((n_peaks, y_shape, x_shape),          dtype=float)
    spot_images_3d = np.zeros((n_peaks, y_shape, x_shape, z_shape), dtype=float)
    fill_values = []
    for n in range(n_peaks):
        fill_values.append(2*(n if n != 0 else n+1))
        spot_images_2d[n] = np.full((y_shape, x_shape), fill_value=fill_values[n])
        spot_images_3d[n] = np.full((y_shape, x_shape, z_shape), fill_value=fill_values[n])
    output_2d_mean   = get_average_spot_image(spot_images_2d)
    output_3d_mean   = get_average_spot_image(spot_images_3d)
    output_2d_median = get_average_spot_image(spot_images_2d, av_type='median')
    output_3d_median = get_average_spot_image(spot_images_3d, av_type='median')
    assert output_2d_mean.shape   == (y_shape, x_shape), 'Unexpected 2d output shape'
    assert output_3d_mean.shape   == (y_shape, x_shape, z_shape), 'Unexpected 3d output shape'
    assert output_2d_median.shape == (y_shape, x_shape), 'Unexpected 2d output shape'
    assert output_3d_median.shape == (y_shape, x_shape, z_shape), 'Unexpected 3d output shape'
    assert np.allclose(output_2d_mean,   np.mean(fill_values)), 'Unexpected 2d output value'
    assert np.allclose(output_3d_mean,   np.mean(fill_values)), 'Unexpected 3d output value'
    assert np.allclose(output_2d_median, np.median(fill_values)), 'Unexpected 2d output value'
    assert np.allclose(output_3d_median, np.median(fill_values)), 'Unexpected 3d output value'
    # Check nan value logic
    spot_images_2d[0:2] = np.nan
    spot_images_3d[0:2] = np.nan
    assert np.allclose(get_average_spot_image(spot_images_2d,   'mean'), np.mean  (fill_values[2:])), \
        'Unexpected 2d output with nan values'
    assert np.allclose(get_average_spot_image(spot_images_3d,   'mean'), np.mean  (fill_values[2:])), \
        'Unexpected 3d output with nan values'
    assert np.allclose(get_average_spot_image(spot_images_2d, 'median'), np.median(fill_values[2:])), \
        'Unexpected 2d output with nan values'
    assert np.allclose(get_average_spot_image(spot_images_3d, 'median'), np.median(fill_values[2:])), \
        'Unexpected 3d output with nan values'
    #TODO: Add unit tests for symmetric parameters included
