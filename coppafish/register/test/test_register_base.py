import os
import itertools
import numpy as np

from coppafish.utils import tiles_io
from coppafish.setup import NotebookPage
from coppafish.register import base as reg_base
from coppafish.register import preprocessing as reg_pre


def test_find_shift_array():
    # TODO: Make sure we have access to skimage data
    # Set up random data (10, 256, 256)
    rng = np.random.RandomState(12)
    brain = rng.rand(10, 256, 256)
    brain_shifted = reg_pre.custom_shift(brain, np.array([0, 10, 10]))
    # Test the function
    z_box, y_box, x_box = 10, 128, 128
    brain_split, pos = reg_pre.split_3d_image(brain, 1, 2, 2, z_box, y_box, x_box)
    brain_shifted_split, _ = reg_pre.split_3d_image(brain_shifted, 1, 2, 2, z_box, y_box, x_box)
    shift_array, shift_corr = reg_base.find_shift_array(brain_split, brain_shifted_split, pos, r_threshold=0.5)
    # Test that the shape is correct
    assert shift_array.shape == (2 * 2, 3)
    # Test that the values are correct
    assert np.allclose(shift_array, np.array([[0, 10, 10],
                                              [0, 10, 10],
                                              [0, 10, 10],
                                              [0, 10, 10]]))
    assert np.allclose(shift_corr, np.array([1, 1, 1, 1]))


def test_find_z_tower_shifts():
    # set up data (60, 128, 128)
    rng = np.random.RandomState(31)
    cell = rng.rand(60, 128, 128)
    cell_shifted = reg_pre.custom_shift(cell, np.array([3, 0, 0]))
    # Test the function
    z_box, y_box, x_box = 10, 64, 64
    cell_split, pos = reg_pre.split_3d_image(cell, 6, 4, 4, z_box, y_box, x_box)
    cell_shift_split, _ = reg_pre.split_3d_image(cell_shifted, 6, 4, 4, z_box, y_box, x_box)
    pos = pos.reshape(6, 4, 4, 3)
    pos = pos[:, 0, 0]
    cell_tower = cell_split[:, 0, 0]
    cell_shift_tower = cell_shift_split[:, 0, 0]
    shift_array, shift_corr = reg_base.find_z_tower_shifts(cell_tower, cell_shift_tower, pos, pearson_r_threshold=0.5)
    # Test that the shape is correct
    assert shift_array.shape == (6, 3)
    # Test that the values are correct. They should be 6 copies of [3, 0, 0]
    for i in range(6):
        assert np.allclose(shift_array[i], np.array([3, 0, 0]))
    assert np.allclose(shift_corr, np.array([1, 1, 1, 1, 1, 1]))


def test_find_zyx_shift():
    rng = np.random.RandomState(52)
    kidney = np.sum(rng.rand(16, 512, 512, 3), axis=-1)[:, :128, :128]
    kidney_shifted = reg_pre.custom_shift(kidney, np.array([3, 15, 20]))
    # Test the function
    shift, shift_corr = reg_base.find_zyx_shift(kidney, kidney_shifted, pearson_r_threshold=0.5)
    # Test that the shape is correct
    assert shift.shape == (3,)
    # Test that the values are correct
    assert np.allclose(shift, np.array([3, 15, 20]))
    assert np.allclose(shift_corr, 1)


def test_ols_regression():
    # set up data
    rng = np.random.RandomState(0)
    pos = rng.rand(10, 3)
    shift1 = 5 * pos - pos
    # Test the function
    transform = reg_base.ols_regression(shift1, pos)
    # Test that the shape is correct
    assert transform.shape == (3, 4)
    # Test that the values are correct
    assert np.allclose(transform, np.array([[5, 0, 0, 0],
                                            [0, 5, 0, 0],
                                            [0, 0, 5, 0]]))


def test_huber_regression():
    rng = np.random.RandomState(0)
    pos = rng.rand(10, 3)
    shift1 = 5 * pos - pos
    # Test the function
    transform = reg_base.huber_regression(shift1, pos, False)
    # Test that the shape is correct
    assert transform.shape == (3, 4)
    # Test that the values are correct
    assert np.allclose(transform, np.array([[5, 0, 0, 0],
                                            [0, 5, 0, 0],
                                            [0, 0, 5, 0]]), atol=2e-6)


def test_brightness_scale():
    rng = np.random.RandomState(0)
    nx = 4
    ny = 5
    nz = 6
    seq = rng.randint(2**8, dtype=np.int32, size=(nz, ny, nx))
    preseq = reg_pre.custom_shift(seq, [1, 2, 0]) * 4 + 1
    scale, sub_image_seq, sub_image_preseq = reg_base.brightness_scale(preseq, seq, intensity_percentile=0.5)
    assert isinstance(scale, float), "Expected scale to be type float"
    assert isinstance(sub_image_seq, np.ndarray), "Expected sub_image_seq to be type ndarray"
    assert isinstance(sub_image_preseq, np.ndarray), "Expected sub_image_preseq to be type ndarray"
    assert np.isclose(scale, 0.25, atol=1e-2)
