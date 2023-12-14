from coppafish.register import base as reg_base
from coppafish.register import preprocessing as reg_pre
from skimage import data

import numpy as np


def test_apply_image_shift():
    rng = np.random.RandomState(0)
    im = rng.randint(np.iinfo(np.uint16).max, size=(2, 3, 4, 5))
    output = reg_pre.apply_image_shift(im, -5)
    assert output.dtype.name == 'int32', "Expected output to be of type `np.int32`"
    assert (output == (im.astype(np.int32) - 5)).all(), "Unexpected output after image shift"


def test_replace_scale():
    # set up transform and scales
    rng = np.random.RandomState(0)
    transform = rng.rand(10, 2, 3, 4)
    scale = np.ones((3, 10, 2))
    # replace scale
    transform = reg_pre.replace_scale(transform, scale)
    # check that all diagonals are 1
    for t in range(10):
        for r in range(2):
            assert np.allclose(np.diag(transform[t, r]), np.ones(3))


def test_populate_full():
    # Setup data
    rng = np.random.RandomState(0)
    sublist1, sublist2 = [0, 2], [1, 3]
    list1, list2 = [0, 1, 2, 3], [0, 1, 2, 3]
    array = rng.rand(2, 2, 3, 4)
    # Test the function
    array_new = reg_pre.populate_full(sublist1, list1, sublist2, list2, array)
    # This should have 2 * 2 non-zero matrices
    # All odd matrices in axis 0 should be 0, all even matrices in axis 1 should be 0
    assert np.count_nonzero(array_new[:, :, 0, 0]) == 2 * 2
    assert np.all(array_new[1::2, :] == 0)
    assert np.all(array_new[:, ::2] == 0)


def test_yxz_to_zyx():
    # Setup data
    rng = np.random.RandomState(0)
    im = rng.rand(1, 2, 3)
    new_im = reg_pre.yxz_to_zyx(im)
    # Test that the shape is correct
    assert new_im.shape == (3, 1, 2)


def test_n_matches_to_frac_matches():
    # Setup data
    rng = np.random.RandomState(0)
    n_matches = rng.randint(0, 10, size=(3,4,10))
    spot_no = np.ones((3, 4)) * 10
    # Test the function
    frac_matches = reg_pre.n_matches_to_frac_matches(n_matches, spot_no)
    # Test that the shape is correct
    assert frac_matches.shape == (3, 4, 10)
    # Test that the values are correct
    assert np.allclose(frac_matches, n_matches / 10)


def test_split_3d_image():
    # Setup data (10, 256, 256)
    brain = data.brain()
    # Test the function
    z_box, y_box, x_box = 6, 64, 64
    brain_split, pos = reg_pre.split_3d_image(brain, 2, 4, 4, z_box, y_box, x_box)
    # Test that the shape is correct
    assert brain_split.shape == (2, 4, 4, 6, 64, 64)
    # Test that the values are correct
    assert np.allclose(brain_split[0, 0, 0], brain[:6, :64, :64])
    # Test that the positions are correct
    assert all(pos[0] == [z_box // 2, y_box // 2, x_box // 2])


def test_compose_affine():
    # Setup data
    matrix1 = 2 * np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])
    matrix2 = 3 * np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])
    matrix3 = np.array([[1, 0, 0, 1],
                        [0, 1, 0, 2],
                        [0, 0, 1, 3]])
    matrix4 = np.array([[1, 0, 0, 4],
                        [0, 1, 0, 5],
                        [0, 0, 1, 6]])
    # Test the function
    scaled_matrix = reg_pre.compose_affine(matrix1, matrix2)
    summed_matrix = reg_pre.compose_affine(matrix3, matrix4)
    # Test that the shape is correct
    assert scaled_matrix.shape == (3, 4)
    assert summed_matrix.shape == (3, 4)
    # Test that the values are correct
    assert np.allclose(scaled_matrix, 6 * np.eye(3, 4))
    assert np.allclose(summed_matrix, np.array([[1, 0, 0, 5],
                                                [0, 1, 0, 7],
                                                [0, 0, 1, 9]]))


def test_invert_affine():
    # set up data
    rng = np.random.RandomState(0)
    affine = rng.rand(3, 4)
    # invert affine
    affine_inv = reg_pre.invert_affine(affine)
    # check that the inverse is correct
    assert np.allclose(reg_pre.compose_affine(affine, affine_inv), np.eye(3, 4))
    assert np.allclose(reg_pre.compose_affine(affine_inv, affine), np.eye(3, 4))


def test_yxz_to_zyx_affine():
    # set up data
    matrix_yxz = np.array([[1, 0, 0],
                           [0, 2, 0],
                           [0, 0, 3],
                           [1, 2, 3]])
    # convert to zyx
    matrix_zyx = reg_pre.yxz_to_zyx_affine(matrix_yxz)
    # check that the shape is correct
    assert matrix_zyx.shape == (3, 4)
    # check that the values are correct
    assert np.allclose(matrix_zyx, np.array([[3, 0, 0, 3],
                                             [0, 1, 0, 1],
                                             [0, 0, 2, 2]]))


def test_zyx_to_yxz_affine():
    # set up data
    matrix_zyx = np.array([[3, 0, 0, 3],
                           [0, 1, 0, 1],
                           [0, 0, 2, 2]]).astype(float)
    # convert to zyx
    matrix_yxz = reg_pre.zyx_to_yxz_affine(matrix_zyx)
    # check that the shape is correct
    assert matrix_yxz.shape == (4, 3)
    # check that the values are correct
    assert np.allclose(matrix_yxz, np.array([[1, 0, 0],
                                             [0, 2, 0],
                                             [0, 0, 3],
                                             [1, 2, 3]]))


def test_custom_shift():
    # set up data
    im = np.sum(data.astronaut(), axis=2)
    shift = np.array([10, 20]).astype(int)
    im_new = reg_pre.custom_shift(im, shift)
    # check that the shape is correct
    assert im_new.shape == im.shape
    # check that the values are correct
    assert np.allclose(im_new[10:, 20:], im[:-10, :-20])
    assert np.allclose(im_new[:10, :20], 0)


def test_merge_subvols():
    # set up data
    rng = np.random.RandomState(0)
    subvols = rng.rand(2, 3, 4, 5)
    pos = np.array([[0, 0, 0],
                    [10, 10, 10]])
    subvol_3 = np.ones((3, 4, 5))
    pos_3 = np.array([1, 1, 1])
    # merge subvols
    merged = reg_pre.merge_subvols(pos, subvols)
    # TODO: Do this after the meeting
    # check that the shape is correct
    assert merged.shape == (13, 14, 15)
    # check that the values are correct
    assert np.allclose(merged[:3, :4, :5], subvols[0])
    assert np.allclose(merged[10:, 10:, 10:], subvols[1])

