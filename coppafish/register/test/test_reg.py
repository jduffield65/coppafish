import numpy as np
from coppafish.register import base as reg_base
from coppafish.register import preprocessing as reg_pre
from skimage import data


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
    im = rng.rand(10, 20, 30)
    new_im = reg_pre.yxz_to_zyx(im)
    # Test that the shape is correct
    assert new_im.shape == (30, 10, 20)


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
    # merge subvols
    merged = reg_pre.merge_subvols(pos, subvols)
    # check that the shape is correct
    assert merged.shape == (13, 14, 15)
    # check that the values are correct
    assert np.allclose(merged[:3, :4, :5], subvols[0])
    assert np.allclose(merged[10:, 10:, 10:], subvols[1])


def test_find_shift_array():
    # set up data (10, 256, 256)
    brain = data.brain()
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
    cell = data.cells3d()[:, 1]
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
    kidney = np.sum(data.kidney(), axis=-1)[:, :128, :128]
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