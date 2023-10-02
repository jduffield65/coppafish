from coppafish.utils.base import round_any, setdiff2d, expand_channels

import numpy as np


def test_round_any():
    assert np.allclose(round_any(100., 150., round_type='round'), 150.)
    assert np.allclose(round_any(0.1, 100., round_type='ceil'), 100.)
    assert np.allclose(round_any(0.1, 100., round_type='floor'), 0.)
    # np.ndarray tests
    a = np.array([0.1, 100., 150.])
    assert np.allclose(round_any(a, 100., round_type='round'), np.array([0., 100., 200.]))
    assert np.allclose(round_any(a, 101., round_type='ceil'), np.array([101., 101., 202.]))
    assert np.allclose(round_any(a, 101., round_type='floor'), np.array([0., 0., 101.]))


def test_setdiff2d():
    n_elements = 100
    n_element_size = 10
    rng = np.random.RandomState(20)
    arr1 = rng.rand(n_elements, n_element_size)
    arr2 = np.empty((0,n_element_size))    
    output = setdiff2d(arr1, arr2)
    # We sort results in ascending order to make sure all the elements are there
    assert np.allclose(np.sort(output, axis=0), np.sort(arr1, axis=0))
    # If true, copy element in arr1 into arr2
    copies = rng.randint(2, size=(n_elements), dtype=bool)
    arr2 = np.append(arr2, arr1[copies], axis=0)
    arr2 = np.append(arr2, rng.rand(n_elements, n_element_size) * 2 + 5, axis=0)
    output = setdiff2d(arr1, arr2)
    assert np.allclose(np.sort(output, axis=0), np.sort(arr1[~copies], axis=0))


def test_expand_channels():
    #TODO: Create unit test
    assert True
