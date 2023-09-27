from coppafish.utils.base import round_any

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
