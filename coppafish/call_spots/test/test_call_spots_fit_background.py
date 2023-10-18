import numpy as np

from coppafish.call_spots.background import fit_background


def test_fit_background():
    n_spots = 2
    n_rounds = 1
    n_channels = 2
    # rng = np.random.RandomState(0)
    spot_colours = np.empty((n_spots, n_rounds, n_channels))
    spot_colours[0,:,:] = 1
    spot_colours[1,:,:] = 0

    # Test with two spots, first one is just ones in every round and channel, the other is just zeroes everywhere.
    # The residual should then become just zeroes everywhere when the weight_shift is zero/super close to zero.
    residual1, coef1, background_vectors1 = fit_background(spot_colours, weight_shift=0)
    assert residual1.shape == spot_colours.shape, \
        'Expected outputted residual to be the same shape as the spot_colours array'
    assert coef1.shape == (n_spots, n_channels), 'Expected coefs to have shape n_spots x n_channels'
    assert background_vectors1.shape == (n_channels, n_rounds, n_channels), \
        'Expected coefs to have shape n_channels x n_spots x n_channels'
    assert np.allclose(residual1, 0), 'Expected all residuals to become zero after background fitting'
    # Create a random pattern of only background noise and check it removes it
    n_spots = 3
    n_rounds = 4
    n_channels = 5
    rng = np.random.RandomState(81)
    # Weighting given to background vector
    bg_weightings = rng.random((n_channels))
    spot_colours = np.ones((n_spots, n_rounds, n_channels))
    for c in range(n_channels):
        spot_colours[:,:,c] *= bg_weightings[c]
    residual1, coef1, background_vectors1 = fit_background(spot_colours, weight_shift=0)
    assert residual1.shape == spot_colours.shape, \
        'Expected outputted residual to be the same shape as the spot_colours array'
    assert coef1.shape == (n_spots, n_channels), 'Expected coefs to have shape n_spots x n_channels'
    assert background_vectors1.shape == (n_channels, n_rounds, n_channels), \
        'Expected coefs to have shape n_channels x n_spots x n_channels'
    assert np.allclose(residual1, 0), 'Expected all residuals to become near zero after background fitting'
    # Test weight_shift by seeing if it reduces the variance when increased
    spot_colours = rng.random((n_spots, n_rounds, n_channels)) * 10
    residual1, coef1, background_vectors1 = fit_background(spot_colours, weight_shift=0)
    residual2, coef2, background_vectors2 = fit_background(spot_colours, weight_shift=10)
    assert (residual1 != residual2).all(), 'Expecting different results when changing weight_shift'
    assert np.std(residual1) > np.std(residual2), 'Expected weight_shift to reduce the variance in the residuals'
