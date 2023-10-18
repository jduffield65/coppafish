import numpy as np
import pytest


@pytest.mark.optimised
def test_get_spot_intensity():
    from coppafish.call_spots import qual_check_optimised
    n_spots = 2
    n_rounds = 3
    n_channels = 4
    spot_colors = np.zeros((n_spots, n_rounds, n_channels), dtype=float)
    spot_colors[0,0,0] = 1.
    spot_colors[1,0,3] = 2.
    spot_colors[1,1,3] = 2.
    output = qual_check_optimised.get_spot_intensity(spot_colors)
    assert output.ndim == 1, 'Expect a vector output'
    assert output.size == n_spots, 'Expect vector dimensions to be n_spots'
    assert np.allclose(output[0], 0.), 'Expect first spot median to be zero'
    assert np.allclose(output[1], 2.), 'Expect second spot median to be 2'
