import pytest
import numpy as np


@pytest.mark.optimised
def test_get_spot_intensity_equality():
    from coppafish.call_spots.qual_check import get_spot_intensity
    from coppafish.call_spots.qual_check_optimised import get_spot_intensity as get_spot_intensity_jax
    rng = np.random.RandomState(8)
    n_spots = 11
    n_rounds = 5
    n_channels = 6
    spot_colours = rng.rand(n_spots, n_rounds, n_channels)
    output = get_spot_intensity(spot_colours)
    assert output.shape == (n_spots, )
    output_jax = get_spot_intensity_jax(spot_colours)
    assert output_jax.shape == (n_spots, )
    assert np.allclose(output, output_jax), 'Expected similar output from jax and non-jax versions'
