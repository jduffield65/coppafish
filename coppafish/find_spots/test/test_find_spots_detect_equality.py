import numpy as np
import pytest


@pytest.mark.optimised
def test_detect_spots_equality():
    from coppafish.find_spots.detect import detect_spots
    from coppafish.find_spots.detect_optimised import detect_spots as detect_spots_jax
    
    rng = np.random.RandomState(8)
    n_x = 9
    n_y = 10
    n_z = 11
    image = rng.rand(n_y, n_x, n_z)
    intensity_thresh = rng.rand() * 0.5
    radius_xy = 3
    radius_z = 2
    for remove_duplicates in [True, False]:
        peak_yxz, peak_intensity = detect_spots(image, intensity_thresh, radius_xy, radius_z, remove_duplicates)
        n_peaks = peak_yxz.shape[0]
        assert peak_yxz.shape == (n_peaks, image.ndim)
        assert peak_intensity.shape == (n_peaks, )
        peak_yxz_jax, peak_intensity_jax = detect_spots_jax(image, intensity_thresh, radius_xy, radius_z, 
                                                            remove_duplicates)
        n_peaks = peak_yxz_jax.shape[0]
        assert peak_yxz_jax.shape == (n_peaks, image.ndim)
        assert peak_intensity_jax.shape == (n_peaks, )
        assert np.allclose(peak_yxz, peak_yxz_jax)
        assert np.allclose(peak_intensity, peak_intensity_jax)
