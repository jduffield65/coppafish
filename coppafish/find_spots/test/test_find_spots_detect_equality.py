import numpy as np
import pytest

from coppafish import utils


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


@pytest.mark.optimised
def test_get_local_maxima_equality():
    from coppafish.find_spots.detect import get_local_maxima
    from coppafish.find_spots.detect_optimised import get_local_maxima_jax

    rng = np.random.RandomState(36)
    image = rng.rand(9, 10, 11)
    se = rng.randint(2, size=(4, 5, 6))
    se_shifts = utils.morphology.filter_optimised.get_shifts_from_kernel(se)
    pad_size_y = 1
    pad_size_x = 2
    pad_size_z = 3
    intensity_thresh = 0.3
    consider_yxz = np.where(image > intensity_thresh)
    consider_intensity = image[consider_yxz]
    consider_yxz = list(consider_yxz)
    output = get_local_maxima(image, se_shifts, pad_size_y, pad_size_x, pad_size_z, consider_yxz, 
                              consider_intensity)
    assert output.shape == (consider_intensity.size, )
    output_jax = get_local_maxima_jax(image, se_shifts, pad_size_y, pad_size_x, pad_size_z, 
                                      consider_yxz, consider_intensity)
    assert output_jax.shape == (consider_intensity.size, )
    assert np.allclose(output, output_jax)
