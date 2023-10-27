import pytest
import numpy as np


@pytest.mark.optimised
def test_get_shifts_from_kernel_equality():
    from coppafish.utils.morphology.filter import get_shifts_from_kernel
    from coppafish.utils.morphology.filter_optimised import get_shifts_from_kernel as get_shifts_from_kernel_jax
    
    rng = np.random.RandomState(10)
    kernel = rng.randint(-10, 11, size=(11, 10, 7))
    output = get_shifts_from_kernel(kernel)
    output_jax = get_shifts_from_kernel_jax(kernel)
    for i in range(len(output)):
        assert np.all(output[i] == output_jax[i])
