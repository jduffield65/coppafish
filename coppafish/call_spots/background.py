import numpy as np
from typing import Tuple


def fit_background(spot_colors: np.ndarray, weight_shift: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This determines the coefficient of the background vectors for each spot.
    Coefficients determined using a weighted dot product as to avoid overfitting
    and accounting for the fact that background coefficients are not updated after this.

    !!! note
        `background_vectors[i]` is 1 in channel `i` for all rounds and 0 otherwise.
        It is then normalised to have L2 norm of 1 when summed over all rounds and channels.
    !!! note
        If weight_shift < 1e-20, then it is set to 1e-20 to avoid any divergence.

    Args:
        spot_colors: `float [n_spots x n_rounds x n_channels]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        weight_shift: shift to apply to weighting of each background vector to limit boost of weak spots.

    Returns:
        - residual - `float [n_spots x n_rounds x n_channels]`.
            `spot_colors` after background removed.
        - coef - `float [n_spots, n_channels]`.
            coefficient value for each background vector found for each spot.
        - background_vectors `float [n_channels x n_rounds x n_channels]`.
            background_vectors[c] is the background vector for channel c.

    """
    # if weight_shift < 1e-20:
    #     warnings.warn(f'weight_shift value given, {weight_shift} is below 1e-20.'
    #                   f'Using weight_shift=1e-20 to stop blow up to infinity.')
    weight_shift = np.clip(weight_shift, 1e-20, np.inf)  # ensure weight_shift > 1e-20 to avoid blow up to infinity.

    n_rounds, n_channels = spot_colors[0].shape
    background_vectors = np.repeat(np.expand_dims(np.eye(n_channels), axis=1), n_rounds, axis=1)
    # give background_vectors an L2 norm of 1 so can compare coefficients with other genes.
    background_vectors = background_vectors / np.linalg.norm(background_vectors, axis=(1, 2), keepdims=True)

    weight_factor = 1 / (np.abs(spot_colors) + weight_shift)
    spot_weight = spot_colors * weight_factor
    background_weight = np.ones((1, n_rounds, n_channels)) * background_vectors[0, 0, 0] * weight_factor
    coef = np.sum(spot_weight * background_weight, axis=1) / np.sum(background_weight ** 2, axis=1)
    residual = spot_colors - np.expand_dims(coef, 1) * np.ones((1, n_rounds, n_channels)) * background_vectors[0, 0, 0]

    return residual, coef, background_vectors
