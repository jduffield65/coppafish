from typing import Optional, Union, List
import numpy as np
from ..setup import NotebookPage, Notebook


def get_spot_intensity(spot_colors: np.ndarray) -> np.ndarray:
    """
    Finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.
    Logic is that we expect spots that are genes to have at least one large intensity value in each round
    so high spot intensity is more indicative of a gene.

    Args:
        spot_colors: ```float [n_spots x n_rounds x n_channels]```.
            Spot colors normalised to equalise intensities between channels (and rounds).

    Returns:
        ```float [n_spots]```.
            ```[s]``` is the intensity of spot ```s```.
    """
    check_spot = np.random.randint(spot_colors.shape[0])
    diff_to_int = np.round(spot_colors[check_spot]).astype(int) - spot_colors[check_spot]
    if np.abs(diff_to_int).max() == 0:
        raise ValueError(f"spot_intensities should be found using normalised spot_colors."
                         f"\nBut for spot {check_spot}, spot_colors given are integers indicating they are "
                         f"the raw intensities.")
    round_max_color = np.max(spot_colors, axis=2)
    return np.median(round_max_color, axis=1)


def omp_spot_score(nbp: NotebookPage, score_multiplier: float,
                   spot_no: Optional[Union[int, List, np.ndarray]] = None,
                   n_neighbours_pos: Optional[Union[np.ndarray, int]] = None,
                   n_neighbours_neg: Optional[Union[np.ndarray, int]] = None) -> Union[float, np.ndarray]:
    """
    Score for omp gene assignment

    Args:
        nbp: OMP Notebook page
        score_multiplier: `score = score_multiplier * n_pos_neighb + n_neg_neighb`.
            So this influences the importance of positive coefficient neighbours vs negative.
        spot_no: Which spots to get score for. If `None`, all scores will be found.

    Returns:
        Score for each spot in spot_no if given, otherwise all spot scores.
    """
    max_score = score_multiplier * np.sum(nbp.spot_shape == 1) + np.sum(nbp.spot_shape == -1)
    if n_neighbours_pos is None:
        n_neighbours_pos = nbp.n_neighbours_pos
    if n_neighbours_neg is None:
        n_neighbours_neg = nbp.n_neighbours_neg
    if spot_no is None:
        score = (score_multiplier * n_neighbours_pos + n_neighbours_neg) / max_score
    else:
        score = (score_multiplier * n_neighbours_pos[spot_no] + n_neighbours_neg[spot_no]) / max_score
    return score


def get_intensity_thresh(nb: Notebook) -> float:
    """
    Gets threshold for intensity from parameters in `config file` or Notebook.

    Args:
        nb: Notebook containing at least the `call_spots` page.

    Returns:
        intensity threshold
    """
    if nb.has_page('thresholds'):
        intensity_thresh = nb.thresholds.intensity
    else:
        config = nb.get_config()['thresholds']
        intensity_thresh = config['intensity']
        if intensity_thresh is None:
            intensity_thresh = nb.call_spots.gene_efficiency_intensity_thresh
    return intensity_thresh


def quality_threshold(nb: Notebook, method: str = 'omp') -> np.ndarray:
    """
    Indicates which spots pass both the score and intensity quality thresholding.

    Args:
        nb: Notebook containing at least the `ref_spots` page.
        method: `'ref'` or `'omp'` indicating which spots to consider.

    Returns:
        `bool [n_spots]` indicating which spots pass quality thresholding.

    """
    if method.lower() != 'omp' and method.lower() != 'ref' and method.lower() != 'anchor':
        raise ValueError(f"method must be 'omp' or 'anchor' but {method} given.")
    intensity_thresh = get_intensity_thresh(nb)
    if nb.has_page('thresholds'):
        if method.lower() == 'omp':
            score_thresh = nb.thresholds.score_omp
            score_multiplier = nb.thresholds.score_omp_multiplier
        else:
            score_thresh = nb.thresholds.score_ref
    else:
        config = nb.get_config()['thresholds']
        if method.lower() == 'omp':
            score_thresh = config['score_omp']
            score_multiplier = config['score_omp_multiplier']
        else:
            score_thresh = config['score_ref']
    if method.lower() == 'omp':
        intensity = nb.omp.intensity
        score = omp_spot_score(nb.omp, score_multiplier)
    else:
        intensity = nb.ref_spots.intensity
        score = nb.ref_spots.score
    qual_ok = np.array([score > score_thresh, intensity > intensity_thresh]).all(axis=0)
    return qual_ok
