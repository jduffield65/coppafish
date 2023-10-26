from typing import Optional, Union, List
import numpy as np
import numpy.typing as npt

from ..setup import NotebookPage, Notebook


def get_spot_intensity(spot_colors: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.

    Args:
        spot_colors (`[n_spots x n_rounds x n_channels] ndarray[float]`: spot colors normalised to equalise intensities 
            between channels (and rounds).

    Returns:
        `[n_spots] ndarray[float]`: index `s` is the intensity of spot `s`.

    Notes:
        Logic is that we expect spots that are genes to have at least one large intensity value in each round
        so high spot intensity is more indicative of a gene.
    """
    rng = np.random.RandomState(0)
    check_spot = rng.randint(spot_colors.shape[0])
    diff_to_int = np.round(spot_colors[check_spot]).astype(int) - spot_colors[check_spot]
    if np.abs(diff_to_int).max() == 0:
        raise ValueError(f"spot_intensities should be found using normalised spot_colors."
                         f"\nBut for spot {check_spot}, spot_colors given are integers indicating they are "
                         f"the raw intensities.")
    # Max over all channels, then median over all rounds
    intensities = np.median(np.max(spot_colors, axis=2), axis=1)
    return intensities


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


def quality_threshold(nb: Notebook, method: str = 'omp', intensity_thresh: float = 0,
                      score_thresh: float = 0) -> np.ndarray:
    """
    Indicates which spots pass both the score and intensity quality thresholding.

    Args:
        nb: Notebook containing at least the `ref_spots` page.
        method: `'ref'` or `'omp'` indicating which spots to consider.
        intensity_thresh: Intensity threshold for spots included.
        score_thresh: Score threshold for spots included.

    Returns:
        `bool [n_spots]` indicating which spots pass quality thresholding.

    """
    if method.lower() != 'omp' and method.lower() != 'ref' and method.lower() != 'anchor':
        raise ValueError(f"method must be 'omp' or 'anchor' but {method} given.")
    method_omp = method.lower() == 'omp'
    # If thresholds are not given, get them from config file or notebook (preferably from notebook)
    if intensity_thresh == 0 and score_thresh == 0:
        intensity_thresh = get_intensity_thresh(nb)
        config = nb.get_config()['thresholds']
        score_thresh = config['score_omp'] if method_omp else config['score_ref']
        score_multiplier = config['score_omp_multiplier'] if method_omp else None
        # if thresholds page exists, use those values to override config file
        if nb.has_page('thresholds'):
            score_thresh = nb.thresholds.score_omp if method_omp else nb.thresholds.score_ref
            score_multiplier = nb.thresholds.score_omp_multiplier if method_omp else None
    else:
        score_multiplier = 1

    intensity = nb.omp.intensity if method_omp else nb.ref_spots.intensity
    score = omp_spot_score(nb.omp, score_multiplier) if method_omp else nb.ref_spots.score
    qual_ok = np.array([score > score_thresh, intensity > intensity_thresh]).all(axis=0)
    return qual_ok
