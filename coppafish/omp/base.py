import numpy as np
from ..setup import NotebookPage
from ..utils.base import round_any


def get_initial_intensity_thresh(config: dict, nbp: NotebookPage) -> float:
    """
    Gets absolute intensity threshold from config file. OMP will only be run on
    pixels with absolute intensity greater than this.

    Args:
        config: `omp` section of config file.
        nbp: `call_spots` *NotebookPage*

    Returns:
        Either `config['initial_intensity_thresh']` or
            `nbp.abs_intensity_percentile[config['initial_intensity_thresh_percentile']]`.

    """
    initial_intensity_thresh = config['initial_intensity_thresh']
    if initial_intensity_thresh is None:
        config['initial_intensity_thresh'] = \
            round_any(nbp.abs_intensity_percentile[config['initial_intensity_thresh_percentile']],
                      config['initial_intensity_precision'])
    initial_intensity_thresh = \
        float(np.clip(config['initial_intensity_thresh'], config['initial_intensity_thresh_min'],
                      config['initial_intensity_thresh_max']))
    return initial_intensity_thresh
