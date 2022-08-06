import warnings
import numpy as np
from ..setup import Notebook


def check_shifts_stitch(nb: Notebook):
    """
    This checks that a decent number of shifts computed in the stitch stage of the pipeline
    are acceptable (`score > score_thresh`).

    An error will be raised if the fraction of shifts with `score < score_thresh`
    exceeds `config['stitch']['n_shifts_error_fraction']`.

    Args:
        nb: *Notebook* containing `stitch` page.
    """
    n_shifts = 0
    n_fail = 0
    message = ""
    config = nb.get_config()['stitch']
    directions = ['south', 'west']
    dir_opp = {'south': 'north', 'west': 'east'}
    for j in directions:
        n_shifts += len(nb.stitch.__getattribute__(f"{j}_score"))
        fail_ind = np.where((nb.stitch.__getattribute__(f"{j}_score") <
                                nb.stitch.__getattribute__(f"{j}_score_thresh")).flatten())[0]
        n_fail += len(fail_ind)
        if len(fail_ind) > 0:
            fail_info = np.zeros((len(fail_ind), 7), dtype=int)
            fail_info[:, :2] = nb.stitch.__getattribute__(f"{j}_pairs")[fail_ind]
            fail_info[:, 2:5] = nb.stitch.__getattribute__(f"{j}_shifts")[fail_ind]
            fail_info[:, 5] = nb.stitch.__getattribute__(f"{j}_score")[fail_ind].flatten()
            fail_info[:, 6] = nb.stitch.__getattribute__(f"{j}_score_thresh")[fail_ind].flatten()
            message = message + f"\nInfo for the {len(fail_ind)} shifts with score < score_thresh in {j} direction:\n" \
                                f"Tile, Tile to {dir_opp[j]}, Y shift, X shift, Z shift, score, score_thresh\n" \
                                f"{fail_info}"
    n_error_thresh = int(np.floor(config['n_shifts_error_fraction'] * n_shifts))
    if n_fail > n_error_thresh:
        message = message + f"\n{n_fail}/{n_shifts} shifts have score < score_thresh.\n" \
                            f"This exceeds error threshold of {n_error_thresh}.\nLook at the following diagnostics " \
                            f"to decide if stitching is acceptable to continue:\n" \
                            f"iss.plot.view_stitch_shift_info\niss.plot.view_stitch\niss.plot.view_stitch_overlap"
        raise ValueError(f"{message}")
    elif n_fail >= 1:
        warnings.warn(message)
