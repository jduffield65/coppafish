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
                            f"coppafish.plot.view_stitch_shift_info\ncoppafish.plot.view_stitch\n" \
                            f"coppafish.plot.view_stitch_overlap\n" \
                            f"coppafish.plot.view_stitch_search\nIf stitching looks wrong, maybe try re-running with " \
                            f"different configuration parameters e.g. smaller shift_step or larger shift_max_range."
        raise ValueError(f"{message}")
    elif n_fail >= 1:
        warnings.warn(message)


def check_shifts_register(nb: Notebook):
    """
    This checks that a decent number of shifts computed in the `register_initial` stage of the pipeline
    are acceptable (`score > score_thresh`).

    An error will be raised if the fraction of shifts with `score < score_thresh`
    exceeds `config['register_initial']['n_shifts_error_fraction']`.

    Args:
        nb: *Notebook* containing `stitch` page.
    """
    r_ref = nb.basic_info.ref_round
    c_ref = nb.basic_info.ref_channel
    c_shift = nb.register_initial.shift_channel
    use_rounds = np.asarray(nb.basic_info.use_rounds)
    use_tiles = np.asarray(nb.basic_info.use_tiles)
    n_shifts = len(use_rounds) * len(use_tiles)
    n_fail = 0
    config = nb.get_config()['register_initial']
    shift = nb.register_initial.shift
    score = nb.register_initial.shift_score
    score_thresh = nb.register_initial.shift_score_thresh
    fail_info = np.zeros((0, 7), dtype=int)
    for r in nb.basic_info.use_rounds:
        fail_tiles = use_tiles[np.where((score[use_tiles, r] < score_thresh[use_tiles, r]).flatten())[0]]
        n_fail += len(fail_tiles)
        if len(fail_tiles) > 0:
            fail_info_r = np.zeros((len(fail_tiles), 7), dtype=int)
            fail_info_r[:, 0] = r
            fail_info_r[:, 1] = fail_tiles
            fail_info_r[:, 2:5] = shift[fail_tiles, r]
            fail_info_r[:, 5] = score[fail_tiles, r].flatten()
            fail_info_r[:, 6] = score_thresh[fail_tiles, r].flatten()
            fail_info = np.append(fail_info, fail_info_r, axis=0)
    if n_fail >= 1:
        message = f"\nInfo for the {n_fail} shifts from round {r_ref}/channel {c_ref} to channel {c_shift}" \
                  f" with score < score_thresh:\n" \
                  f"Round, Tile, Y shift, X shift, Z shift, score, score_thresh\n" \
                  f"{fail_info}"
        n_error_thresh = int(np.floor(config['n_shifts_error_fraction'] * n_shifts))
        if n_fail > n_error_thresh:
            message = message + f"\n{n_fail}/{n_shifts} shifts have score < score_thresh.\n" \
                                f"This exceeds error threshold of {n_error_thresh}.\nLook at the following " \
                                f"diagnostics to decide if shifts are acceptable to continue:\n" \
                                f"coppafish.plot.view_register_shift_info\ncoppafish.plot.view_register_search\n" \
                                f"coppafish.plot.view_icp\n" \
                                f"If shifts looks wrong, maybe try re-running with " \
                                f"different configuration parameters e.g. smaller shift_step or larger shift_max_range."

            # Recommend channel with the most spots on the tile/round for which it has the least.
            spot_no = nb.find_spots.spot_no[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds)]
            # For each channel this is number of spots on tile/round with the least spots.
            spot_no = np.min(spot_no, axis=(0, 1))
            c_most_spots = np.argmax(spot_no)
            if c_most_spots != nb.register_initial.shift_channel:
                message = message + f"\nAlso consider changing config['register_initial']['shift_channel']. " \
                                    f"Current channel {c_ref} has at least {spot_no[c_ref]} on all tiles and rounds " \
                                    f"but channel {c_most_spots} has at least {spot_no[c_most_spots]}."
            raise ValueError(f"{message}")
        else:
            warnings.warn(message)
