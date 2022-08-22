import numpy as np
from tqdm import tqdm
from ..stitch import compute_shift, update_shifts
from ..find_spots import spot_yxz
from ..setup.notebook import NotebookPage
import warnings


def register_initial(config: dict, nbp_basic: NotebookPage, spot_details: np.ndarray) -> NotebookPage:
    """
    This finds the shift between ref round/channel to each imaging round for each tile.
    These are then used as the starting point for determining the affine transforms in `pipeline/register.py`.

    See `'register_initial'` section of `notebook_comments.json` file
    for description of the variables in the page.

    Args:
        config: Dictionary obtained from `'register_initial'` section of config file.
        nbp_basic: `basic_info` notebook page
        spot_details: `int [n_spots x 7]`.
            `spot_details[s]` is `[tile, round, channel, isolated, y, x, z]` of spot `s`.
            This is saved in the find_spots notebook page i.e. `nb.find_spots.spot_details`.

    Returns:
        `NotebookPage[register_initial]` - Page contains information about how shift between ref round/channel
            to each imaging round for each tile was found.
    """
    nbp_debug = NotebookPage("register_initial")
    if config['shift_channel'] is None:
        config['shift_channel'] = nbp_basic.ref_channel
    if not np.isin(config['shift_channel'], nbp_basic.use_channels):
        raise ValueError(f"config['shift_channel'] should be in nb.basic_info.use_channels, but value given is\n"
                         f"{config['shift_channel']} which is not in use_channels = {nbp_basic.use_channels}.")
    nbp_debug.shift_channel = config['shift_channel']

    coords = ['y', 'x', 'z']
    shifts = [{}]
    start_shift_search = np.zeros((nbp_basic.n_rounds, 3, 3), dtype=int)
    for i in range(len(coords)):
        shifts[0][coords[i]] = np.arange(config['shift_min'][i],
                                         config['shift_max'][i] +
                                         config['shift_step'][i] / 2, config['shift_step'][i]).astype(int)
        start_shift_search[nbp_basic.use_rounds, i, :] = [config['shift_min'][i], config['shift_max'][i],
                                                          config['shift_step'][i]]
    if not nbp_basic.is_3d:
        config['shift_widen'][2] = 0  # so don't look for shifts in z direction
        config['shift_max_range'][2] = 0
        shifts[0]['z'] = np.array([0], dtype=int)
        start_shift_search[:, 2, :2] = 0
    shifts = shifts * nbp_basic.n_rounds  # get one set of shifts for each round

    shift = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds, 3), dtype=int)
    shift_score = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds), dtype=float)
    shift_score_thresh = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds), dtype=float)

    c_ref = nbp_basic.ref_channel
    r_ref = nbp_basic.ref_round
    c_imaging = config['shift_channel']
    # to convert z coordinate units to xy pixels when calculating distance to nearest neighbours
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
    with tqdm(total=len(nbp_basic.use_rounds) * len(nbp_basic.use_tiles)) as pbar:
        pbar.set_description(f"Finding shift from ref_round({r_ref})/ref_channel({c_ref}) to channel {c_imaging} "
                             f"of all imaging rounds")
        for r in nbp_basic.use_rounds:
            for t in nbp_basic.use_tiles:
                pbar.set_postfix({'round': r, 'tile': t})
                shift[t, r], shift_score[t, r], shift_score_thresh[t, r] = \
                    compute_shift(spot_yxz(spot_details, t, r_ref, c_ref), spot_yxz(spot_details, t, r, c_imaging),
                                  config['shift_score_thresh'], config['shift_score_thresh_multiplier'],
                                  config['shift_score_thresh_min_dist'], config['shift_score_thresh_max_dist'],
                                  config['neighb_dist_thresh'], shifts[r]['y'], shifts[r]['x'], shifts[r]['z'],
                                  config['shift_widen'], config['shift_max_range'], z_scale,
                                  config['nz_collapse'], config['shift_step'][2])[:3]
                good_shifts = shift_score[:, r] > shift_score_thresh[:, r]
                if np.sum(good_shifts) >= 3:
                    # once found shifts, refine shifts to be searched around these
                    for i in range(len(coords)):
                        shifts[r][coords[i]] = update_shifts(shifts[r][coords[i]], shift[good_shifts, r, i])
                pbar.update(1)
    pbar.close()

    # amend shifts for which score fell below score_thresh
    shift_outlier = shift.copy()
    shift_score_outlier = shift_score.copy()
    n_shifts = len(nbp_basic.use_tiles)
    final_shift_search = np.zeros_like(start_shift_search)
    final_shift_search[:, :, 2] = start_shift_search[:, :, 2]  # spacing does not change
    for r in nbp_basic.use_rounds:
        good_shifts = shift_score[:, r] > shift_score_thresh[:, r]
        for i in range(len(coords)):
            # change shift search to be near good shifts found
            # this will only do something if 3>sum(good_shifts)>0, otherwise will have been done in previous loop.
            if np.sum(good_shifts) > 0:
                shifts[r][coords[i]] = update_shifts(shifts[r][coords[i]], shift[good_shifts, r, i])
            elif good_shifts.size > 0:
                shifts[r][coords[i]] = update_shifts(shifts[r][coords[i]], shift[:, r, i])
        final_shift_search[r, :, 0] = [np.min(shifts[r][key]) for key in shifts[r].keys()]
        final_shift_search[r, :, 1] = [np.max(shifts[r][key]) for key in shifts[r].keys()]
        shift_outlier[good_shifts, r] = 0  # only keep outlier information for not good shifts
        shift_score_outlier[good_shifts, r] = 0
        if (np.sum(good_shifts) < 2 and n_shifts > 4) or (np.sum(good_shifts) == 0 and n_shifts > 0):
            warnings.warn(f"Round {r}: {n_shifts - np.sum(good_shifts)}/{n_shifts} "
                          f"of shifts fell below score threshold")
        for t in np.where(good_shifts == False)[0]:
            if t not in nbp_basic.use_tiles:
                continue
            # re-find shifts that fell below threshold by only looking at shifts near to others found
            # score set to 0 so will find do refined search no matter what.
            shift[t, r], shift_score[t, r] = compute_shift(spot_yxz(spot_details, t, r_ref, c_ref),
                                                           spot_yxz(spot_details, t, r, c_imaging), 0, None, None,
                                                           None, config['neighb_dist_thresh'], shifts[r]['y'],
                                                           shifts[r]['x'], shifts[r]['z'], None, None, z_scale,
                                                           config['nz_collapse'], config['shift_step'][2])[:2]
            warnings.warn(f"\nShift for tile {t} to round {r} changed from\n"
                          f"{shift_outlier[t, r]} to {shift[t, r]}.")

    nbp_debug.shift = shift
    nbp_debug.start_shift_search = start_shift_search
    nbp_debug.final_shift_search = final_shift_search
    nbp_debug.shift_score = shift_score
    nbp_debug.shift_score_thresh = shift_score_thresh
    nbp_debug.shift_outlier = shift_outlier
    nbp_debug.shift_score_outlier = shift_score_outlier

    return nbp_debug
