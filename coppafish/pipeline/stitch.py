from tqdm import tqdm
import numpy as np
import warnings

from .. import utils
from ..stitch import starting_shifts as stitch_starting_shifts
from ..stitch import shift as stich_shift
from ..stitch import tile_origin as stitch_tile_origin
from ..find_spots import base as find_spots_base
from ..setup.notebook import NotebookPage


def stitch(config: dict, nbp_basic: NotebookPage, local_yxz: np.ndarray, spot_no: np.ndarray) -> NotebookPage:
    """
    This gets the origin of each tile such that a global coordinate system can be built.

    See `'stitch'` section of `notebook_comments.json` file
    for description of the variables in the page.

    Args:
        config: Dictionary obtained from `'stitch'` section of config file.
        nbp_basic: `basic_info` notebook page
        local_yxz: `int [n_spots x 3]`.
            `spot_details[s]` is `[y, x, z]` of spot `s`.
            This is saved in the find_spots notebook page i.e. `nb.find_spots.local_yxz`.
        spot_no: 'int[n_tiles x n_rounds x n_channels]'
            'spot_no[t,r,c]' is num_spots found on that [t,r,c]
            This is saved on find_spots notebook page

    Returns:
        `NotebookPage[stitch]` - Page contains information about how tiles were stitched together to give
            global coordinates.
    """
    nbp_debug = NotebookPage("stitch")
    nbp_debug.software_version = utils.system.get_software_verison()
    nbp_debug.revision_hash = utils.system.get_git_revision_hash()
    directions = ['north', 'east']
    coords = ['y', 'x', 'z']
    shifts = stitch_starting_shifts.get_shifts_to_search(config, nbp_basic, nbp_debug)

    if not nbp_basic.is_3d:
        config['nz_collapse'] = None
        config['shift_widen'][2] = 0  # so don't look for shifts in z direction
        config['shift_max_range'][2] = 0

    # initialise variables to store shift info
    shift_info = {'north': {}, 'east': {}}
    for j in directions:
        shift_info[j]['pairs'] = np.zeros((0, 2), dtype=int)
        shift_info[j]['shifts'] = np.zeros((0, 3), dtype=int)
        shift_info[j]['score'] = np.zeros((0, 1), dtype=float)
        shift_info[j]['score_thresh'] = np.zeros((0, 1), dtype=float)

    # find shifts between overlapping tiles
    c = nbp_basic.anchor_channel
    r = nbp_basic.anchor_round
    t_neighb = {'north': [], 'east': []}
    # to convert z coordinate units to xy pixels when calculating distance to nearest neighbours
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
    with tqdm(total=2 * len(nbp_basic.use_tiles)) as pbar:
        pbar.set_description(f"Finding overlap between tiles in round {r} (ref_round)")
        for t in nbp_basic.use_tiles:
            # align to north neighbour followed by east neighbour
            t_neighb['north'] = np.where(np.sum(nbp_basic.tilepos_yx == nbp_basic.tilepos_yx[t, :] + [1, 0],
                                                axis=1) == 2)[0]
            t_neighb['east'] = np.where(np.sum(nbp_basic.tilepos_yx == nbp_basic.tilepos_yx[t, :] + [0, 1],
                                               axis=1) == 2)[0]
            for j in directions:
                pbar.set_postfix({'tile': t, 'direction': j})
                if t_neighb[j].size > 0 and t_neighb[j] in nbp_basic.use_tiles:
                    shift, score, score_thresh = stich_shift.compute_shift(
                        find_spots_base.spot_yxz(local_yxz, t, r, c, spot_no),
                        find_spots_base.spot_yxz(local_yxz, t_neighb[j][0], r, c, spot_no),
                        config['shift_score_thresh'],
                        config['shift_score_thresh_multiplier'],
                        config['shift_score_thresh_min_dist'],
                        config['shift_score_thresh_max_dist'],
                        config['neighb_dist_thresh'], shifts[j]['y'],
                        shifts[j]['x'], shifts[j]['z'],
                        config['shift_widen'], config['shift_max_range'],
                        z_scale, config['nz_collapse'],
                        config['shift_step'][2]
                    )[:3]
                    shift_info[j]['pairs'] = np.append(shift_info[j]['pairs'],
                                                       np.array([t, t_neighb[j][0]]).reshape(1, 2), axis=0)
                    shift_info[j]['shifts'] = np.append(shift_info[j]['shifts'], np.array(shift).reshape(1, 3), axis=0)
                    shift_info[j]['score'] = np.append(shift_info[j]['score'], np.array(score).reshape(1, 1), axis=0)
                    shift_info[j]['score_thresh'] = np.append(shift_info[j]['score_thresh'],
                                                              np.array(score_thresh).reshape(1, 1), axis=0)
                    good_shifts = (shift_info[j]['score'] > shift_info[j]['score_thresh']).flatten()
                    if np.sum(good_shifts) >= 3:
                        # once found shifts, refine shifts to be searched around these
                        for i in range(len(coords)):
                            shifts[j][coords[i]] = stich_shift.update_shifts(shifts[j][coords[i]], 
                                                                             shift_info[j]['shifts'][good_shifts, i])
                pbar.update(1)
    pbar.close()

    # amend shifts for which score fell below score_thresh
    for j in directions:
        good_shifts = (shift_info[j]['score'] > shift_info[j]['score_thresh']).flatten()
        for i in range(len(coords)):
            # change shift search to be near good shifts found
            # this will only do something if 3>sum(good_shifts)>0, otherwise will have been done in previous loop.
            if np.sum(good_shifts) > 0:
                shifts[j][coords[i]] = stich_shift.update_shifts(shifts[j][coords[i]], 
                                                                 shift_info[j]['shifts'][good_shifts, i])
            elif good_shifts.size > 0:
                shifts[j][coords[i]] = stich_shift.update_shifts(shifts[j][coords[i]], shift_info[j]['shifts'][:, i])
        # add outlier variable to shift_info to keep track of those shifts which are changed.
        shift_info[j]['outlier_shifts'] = shift_info[j]['shifts'].copy()
        shift_info[j]['outlier_score'] = shift_info[j]['score'].copy()
        shift_info[j]['outlier_shifts'][good_shifts, :] = 0
        shift_info[j]['outlier_score'][good_shifts, :] = 0
        if (np.sum(good_shifts) < 2 and len(good_shifts) > 4) or (np.sum(good_shifts) == 0 and len(good_shifts) > 0):
            warnings.warn(f"{len(good_shifts) - np.sum(good_shifts)}/{len(good_shifts)}"
                          f" of shifts fell below score threshold")
        for i in np.where(good_shifts == False)[0]:
            t = shift_info[j]['pairs'][i, 0]
            t_neighb = shift_info[j]['pairs'][i, 1]
            # re-find shifts that fell below threshold by only looking at shifts near to others found
            # Don't allow any widening so shift found must be in this range.
            # score_thresh given is 0, so it is not re-computed.
            shift_info[j]['shifts'][i], shift_info[j]['score'][i] = stich_shift.compute_shift(
                find_spots_base.spot_yxz(local_yxz, t, r, c, spot_no),
                find_spots_base.spot_yxz(local_yxz, t_neighb, r, c, spot_no), 0, None, None,
                None, config['neighb_dist_thresh'], shifts[j]['y'], shifts[j]['x'], shifts[j]['z'],
                None, None, z_scale, config['nz_collapse'], config['shift_step'][2]
            )[:2]
            warnings.warn(f"\nShift from tile {t} to tile {t_neighb} changed from\n"
                          f"{shift_info[j]['outlier_shifts'][i]} to {shift_info[j]['shifts'][i]}.")

    # The following is a hack to order the tiles in the correct way if the extraction has not been done correctly.
    if config['flip_y']:
        shift_info['north']['shifts'][:, 0] = -shift_info['north']['shifts'][:, 0]
        shift_info['east']['shifts'][:, 0] = -shift_info['east']['shifts'][:, 0]
    if config['flip_x']:
        shift_info['north']['shifts'][:, 1] = -shift_info['north']['shifts'][:, 1]
        shift_info['east']['shifts'][:, 1] = -shift_info['east']['shifts'][:, 1]

    # get tile origins in global coordinates.
    # global coordinates are built about central tile so find this first
    tile_dist_to_centre = np.linalg.norm(nbp_basic.tilepos_yx[nbp_basic.use_tiles] -
                                         np.mean(nbp_basic.tilepos_yx, axis=0), axis=1)
    centre_tile = nbp_basic.use_tiles[tile_dist_to_centre.argmin()]

    # Currently this approach does not work when not all tiles used are connected, so check this first.
    min_hamming_dist = np.zeros(nbp_basic.n_tiles)
    for t in nbp_basic.use_tiles:
        # find the min distance between this tile and all other tiles used
        hamming_dist = np.sum(np.abs(nbp_basic.tilepos_yx[t] - nbp_basic.tilepos_yx[nbp_basic.use_tiles]), axis=1).astype(float)
        hamming_dist[hamming_dist == 0] = np.inf
        min_hamming_dist[t] = np.min(hamming_dist)
    min_hamming_dist = min_hamming_dist[nbp_basic.use_tiles]
    all_tiles_connected = np.all(min_hamming_dist == 1)
    no_tiles_connected = np.all(min_hamming_dist > 1)

    if all_tiles_connected:
        tile_origin = stitch_tile_origin.get_tile_origin(shift_info['north']['pairs'], shift_info['north']['shifts'], 
                                                         shift_info['east']['pairs'], shift_info['east']['shifts'], 
                                                         nbp_basic.n_tiles, centre_tile)
    elif no_tiles_connected:
        if nbp_basic.n_tiles > 1:
            warnings.warn("No tiles used are connected, so cannot find tile origins. "
                        "Setting all tile origins to non-overlapping values.")
        tile_origin = np.zeros((nbp_basic.n_tiles, 3))
        tile_origin[:, 2] = nbp_basic.nz / 2
        tile_origin[:, :2] = nbp_basic.tilepos_yx * (1 - config['expected_overlap']) * nbp_basic.tile_sz
        tile_origin = tile_origin - tile_origin[centre_tile]
        # set unused tiles to nan
        unused_tiles = np.setdiff1d(np.arange(nbp_basic.n_tiles), nbp_basic.use_tiles)
        tile_origin[unused_tiles, :] = np.nan

    else:
        # Raise error if some tiles are connected and some are not.
        raise ValueError("Some tiles used are connected and some are not, so cannot find tile origins. "
                         "Either all tiles or no tiles should be connected.")

    if nbp_basic.is_3d is False:
        tile_origin[:, 2] = 0  # set z coordinate to 0 for all tiles if 2d

    # add tile origin to debugging notebook so don't have whole page for one variable,
    # and need to add other rounds to it in registration stage anyway.
    nbp_debug.tile_origin = tile_origin
    # save all shift info to debugging page
    for j in directions:
        nbp_debug.__setattr__(j + '_' + 'final_shift_search', np.zeros((3, 3), dtype=int))
        nbp_debug.__getattribute__(j + '_' + 'final_shift_search')[:, 0] = \
            [np.min(shifts[j][key]) for key in shifts[j].keys()]
        nbp_debug.__getattribute__(j + '_' + 'final_shift_search')[:, 1] = [np.max(shifts[j][key]) for key in
                                                                            shifts[j].keys()]
        nbp_debug.__getattribute__(j + '_' + 'final_shift_search')[:, 2] = \
            nbp_debug.__getattribute__(j + '_' + 'start_shift_search')[:, 2]
        for var in shift_info[j].keys():
            nbp_debug.__setattr__(j + '_' + var, shift_info[j][var])
    return nbp_debug
