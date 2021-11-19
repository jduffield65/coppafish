from iss.setup.notebook import NotebookPage
import iss.find_spots as fs
import iss.utils
from tqdm import tqdm
import numpy as np


def find_spots(config, nbp_file, nbp_basic, auto_thresh):
    nbp = NotebookPage("find_spots")
    if nbp_basic['3d'] is False:
        # set z details to None if using 2d pipeline
        config['radius_z'] = None
        config['isolation_radius_z'] = None
    nbp_params = NotebookPage("find_spots_params", config)  # params page inherits info from config
    if nbp_basic['3d'] is False:
        # add None params to nbp so can run detect spots using same line for 2d and 3d
        nbp_params['radius_z'] = None
        nbp_params['isolation_radius_z'] = None
    spot_yx = np.zeros((nbp_basic['n_tiles'], nbp_basic['n_channels'],
                        nbp_basic['n_rounds']+nbp_basic['n_extra_rounds']), dtype=object)
    nbp['spot_no'] = np.zeros_like(spot_yx, dtype=int)
    # only find isolated in ref round hence no round axis
    spot_isolated = np.zeros((nbp_basic['n_tiles'], nbp_basic['n_channels']), dtype=object)
    use_rounds = nbp_basic['use_rounds']
    n_images = len(use_rounds) * len(nbp_basic['use_tiles']) * len(nbp_basic['use_channels'])
    if nbp_basic['use_anchor']:
        use_rounds = use_rounds + [nbp_basic['anchor_round']]
        n_images = n_images + len(nbp_basic['use_tiles'])
    with tqdm(total=n_images) as pbar:
        for r in use_rounds:
            if r == nbp_basic['anchor_round']:
                use_channels = nbp_basic['anchor_channel']
            else:
                use_channels = nbp_basic['use_channels']
            for t in nbp_basic['use_tiles']:
                for c in use_channels:
                    pbar.set_postfix({'round': r, 'tile': t, 'channel': c})
                    image = iss.utils.tiff.load_tile(nbp_file, nbp_basic, t, c, r)
                    spot_yx[t, c, r], spot_intensity = fs.detect_spots(image, auto_thresh[t, c, r],
                                                                       nbp_params['radius_xy'], nbp_params['radius_z'])
                    no_negative_neighbour = fs.check_neighbour_intensity(image, spot_yx[t, c, r], thresh=0)
                    spot_yx[t, c, r] = spot_yx[t, c, r][no_negative_neighbour]
                    spot_intensity = spot_intensity[no_negative_neighbour]
                    if r == nbp_basic['ref_round']:
                        # if reference round, keep all spots, and record if isolated
                        if nbp_params.has_item('isolation_thresh'):
                            isolation_thresh = nbp_params['isolation_thresh']
                        else:
                            isolation_thresh = nbp_params['auto_isolation_thresh_multiplier'] * auto_thresh[t, c, r]
                        spot_isolated[t, c] = fs.get_isolated(image, spot_yx[t, c, r], isolation_thresh,
                                                              nbp_params['isolation_radius_inner'],
                                                              nbp_params['isolation_radius_xy'],
                                                              nbp_params['isolation_radius_z'])

                    else:
                        # if imaging round, only keep highest intensity spots as only used for registration
                        descend_intensity_arg = np.argsort(spot_intensity)[::-1]
                        spot_yx[t, c, r] = spot_yx[t, c, r][descend_intensity_arg[:nbp_params['max_spots']]]
                    spot_yx[t, c, r] = spot_yx[t, c, r]
                    nbp['spot_no'][t, c, r] = spot_yx[t, c, r].shape[0]
                    pbar.update(1)
    nbp['spot_yx'] = spot_yx  # TODO: how to save and load numpy object array without pickle
    nbp['spot_isolated'] = spot_isolated
    return nbp, nbp_params
