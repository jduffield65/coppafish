from .. import utils, setup
from .. import find_spots as fs
from tqdm import tqdm
import numpy as np


def find_spots(config, nbp_file, nbp_basic, auto_thresh):
    nbp = setup.NotebookPage("find_spots")
    if nbp_basic['3d'] is False:
        # set z details to None if using 2d pipeline
        config['radius_z'] = None
        config['isolation_radius_z'] = None
    nbp_params = setup.NotebookPage("find_spots_params", config)  # params page inherits info from config
    if nbp_basic['3d'] is False:
        # add None params to nbp so can run detect spots using same line for 2d and 3d
        nbp_params['radius_z'] = None
        nbp_params['isolation_radius_z'] = None
    # have to save spot_yxz and spot_isolated as table to stop pickle issues associated with numpy object arrays.
    # columns of spot_details are: tile, channel, round, isolated, y, x, z
    spot_details = np.empty((0, 7), dtype=int)
    nbp['spot_no'] = np.zeros((nbp_basic['n_tiles'], nbp_basic['n_rounds']+nbp_basic['n_extra_rounds'],
                               nbp_basic['n_channels']), dtype=int)
    use_rounds = nbp_basic['use_rounds']
    n_images = len(use_rounds) * len(nbp_basic['use_tiles']) * len(nbp_basic['use_channels'])
    if nbp_basic['use_anchor']:
        use_rounds = use_rounds + [nbp_basic['anchor_round']]
        n_images = n_images + len(nbp_basic['use_tiles'])
    with tqdm(total=n_images) as pbar:
        for r in use_rounds:
            if r == nbp_basic['anchor_round']:
                use_channels = [nbp_basic['anchor_channel']]
            else:
                use_channels = nbp_basic['use_channels']  # TODO: this will fail if use_channels is an integer not array
            for t in nbp_basic['use_tiles']:
                for c in use_channels:
                    pbar.set_postfix({'round': r, 'tile': t, 'channel': c})
                    image = utils.tiff.load_tile(nbp_file, nbp_basic, t, r, c)
                    spot_yxz, spot_intensity = fs.detect_spots(image, auto_thresh[t, r, c],
                                                               nbp_params['radius_xy'], nbp_params['radius_z'])
                    no_negative_neighbour = fs.check_neighbour_intensity(image, spot_yxz, thresh=0)
                    spot_yxz = spot_yxz[no_negative_neighbour]
                    spot_intensity = spot_intensity[no_negative_neighbour]
                    if r == nbp_basic['ref_round']:
                        # if reference round, keep all spots, and record if isolated
                        if nbp_params.has_item('isolation_thresh'):
                            isolation_thresh = nbp_params['isolation_thresh']
                        else:
                            isolation_thresh = nbp_params['auto_isolation_thresh_multiplier'] * auto_thresh[t, r, c]
                        spot_isolated = fs.get_isolated(image, spot_yxz, isolation_thresh,
                                                        nbp_params['isolation_radius_inner'],
                                                        nbp_params['isolation_radius_xy'],
                                                        nbp_params['isolation_radius_z'])

                    else:
                        # if imaging round, only keep highest intensity spots as only used for registration
                        descend_intensity_arg = np.argsort(spot_intensity)[::-1]
                        spot_yxz = spot_yxz[descend_intensity_arg[:nbp_params['max_spots']]]
                        # don't care if these spots isolated so say they are not
                        spot_isolated = np.zeros(spot_yxz.shape[0], dtype=bool)
                    spot_details_trc = np.zeros((spot_yxz.shape[0], spot_details.shape[1]), dtype=int)
                    spot_details_trc[:, :3] = [t, r, c]
                    spot_details_trc[:, 3] = spot_isolated
                    spot_details_trc[:, 4:4+spot_yxz.shape[1]] = spot_yxz  # if 2d pipeline, z coordinate set to 0.
                    spot_details = np.append(spot_details, spot_details_trc, axis=0)
                    nbp['spot_no'][t, r, c] = spot_yxz.shape[0]
                    pbar.update(1)
    nbp['spot_details'] = spot_details
    return nbp, nbp_params
