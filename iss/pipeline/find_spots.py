from .. import utils
from .. import find_spots as fs
from tqdm import tqdm
import numpy as np
from ..setup.notebook import NotebookPage


def find_spots(config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage, auto_thresh: np.ndarray) -> NotebookPage:
    """
    This function turns each tiff file in the tile directory into a point cloud, saving the results
    as `spot_details` in the `find_spots` notebook page.

    See `'find_spots'` section of `notebook_comments.json` file
    for description of the variables in the page.

    Args:
        config: Dictionary obtained from `'find_spots'` section of config file.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        auto_thresh: `float [n_tiles x n_rounds x n_channels]`.
            `auto_thresh[t, r, c]` is the threshold for the tiff file corresponding to tile `t`, round `r`, channel `c`
            such that all local maxima with pixel values greater than this are considered spots.

    Returns:
        `NotebookPage[find_spots]` - Page containing point cloud of all tiles, rounds and channels.
    """
    nbp = NotebookPage("find_spots")
    if nbp_basic.is_3d is False:
        # set z details to None if using 2d pipeline
        config['radius_z'] = None
        config['isolation_radius_z'] = None
        max_spots = config['max_spots_2d']
    else:
        max_spots = config['max_spots_3d']

    # record threshold for isolated spots in each tile of reference round/channel
    if config['isolation_thresh'] is None:
        nbp.isolation_thresh = auto_thresh[:, nbp_basic.ref_round, nbp_basic.anchor_channel] * \
                                  config['auto_isolation_thresh_multiplier']
    else:
        nbp.isolation_thresh = np.ones_like(auto_thresh[:, nbp_basic.ref_round, nbp_basic.anchor_channel]) * \
                                  config['isolation_thresh']

    # have to save spot_yxz and spot_isolated as table to stop pickle issues associated with numpy object arrays.
    # columns of spot_details are: tile, channel, round, isolated, y, x, z
    # max value is y or x coordinate of around 2048 hence can use int16.
    spot_details = np.empty((0, 7), dtype=np.int16)
    nbp.spot_no = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds+nbp_basic.n_extra_rounds,
                               nbp_basic.n_channels), dtype=np.int32)
    use_rounds = nbp_basic.use_rounds
    n_images = len(use_rounds) * len(nbp_basic.use_tiles) * len(nbp_basic.use_channels)
    if nbp_basic.use_anchor:
        use_rounds = use_rounds + [nbp_basic.anchor_round]
        n_images = n_images + len(nbp_basic.use_tiles)
    n_z = np.max([1, nbp_basic.is_3d * nbp_basic.nz])
    with tqdm(total=n_images) as pbar:
        pbar.set_description(f"Detecting spots on filtered images saved as npy")
        for r in use_rounds:
            if r == nbp_basic.anchor_round:
                use_channels = [nbp_basic.anchor_channel]
            else:
                use_channels = nbp_basic.use_channels
            for t in nbp_basic.use_tiles:
                for c in use_channels:
                    pbar.set_postfix({'round': r, 'tile': t, 'channel': c})
                    # Find local maxima on shifted uint16 images to save time avoiding conversion to int32.
                    # Then need to shift the detect_spots and check_neighb_intensity thresh correspondingly.
                    image = utils.npy.load_tile(nbp_file, nbp_basic, t, r, c, apply_shift=False)
                    spot_yxz, spot_intensity = fs.detect_spots(image,
                                                               auto_thresh[t, r, c] + nbp_basic.tile_pixel_value_shift,
                                                               config['radius_xy'], config['radius_z'], True)
                    no_negative_neighbour = fs.check_neighbour_intensity(image, spot_yxz,
                                                                         thresh=nbp_basic.tile_pixel_value_shift)
                    spot_yxz = spot_yxz[no_negative_neighbour]
                    spot_intensity = spot_intensity[no_negative_neighbour]
                    if r == nbp_basic.ref_round:
                        spot_isolated = fs.get_isolated(image.astype(np.int32) - nbp_basic.tile_pixel_value_shift,
                                                        spot_yxz, nbp.isolation_thresh[t],
                                                        config['isolation_radius_inner'],
                                                        config['isolation_radius_xy'],
                                                        config['isolation_radius_z'])

                    else:
                        # if imaging round, only keep the highest intensity spots on each z plane
                        # as only used for registration
                        keep = np.ones(spot_yxz.shape[0], dtype=bool)
                        for z in range(n_z):
                            if nbp_basic.is_3d:
                                in_z = spot_yxz[:, 2] == z
                            else:
                                in_z = np.ones(spot_yxz.shape[0], dtype=bool)
                            if np.sum(in_z) > max_spots:
                                intensity_thresh = np.sort(spot_intensity[in_z])[-max_spots]
                                keep[np.logical_and(in_z, spot_intensity < intensity_thresh)] = False
                        spot_yxz = spot_yxz[keep]
                        # don't care if these spots isolated so say they are not
                        spot_isolated = np.zeros(spot_yxz.shape[0], dtype=bool)
                    spot_details_trc = np.zeros((spot_yxz.shape[0], spot_details.shape[1]), dtype=np.int16)
                    spot_details_trc[:, :3] = [t, r, c]
                    spot_details_trc[:, 3] = spot_isolated
                    spot_details_trc[:, 4:4+spot_yxz.shape[1]] = spot_yxz  # if 2d pipeline, z coordinate set to 0.
                    spot_details = np.append(spot_details, spot_details_trc, axis=0)
                    nbp.spot_no[t, r, c] = spot_yxz.shape[0]
                    pbar.update(1)
    nbp.spot_details = spot_details
    return nbp
