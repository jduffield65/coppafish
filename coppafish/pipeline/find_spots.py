from tqdm import tqdm
import numpy as np
import itertools
import numpy.typing as npt
from typing import Optional

from .. import find_spots as fs
from .. import utils
from ..setup.notebook import NotebookPage
from ..utils import tiles_io


def find_spots(
    config: dict,
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_filter: NotebookPage,
    auto_thresh: np.ndarray,
    image_t: Optional[npt.NDArray[np.int32]] = None,
) -> NotebookPage:
    """
    This function turns each tiff file in the tile directory into a point cloud, saving the results
    as `spot_details` in the `find_spots` notebook page.

    See `'find_spots'` section of `notebook_comments.json` file
    for description of the variables in the page.

    Args:
        config (dict): Dictionary obtained from `'find_spots'` section of config file.
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_extract (NotebookPage): `extract` notebook page.
        nbp_filter (NotebookPage): `filter` notebook page.
        auto_thresh (`[n_tiles x n_rounds x n_channels] ndarray[float]`): `auto_thresh[t, r, c]` is the threshold for
            the tiff file corresponding to tile `t`, round `r`, channel `c` such that all local maxima with pixel
            values greater than this are considered spots.
        image_t (`(n_rounds x n_channels x nz x ny x nx) ndarray[uint16]`, optional): extracted and filtered image
            for a single tile. If given, find_spots runs on the single tile and returns its NotebookPage. Default: not
            given.

    Returns:
        `NotebookPage[find_spots]` - Page containing point cloud of all tiles, rounds and channels.
    """
    n_z = np.max([1, nbp_basic.is_3d * nbp_basic.nz])
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels

    if image_t is not None:
        assert (
            len(nbp_basic.use_tiles) == 1
        ), "Require use_tiles and n_tiles to be for a single tile if image_t is given"
        if nbp_basic.is_3d:
            assert image_t.ndim == 5, f"Expected image_t in 3D to have 5 dimensions, got {image_t.ndim}"
        else:
            assert image_t.ndim == 4, f"Expected image_t in 2D to have 4 dimensions, got {image_t.ndim}"

    # Phase 0: Initialisation
    nbp = NotebookPage("find_spots")
    nbp.software_version = utils.system.get_software_verison()
    nbp.revision_hash = utils.system.get_git_revision_hash()
    if nbp_basic.is_3d is False:
        # set z details to None if using 2d pipeline
        config["radius_z"] = None
        config["isolation_radius_z"] = None
        max_spots = config["max_spots_2d"]
    else:
        max_spots = config["max_spots_3d"]

    # record threshold for isolated spots in each tile of reference round/channel
    if config["isolation_thresh"] is None:
        nbp.isolation_thresh = (
            auto_thresh[:, nbp_basic.anchor_round, nbp_basic.anchor_channel]
            * config["auto_isolation_thresh_multiplier"]
        )
    else:
        nbp.isolation_thresh = (
            np.ones_like(auto_thresh[:, nbp_basic.anchor_round, nbp_basic.anchor_channel]) * config["isolation_thresh"]
        )

    # Phase 1: Load in previous results if they exist
    spot_info = fs.load_spot_info(
        nbp_file.spot_details_info,
        nbp_basic.n_tiles,
        nbp_basic.n_rounds,
        nbp_basic.n_extra_rounds,
        nbp_basic.n_channels,
    )
    # Define use_indices as a [n_tiles x n_rounds x n_channels] boolean array where use_indices[t, r, c] is True if
    # we want to use tile `t`, round `r`, channel `c` to find spots.
    use_indices = np.zeros(
        (nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.use_anchor + nbp_basic.use_preseq, nbp_basic.n_channels),
        dtype=bool,
    )
    for t, r, c in itertools.product(
        use_tiles, use_rounds + nbp_basic.use_preseq * [nbp_basic.pre_seq_round], use_channels
    ):
        use_indices[t, r, c] = True
    for t in use_tiles:
        use_indices[t, nbp_basic.anchor_round, nbp_basic.anchor_channel] = True

    uncompleted = np.logical_and(use_indices, np.logical_not(spot_info["completed"]))

    # Phase 2: Detect spots on uncompleted tiles, rounds and channels
    with tqdm(
        total=np.sum(uncompleted),
        postfix={"tile": t, "round": r, "channel": c},
        desc=f"Detecting spots on filtered {nbp_extract.file_type} images",
    ) as pbar:
        # Loop over uncompleted tiles, rounds and channels
        for t, r, c in np.argwhere(uncompleted):
            # Then need to shift the detect_spots and check_neighb_intensity thresh correspondingly.
            if image_t is None:
                image_trc = tiles_io.load_image(
                    nbp_file,
                    nbp_basic,
                    nbp_extract.file_type,
                    t,
                    r,
                    c,
                    apply_shift=False,
                    suffix="_raw" if r == nbp_basic.pre_seq_round else "",
                )
            else:
                # zyx -> yxz
                image_trc = image_t[r, c].transpose((1, 2, 0))
            local_yxz, spot_intensity = fs.detect_spots(
                image_trc,
                auto_thresh[t, r, c] + nbp_basic.tile_pixel_value_shift,
                config["radius_xy"],
                config["radius_z"],
                True,
            )
            no_negative_neighbour = fs.check_neighbour_intensity(
                image_trc, local_yxz, thresh=nbp_basic.tile_pixel_value_shift
            )
            local_yxz = local_yxz[no_negative_neighbour]
            spot_intensity = spot_intensity[no_negative_neighbour]
            # If r is a reference round, we also get info about whether the spots are isolated
            if r == nbp_basic.anchor_round:
                isolated_spots = fs.get_isolated(
                    image_trc.astype(np.int32) - nbp_basic.tile_pixel_value_shift,
                    local_yxz,
                    nbp.isolation_thresh[t],
                    config["isolation_radius_inner"],
                    config["isolation_radius_xy"],
                    config["isolation_radius_z"],
                )
                spot_info["isolated"] = np.append(spot_info["isolated"], isolated_spots)
            else:
                # if imaging round, only keep the highest intensity spots on each z plane
                local_yxz = fs.filter_intense_spots(local_yxz, spot_intensity, n_z, max_spots)

            # Save results to spot_info
            spot_info["spot_yxz"] = np.vstack((spot_info["spot_yxz"], local_yxz))
            spot_info["spot_no"][t, r, c] = local_yxz.shape[0]
            spot_info["completed"][t, r, c] = True
            assert spot_info["spot_yxz"].shape[0] == np.sum(
                spot_info["spot_no"]
            ), "spot_yxz and spot_no do not match. Tile {}, round {}, channel {}".format(t, r, c)
            np.savez(
                nbp_file.spot_details_info,
                spot_info["spot_yxz"],
                spot_info["spot_no"],
                spot_info["isolated"],
                spot_info["completed"],
            )
            pbar.update(1)

    # Phase 3: Save results to notebook page
    nbp.spot_yxz = spot_info["spot_yxz"]
    nbp.spot_no = spot_info["spot_no"]
    nbp.isolated_spots = spot_info["isolated"]

    return nbp
