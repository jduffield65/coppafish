import os
import time
import pickle
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

from ..setup.notebook import NotebookPage
from .. import utils
from ..utils import tiles_io


# TODO: Add parameter return_image: bool to return the image for every round & channel when running on a single tile
def run_extract(
    config: dict,
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_scale: NotebookPage,
) -> Tuple[NotebookPage, NotebookPage, Optional[np.ndarray]]:
    """
    This reads in images from the raw `nd2` files, filters them and then saves them as `config[extract][file_type]`
    files in the tile directory. Also gets `auto_thresh` for use in turning images to point clouds and `hist_values`,
    `hist_counts` required for normalisation between channels.

    Args:
        config (dict): dictionary obtained from 'extract' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.
        nbp_scale (NotebookPage): 'scale' notebook page.

    Returns:
        - `NotebookPage[extract]`: page containing `auto_thresh` for use in turning images to point clouds and
            `hist_values`, `hist_counts` required for normalisation between channels.
        - `NotebookPage[extract_debug]`: page containing variables which are not needed later in the pipeline but may
            be useful for debugging purposes.
        - (`(n_rounds x n_channels x nz x ny x nx) ndarray[uint16]` or None): If running on a single tile, returns all
            extracted images, otherwise returns None.

    Notes:
        - See `'extract'` and `'extract_debug'` sections of `notebook_comments.json` file for description of the
            variables in each page.
    """
    # TODO: Refactor this function to make it more readable
    # initialise notebook pages
    if not nbp_basic.is_3d:
        # config["deconvolve"] = False  # only deconvolve if 3d pipeline
        raise NotImplementedError(f"coppafish 2d is not in a stable state, please contact a dev to add this. Sorry! ;(")

    start_time = time.time()
    nbp = NotebookPage("extract")
    nbp_debug = NotebookPage("extract_debug")
    nbp.file_type = config["file_type"]
    nbp.continuous_dapi = config["continuous_dapi"]

    # get rounds to iterate over
    use_channels_anchor = [c for c in [nbp_basic.dapi_channel, nbp_basic.anchor_channel] if c is not None]
    use_channels_anchor.sort()
    if nbp_basic.use_anchor:
        # always have anchor as first round after imaging rounds
        round_files = nbp_file.round + [nbp_file.anchor]
        use_rounds = np.arange(len(round_files))
        n_images = (
            (len(use_rounds) - 1) * len(nbp_basic.use_tiles) * len(nbp_basic.use_channels)
            + len(nbp_basic.use_tiles) * len(use_channels_anchor)
            + len(nbp_basic.use_tiles) * len(nbp_basic.use_rounds) * config["continuous_dapi"]
        )
    else:
        round_files = nbp_file.round
        use_rounds = nbp_basic.use_rounds
        n_images = len(use_rounds) * len(nbp_basic.use_tiles) * len(nbp_basic.use_channels)

    # If we have a pre-sequencing round, add this to round_files at the end
    if nbp_basic.use_preseq:
        round_files = round_files + [nbp_file.pre_seq]
        use_rounds = np.arange(len(round_files))
        pre_seq_round = len(round_files) - 1
        n_images += len(nbp_basic.use_tiles) * len(nbp_basic.use_channels)
    else:
        pre_seq_round = None

    return_image_t = len(nbp_basic.use_tiles) == 1
    if return_image_t:
        image_t = np.zeros(
            (
                max(use_rounds) + 1,
                max(nbp_basic.use_channels + [nbp_basic.dapi_channel] * config["continuous_dapi"]) + 1,
                max(nbp_basic.use_z) + 1,
                nbp_basic.tile_sz,
                nbp_basic.tile_sz,
            ),
            dtype=np.uint16,
        )
    else:
        image_t = None

    with tqdm(
        total=n_images, desc=f"Loading raw {nbp_file.raw_extension} tiles and saving as {config['file_type']}"
    ) as pbar:
        for r in use_rounds:
            round_dask_array, metadata = utils.raw.load_dask(nbp_file, nbp_basic, r=r)

            if metadata is not None:
                # Dump all metadata to pickle file
                with open(os.path.join(nbp_file.tile_unfiltered_dir, f"nd2_metadata_r{r}.pkl"), "w") as file:
                    pickle.dump(metadata, file)

            if r == nbp_basic.anchor_round:
                use_channels = use_channels_anchor
            else:
                use_channels = nbp_basic.use_channels + [nbp_basic.dapi_channel] * config["continuous_dapi"]

            # convolve_2d each image
            for t in nbp_basic.use_tiles:
                if not nbp_basic.is_3d:
                    # for 2d all channels in same file
                    file_exists = tiles_io.image_exists(nbp_file.tile_unfiltered[t][r], config["file_type"])
                    if file_exists:
                        # mmap load in image for all channels if tiff exists
                        im_all_channels_2d = np.load(nbp_file.tile_unfiltered[t][r], mmap_mode="r")
                    else:
                        # Only save 2d data when all channels collected
                        # For channels not used, keep all pixels 0.
                        im_all_channels_2d = np.zeros(
                            (nbp_basic.n_channels, nbp_basic.tile_sz, nbp_basic.tile_sz), dtype=np.int32
                        )
                        # FIXME: Save 2d images
                for c in use_channels:
                    if nbp_basic.is_3d:
                        if r != pre_seq_round:
                            file_path = nbp_file.tile_unfiltered[t][r][c]
                            file_exists = tiles_io.image_exists(file_path, config["file_type"])
                        else:
                            file_path = nbp_file.tile_unfiltered[t][r][c]
                            file_path = file_path[: file_path.index(config["file_type"])] + "_raw" + config["file_type"]
                            file_exists = tiles_io.image_exists(file_path, config["file_type"])
                        pbar.set_postfix({"round": r, "tile": t, "channel": c, "exists": str(file_exists)})

                        # TODO: Save the raw uint16 dask array file
                        im = utils.raw.load_image(nbp_file, nbp_basic, t, c, round_dask_array, r, nbp_basic.use_z)
                        im = im.astype(np.uint16, casting="safe")
                        # yxz -> zyx
                        im = im.transpose((2, 0, 1))
                        tiles_io._save_image(im, file_path, config["file_type"])
                        if return_image_t:
                            image_t[r, c] = im
                        del im
                        pbar.update(1)
    end_time = time.time()
    nbp_debug.time_taken = end_time - start_time
    return nbp, nbp_debug, image_t
