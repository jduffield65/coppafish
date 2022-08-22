import numpy as np
from ..setup import Notebook
import warnings


def check_transforms(nb: Notebook):
    """
    This checks that a decent number of affine transforms computed in the register stage of the pipeline
    are acceptable (`n_matches > n_matches_thresh`).

    If for any of the following, the fraction of transforms with
    `n_matches < n_matches_thresh` exceeds `config['register']['n_transforms_error_fraction']`
    an error will be raised.

    * Each channel across all rounds and tiles.
    * Each tile across all rounds and channels.
    * Each round across all tile and channels.

    Args:
        nb: *Notebook* containing `find_spots` page.

    """
    config = nb.get_config()['register']
    use_tiles = np.asarray(nb.basic_info.use_tiles)
    use_rounds = np.asarray(nb.basic_info.use_rounds)
    use_channels = np.asarray(nb.basic_info.use_channels)
    n_matches = nb.register_debug.n_matches[np.ix_(use_tiles, use_rounds, use_channels)]
    n_matches_thresh = nb.register_debug.n_matches_thresh[np.ix_(use_tiles, use_rounds, use_channels)]
    error_message = ""

    # Consider bad channels first as most likely to have consistently failed transform
    n_transforms = len(use_tiles) * len(use_rounds)
    n_transforms_error = int(np.clip(np.floor(n_transforms * config['n_transforms_error_fraction']), 1, np.inf))
    n_transforms_fail = np.zeros(len(use_channels), dtype=int)
    for c in range(len(use_channels)):
        failed = np.vstack(np.where(n_matches[:, :, c] < n_matches_thresh[:, :, c])).T
        n_transforms_fail[c] = failed.shape[0]
        if n_transforms_fail[c] > 0:
            failed_info = np.zeros((n_transforms_fail[c], 4), dtype=int)
            failed_info[:, 0] = use_tiles[failed[:, 0]]
            failed_info[:, 1] = use_rounds[failed[:, 1]]
            failed_info[:, 2] = n_matches[failed[:,0], failed[:,1], c]
            failed_info[:, 3] = n_matches_thresh[failed[:, 0], failed[:, 1], c]
            warnings.warn(f"\nChannel {use_channels[c]} - {n_transforms_fail[c]} tiles/rounds with n_matches < "
                          f"n_matches_thresh:\nInformation for failed transforms\n"
                          f"Tile, Round, n_matches, n_matches_thresh:\n{failed_info}")

    fail_inds = np.where(n_transforms_fail >= n_transforms_error)[0]
    if len(fail_inds) > 0:
        error_message = error_message + f"\nChannels that failed: {use_channels[fail_inds]}\n" \
                                        f"This is because out of {n_transforms} tiles/rounds, these channels had " \
                                        f"respectively:\n{n_transforms_fail[fail_inds]}\ntiles/rounds with " \
                                        f"n_matches < n_matches_thresh. " \
                                        f"These are all more than the error threshold of " \
                                        f"{n_transforms_error}.\nConsider removing these from use_channels."
        # don't consider failed channels for subsequent warnings/errors
        use_channels = np.setdiff1d(use_channels, use_channels[fail_inds])
        n_matches = nb.register_debug.n_matches[np.ix_(use_tiles, use_rounds, use_channels)]
        n_matches_thresh = nb.register_debug.n_matches_thresh[np.ix_(use_tiles, use_rounds, use_channels)]

    # Consider bad tiles next as second most likely to have consistently low spot counts in a tile
    n_transforms = len(use_channels) * len(use_rounds)
    n_transforms_error = int(np.clip(np.floor(n_transforms * config['n_transforms_error_fraction']), 1, np.inf))
    n_transforms_fail = np.zeros(len(use_tiles), dtype=int)
    for t in range(len(use_tiles)):
        failed = np.vstack(np.where(n_matches[t] < n_matches_thresh[t])).T
        n_transforms_fail[t] = failed.shape[0]
    fail_inds = np.where(n_transforms_fail >= n_transforms_error)[0]
    if len(fail_inds) > 0:
        error_message = error_message + f"\nTiles that failed: {use_tiles[fail_inds]}\n" \
                                        f"This is because out of {n_transforms} rounds/channels, these tiles had " \
                                        f"respectively:\n{n_transforms_fail[fail_inds]}\nrounds/channels with " \
                                        f"n_matches < n_matches_thresh. " \
                                        f"These are all more than the error threshold of " \
                                        f"{n_transforms_error}.\nConsider removing these from use_tiles."
        # don't consider failed channels for subsequent warnings/errors
        use_tiles = np.setdiff1d(use_tiles, use_tiles[fail_inds])
        n_matches = nb.register_debug.n_matches[np.ix_(use_tiles, use_rounds, use_channels)]
        n_matches_thresh = nb.register_debug.n_matches_thresh[np.ix_(use_tiles, use_rounds, use_channels)]

    # Consider bad rounds last as least likely to have consistently low spot counts in a round
    n_transforms = len(use_channels) * len(use_tiles)
    n_transforms_error = int(np.clip(np.floor(n_transforms * config['n_transforms_error_fraction']), 1, np.inf))
    n_transforms_fail = np.zeros(len(use_rounds), dtype=int)
    for r in range(len(use_rounds)):
        failed = np.vstack(np.where(n_matches[:, r] < n_matches_thresh[:, r])).T
        n_transforms_fail[r] = failed.shape[0]
    fail_inds = np.where(n_transforms_fail >= n_transforms_error)[0]
    if len(fail_inds) > 0:
        error_message = error_message + f"\nRounds that failed: {use_rounds[fail_inds]}\n" \
                                        f"This is because out of {n_transforms} tiles/channels, these rounds had " \
                                        f"respectively:\n{n_transforms_fail[fail_inds]}\ntiles/channels with " \
                                        f"n_matches < n_matches_thresh. " \
                                        f"These are all more than the error threshold of " \
                                        f"{n_transforms_error}.\nConsider removing these from use_rounds."

    if len(error_message) > 0:
        error_message = error_message + f"\nLook at the following diagnostics to decide if transforms " \
                                        f"are acceptable to continue:\n" \
                                        f"coppafish.plot.scale_box_plots\ncoppafish.plot.view_affine_shift_info\n" \
                                        f"coppafish.plot.view_icp"
        raise ValueError(error_message)
