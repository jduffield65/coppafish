import warnings
import numpy as np
from ..setup import Notebook


def check_n_spots(nb: Notebook):
    """
    This checks that a decent number of spots are detected on:

    * Each channel across all rounds and tiles.
    * Each tile across all rounds and channels.
    * Each round across all tile and channels.

    An error will be raised if any of these conditions are violated.

    `config['find_spots']['n_spots_warn_fraction']` and `config['find_spots']['n_spots_error_fraction']`
    are the parameters which determine if warnings/errors will be raised.

    Args:
        nb: *Notebook* containing `find_spots` page.
    """
    # TODO: show example of what error looks like in the docs
    config = nb.get_config()['find_spots']
    if nb.basic_info.is_3d:
        n_spots_warn = config['n_spots_warn_fraction'] * config['max_spots_3d'] * nb.basic_info.nz
    else:
        n_spots_warn = config['n_spots_warn_fraction'] * config['max_spots_2d']
    n_spots_warn = int(np.ceil(n_spots_warn))
    use_tiles = np.asarray(nb.basic_info.use_tiles)
    error_message = ""

    if len(nb.basic_info.use_rounds) > 0:
        use_rounds = np.asarray(nb.basic_info.use_rounds)  # don't consider anchor in this analysis
        use_channels = np.asarray(nb.basic_info.use_channels)
        spot_no = nb.find_spots.spot_no[np.ix_(use_tiles, use_rounds, use_channels)]

        # Consider bad channels first as most likely to have consistently low spot counts in a channel
        n_images = len(use_tiles) * len(use_rounds)
        n_images_error = int(np.floor(n_images * config['n_spots_error_fraction']))
        n_bad_images = np.zeros(len(use_channels), dtype=int)
        for c in range(len(use_channels)):
            bad_images = np.vstack(np.where(spot_no[:, :, c] < n_spots_warn)).T
            n_bad_images[c] = bad_images.shape[0]
            if n_bad_images[c] > 0:
                bad_images[:, 0] = use_tiles[bad_images[:, 0]]
                bad_images[:, 1] = use_rounds[bad_images[:, 1]]
                warnings.warn(f"\nChannel {use_channels[c]} - {n_bad_images[c]} tiles/rounds with n_spots < {n_spots_warn}:"
                              f"\n{bad_images}")

        fail_inds = np.where(n_bad_images >= n_images_error)[0]
        if len(fail_inds) > 0:
            error_message = error_message + f"\nChannels that failed: {use_channels[fail_inds]}\n" \
                                            f"This is because out of {n_images} tiles/rounds, these channels had " \
                                            f"respectively:\n{n_bad_images[fail_inds]}\ntiles/rounds with " \
                                            f"n_spots < {n_spots_warn}. These are all more than the error threshold of " \
                                            f"{n_images_error}.\nConsider removing these from use_channels."
            # don't consider failed channels for subsequent warnings/errors
            use_channels = np.setdiff1d(use_channels, use_channels[fail_inds])
            spot_no = nb.find_spots.spot_no[np.ix_(use_tiles, use_rounds, use_channels)]

        # Consider bad tiles next as second most likely to have consistently low spot counts in a tile
        n_images = len(use_channels) * len(use_rounds)
        n_images_error = int(np.floor(n_images * config['n_spots_error_fraction']))
        n_bad_images = np.zeros(len(use_tiles), dtype=int)
        for t in range(len(use_tiles)):
            bad_images = np.vstack(np.where(spot_no[t] < n_spots_warn)).T
            n_bad_images[t] = bad_images.shape[0]
        fail_inds = np.where(n_bad_images >= n_images_error)[0]
        if len(fail_inds) > 0:
            error_message = error_message + f"\nTiles that failed: {use_tiles[fail_inds]}\n" \
                                            f"This is because out of {n_images} rounds/channels, these tiles had " \
                                            f"respectively:\n{n_bad_images[fail_inds]}\nrounds/channels with " \
                                            f"n_spots < {n_spots_warn}. These are all more than the error threshold of " \
                                            f"{n_images_error}.\nConsider removing these from use_tiles."
            # don't consider failed channels for subsequent warnings/errors
            use_tiles = np.setdiff1d(use_tiles, use_tiles[fail_inds])
            spot_no = nb.find_spots.spot_no[np.ix_(use_tiles, use_rounds, use_channels)]

        # Consider bad rounds last as least likely to have consistently low spot counts in a round
        n_images = len(use_channels) * len(use_tiles)
        n_images_error = int(np.floor(n_images * config['n_spots_error_fraction']))
        n_bad_images = np.zeros(len(use_rounds), dtype=int)
        for r in range(len(use_rounds)):
            bad_images = np.vstack(np.where(spot_no[:, r] < n_spots_warn)).T
            n_bad_images[r] = bad_images.shape[0]
        fail_inds = np.where(n_bad_images >= n_images_error)[0]
        if len(fail_inds) > 0:
            error_message = error_message + f"\nRounds that failed: {use_rounds[fail_inds]}\n" \
                                            f"This is because out of {n_images} tiles/channels, these tiles had " \
                                            f"respectively:\n{n_bad_images[fail_inds]}\ntiles/channels with " \
                                            f"n_spots < {n_spots_warn}. These are all more than the error threshold " \
                                            f"of {n_images_error}.\nConsider removing these from use_rounds."

    # Consider anchor
    if nb.basic_info.use_anchor:
        spot_no = nb.find_spots.spot_no[use_tiles, nb.basic_info.anchor_round, nb.basic_info.anchor_channel]
        n_images = len(use_tiles)
        n_images_error = int(np.floor(n_images * config['n_spots_error_fraction']))
        bad_images = np.where(spot_no < n_spots_warn)[0]
        n_bad_images = len(bad_images)
        if n_bad_images > 0:
            bad_images = use_tiles[bad_images]
            warnings.warn(
                f"\nAnchor - {n_bad_images} tiles with n_spots < {n_spots_warn}:\n"
                f"{bad_images}")

        if n_bad_images >= n_images_error:
            error_message = error_message + f"\nAnchor - tiles {bad_images} all had n_spots < {n_spots_warn}. " \
                                            f"{n_bad_images}/{n_images} tiles failed which is more than the " \
                                            f"error threshold of {n_images_error}.\n" \
                                            f"Consider removing these tiles from use_tiles."

    if len(error_message) > 0:
        error_message = error_message + f"\nThe function coppafish.plot.view_find_spots may be useful for " \
                                        f"investigating why the above tiles/rounds/channels had so few spots detected."
        raise ValueError(error_message)
