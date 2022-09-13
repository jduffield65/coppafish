from coppafish.setup.notebook import Notebook, NotebookPage
import os
import numpy as np
import warnings


def spot_details_conversion(nb: Notebook):
    """
    Function to reformat old find_spots pages into new format.
    :param nb: Notebook with old format of find_spots page
    :return nb_updated: Notebook with updated format of find_spots page
    """

    # First create output directory to save spot_details_info to
    config = nb.get_config()
    spot_details_info_dir = os.path.join(config['file_names']['output_dir'], 'spot_details_info' + '.npz')
    # First convert the old spot_details into the new spot_details
    # Need this to be ordered by t,r,c but its currently ordered r,t,c
    spot_details_old = nb.find_spots.spot_details
    # First order by x
    spot_details_old = spot_details_old[spot_details_old[:, 5].argsort()]
    # Now order by y, so if we have 2 rows with same x, tie will be broken by putting the row with the lower
    # y first. Must specify mergesort to maintain previous order where possible
    spot_details_old = spot_details_old[spot_details_old[:, 4].argsort(kind='mergesort')]
    # Now order by channel
    spot_details_old = spot_details_old[spot_details_old[:, 2].argsort(kind='mergesort')]
    # Now order by round
    spot_details_old = spot_details_old[spot_details_old[:, 1].argsort(kind='mergesort')]
    # Now order by tile
    spot_details_old = spot_details_old[spot_details_old[:, 0].argsort(kind='mergesort')]
    # Finally, just crop the final 3 columns for the new spot_details
    spot_details_new = spot_details_old[:, 4:]

    # spot_no array is unchanged
    spot_no = nb.find_spots.spot_no

    # Now we need to create and populate the isolated spots array. This has length num_ref_spots
    ref_round = nb.basic_info.ref_round
    ref_channel = nb.basic_info.ref_channel
    # Now find the indices (ie the row numbers in the spot_details array) for reference spots
    anchor_indices = [i for i in range(spot_details_old.shape[0]) if spot_details_old[i, 1] == ref_round]
    num_ref_spots = np.sum(spot_no[:, ref_round, ref_channel])
    # Chuck warning if len(anchor_indices) != num_ref_spots
    if len(anchor_indices) != num_ref_spots:
        warnings.warn(f'We should have found {num_ref_spots} anchor indices, but we have '
                      f''f'{len(anchor_indices)} tiles. This may cause an index error.')
    # Read out isolated_spot info for these rows
    isolated_spots = np.array(spot_details_old[anchor_indices, 3], dtype=bool)

    # Now save the spot_details_info
    np.savez(spot_details_info_dir, spot_details_new, spot_no, isolated_spots)

    # Delete old find_spots page
    del nb.find_spots

    # Create new find_spots page
    nbp = NotebookPage("find_spots")

    # Add all updated variables
    nbp.spot_details = spot_details_new
    nbp.spot_no = spot_no
    nbp.isolated_spots = isolated_spots

    # Add the new page
    nb += nbp

    return nb
