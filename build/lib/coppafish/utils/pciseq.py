import pandas as pd
from ..call_spots.qual_check import quality_threshold
from ..setup import NotebookPage, Notebook
from .. import utils
from typing import Optional
import warnings
import os


def get_thresholds_page(nb: Notebook) -> NotebookPage:
    """
    Makes notebook page from thresholds section of config file.

    Args:
        nb: Notebook containing all experiment information.

    Returns:
        thresholds NotebookPage.
    """
    config = nb.get_config()['thresholds']
    if config['intensity'] is None:
        config['intensity'] = nb.call_spots.gene_efficiency_intensity_thresh
    nbp = NotebookPage('thresholds')
    nbp.intensity = config['intensity']
    nbp.score_ref = config['score_ref']
    nbp.score_omp = config['score_omp']
    nbp.score_omp_multiplier = config['score_omp_multiplier']
    return nbp


def export_to_pciseq(nb: Notebook):
    """
    This saves .csv files containing plotting information for pciseq-

    - y - y coordinate of each spot in stitched coordinate system.
    - x - x coordinate of each spot in stitched coordinate system.
    - z_stack - z coordinate of each spot in stitched coordinate system (in units of z-pixels).
    - Gene - Name of gene each spot was assigned to.

    Only spots which pass `quality_threshold` are saved.
    This depends on parameters given in `config['thresholds']`.

    One .csv file is saved for each method: *omp* and *ref_spots* if the notebook contains
    both pages.
    Also adds the *thresholds* page to the notebook and re-saves.
    This is so the *thresholds* section in the config file cannot be further changed.

    Args:
        nb: Notebook for the experiment containing at least the *ref_spots* page.

    """
    page_names = ['omp', 'ref_spots']
    method = ['omp', 'anchor']  # for calling qual_ok
    files_saved = 0
    for i in range(2):
        if not nb.has_page(page_names[i]):
            warnings.warn(f'No file saved for method {method[i]} as notebook does not have a {page_names[i]} page.')
            continue
        if os.path.isfile(nb.file_names.pciseq[i]):
            warnings.warn(f"File {nb.file_names.pciseq[i]} already exists")
            continue
        qual_ok = quality_threshold(nb, method[i])  # only keep spots which pass quality thresholding
        # get coordinates in stitched image
        global_spot_yxz = nb.__getattribute__(page_names[i]).local_yxz + \
                          nb.stitch.tile_origin[nb.__getattribute__(page_names[i]).tile]
        spot_gene = nb.call_spots.gene_names[nb.__getattribute__(page_names[i]).gene_no[qual_ok]]
        global_spot_yxz = global_spot_yxz[qual_ok]
        df_to_export = pd.DataFrame(data=global_spot_yxz, index=spot_gene, columns=['y', 'x', 'z_stack'])
        df_to_export['Gene'] = df_to_export.index
        df_to_export.to_csv(nb.file_names.pciseq[i], index=False)
        print(f'pciSeq file saved for method = {method[i]}: ' + nb.file_names.pciseq[i])
        files_saved += 1

    if files_saved > 0:
        # If saved any files, add thresholds page to notebook so cannot make any further changes to
        # config - will trigger save
        if not nb.has_page('thresholds'):
            nbp_thresholds = get_thresholds_page(nb)
            nb += nbp_thresholds
        else:
            warnings.warn('thresholds', utils.warnings.NotebookPageWarning)
    else:
        warnings.warn(f"No files saved")
