import pandas as pd
import numpy as np
import os

from ..call_spots import qual_check
from ..setup import NotebookPage, Notebook


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


def export_to_pciseq(nb: Notebook, method='omp', intensity_thresh: float = 0, score_thresh: float = 0):
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

    Args:
        nb: Notebook for the experiment containing at least the *ref_spots* page.
        method: `'ref'` or `'omp'` or `'anchor'` or 'prob'. Default: `'omp'`.
        intensity_thresh: Intensity threshold for spots included.
        score_thresh: Score threshold for spots included.

    """
    if method.lower() != 'omp' and method.lower() != 'ref' and method.lower() != 'anchor' and method.lower() != 'prob':
        raise ValueError(f"method must be 'omp', 'anchor' or 'prob' but {method} given.")
    page_name = 'omp' if method.lower() == 'omp' else 'ref_spots'
    index = 0
    if method.lower() == 'ref' or method.lower() == 'anchor':
        index = 1
    elif method.lower() == 'prob':
        index = 2
    if not nb.has_page(page_name):
        raise ValueError(f"Notebook does not contain {page_name} page.")
    if os.path.isfile(nb.file_names.pciseq[index]):
        raise FileExistsError(f"File already exists: {nb.file_names.pciseq[index]}")
    qual_ok = qual_check.quality_threshold(nb, method, intensity_thresh, score_thresh)

    # get coordinates in stitched image
    global_spot_yxz = nb.__getattribute__(page_name).local_yxz + \
                      nb.stitch.tile_origin[nb.__getattribute__(page_name).tile]
    if method.lower() == 'omp':
        spot_gene = nb.call_spots.gene_names[nb.omp.gene_no]
    elif method.lower() == 'ref' or method.lower() == 'anchor':
        spot_gene = nb.call_spots.gene_names[nb.ref_spots.gene_no]
    elif method.lower() == 'prob':
        spot_gene = nb.call_spots.gene_names[np.argmax(nb.ref_spots.gene_probs, axis=1)]
    spot_gene = spot_gene[qual_ok]
    global_spot_yxz = global_spot_yxz[qual_ok]
    df_to_export = pd.DataFrame(data=global_spot_yxz, index=spot_gene, columns=['y', 'x', 'z_stack'])
    df_to_export['Gene'] = df_to_export.index
    df_to_export.to_csv(nb.file_names.pciseq[index], index=False)
    print(f'pciSeq file saved for method = {method}: ' + nb.file_names.pciseq[index])

