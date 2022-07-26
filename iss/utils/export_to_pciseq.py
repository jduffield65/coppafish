import pandas as pd
from ..pipeline.run import initialize_nb
from ..call_spots.qual_check import quality_threshold
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


def export(config_file_path: str, method: str = 'omp'):
    """
    This saves a .csv file containing plotting information for pciseq.
    Also adds the thresholds page to the notebook and re-saves.
    This is so the thresholds cannot be further changed.

    Args:
        config_file_path: Path to config .ini file used to make notebook.
        method: 'omp' or 'anchor, which gene calling method to save

    """
    if method.lower() != 'omp' and method.lower() != 'ref' and method.lower() != 'anchor':
        raise ValueError(f"method must be 'omp' or 'anchor but {method} given.")
    # load notebook
    nb = initialize_nb(config_file_path)

    # Select spot
    qual_ok = quality_threshold(nb, method)
    if method.lower() == 'omp':
        global_spot_yxz = nb.omp.local_yxz + nb.stitch.tile_origin[nb.omp.tile]
        spot_gene = nb.call_spots.gene_names[nb.omp.gene_no[qual_ok]]
    else:
        global_spot_yxz = nb.ref_spots.local_yxz + nb.stitch.tile_origin[nb.ref_spots.tile]
        spot_gene = nb.call_spots.gene_names[nb.ref_spots.gene_no[qual_ok]]
    global_spot_yxz = global_spot_yxz[qual_ok]

    df_to_export = pd.DataFrame(data=global_spot_yxz, index=spot_gene, columns=['y', 'x', 'z_stack'])
    df_to_export['Gene'] = df_to_export.index

    output_path = nb.file_names.output_dir + '/' + config_file_path.split('/')[-1][:-4] + '_' + method + '.csv'
    # Make output_path have method in title
    df_to_export.to_csv(output_path, index=False)
    print('File saved: ' + output_path)

    # Add thresholds page to notebook so cannot make any further changes to config - will trigger save
    nbp_thresholds = get_thresholds_page(nb)
    nb += nbp_thresholds
