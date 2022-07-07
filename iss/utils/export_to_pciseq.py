import pandas as pd
from ..pipeline.run import initialize_nb
from ..call_spots.base import quality_threshold


def export(config_file_path: str):

    # load notebook
    nb = initialize_nb(config_file_path)

    # Select spot
    qual_ok = quality_threshold(nb.omp)

    global_spot_yxz = nb.omp.local_yxz + nb.stitch.tile_origin[nb.omp.tile]
    global_spot_yxz = global_spot_yxz[qual_ok]

    spot_gene = nb.call_spots.gene_names[nb.omp.gene_no[qual_ok]]

    df_to_export = pd.DataFrame(data=global_spot_yxz, index=spot_gene, columns=['y', 'x', 'z_stack'])
    df_to_export['Gene'] = df_to_export.index

    output_path = nb.file_names.output_dir + '/' + config_file_path.split('/')[-1][:-4] + '.csv'
    df_to_export.to_csv(output_path, index=False)
    print('File saved: ' + output_path)


