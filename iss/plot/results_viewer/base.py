import os
import pandas as pd
import numpy as np
from ...call_spots.base import quality_threshold
from .legend import iss_legend
import napari


def iss_plot(nb, method):
    legend_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'legend')
    gene_color = pd.read_csv(os.path.join(legend_folder, 'gene_color.csv'))
    cell_color = pd.read_csv(os.path.join(legend_folder, 'cell_color.csv'))
    if method == 'omp':
        nbp = nb.omp
    else:
        nbp = nb.ref_spots
    qual_ok = quality_threshold(nbp)
    spot_zyx = (nbp.local_yxz + nb.stitch.tile_origin[nbp.tile])[:, [2, 0, 1]]
    spot_zyx = spot_zyx[qual_ok]
    if not nb.basic_info.is_3d:
        spot_zyx = spot_zyx[:, 1:]
    spot_gene = nb.call_spots.gene_names[nbp.gene_no[qual_ok]]

    im_dims = np.rint(spot_zyx.max(axis=0)).astype(int)
    background_image = np.zeros(im_dims, dtype=np.int8)
    # viewer = napari.view_image(background_image)
    viewer = napari.Viewer()
    iss_legend.add_legend(viewer, genes=gene_color, cells=cell_color, celltype=False, gene=True)

    # Add all points as white dots
    point_size = 10  # with size=4, spots are too small to see
    viewer.add_points(spot_zyx, name='GeneSpots', face_color='w', size=point_size)

    gene_color_dict = dict()
    for g in gene_color.index:
        gene_color_dict[gene_color.loc[g, 'GeneNames']] = (
        gene_color.loc[g, 'ColorR'], gene_color.loc[g, 'ColorG'], gene_color.loc[g, 'ColorB'])

    # Add gene spots with ISS color code
    for s in np.unique(gene_color['Symbols']):
        # TODO: set transparency based on spot score
        spots_to_plot = np.isin(spot_gene, gene_color[gene_color['Symbols'] == s]['GeneNames'])
        if spots_to_plot.any():
            coords_to_plot = spot_zyx[spots_to_plot]
            spotcolor_to_plot = [gene_color_dict[i] for i in spot_gene[spots_to_plot]]
            symb_to_plot = np.unique(gene_color[gene_color['Symbols'] == s]['napari_symbol'])[0]
            viewer.add_points(coords_to_plot, face_color=spotcolor_to_plot, symbol=symb_to_plot,
                              name=f'GeneSpots_Symbol_{s}', size=point_size)
    viewer.layers.pop(0)  # delete all gene white spot layer
    napari.run()
