import os
import pandas as pd
import numpy as np
from ...call_spots.base import quality_threshold
from .legend import iss_legend
from ..call_spots import view_codes, view_bleed_matrix, view_bled_codes, view_spot
import napari
from napari.qt import thread_worker
import time


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
    if not nb.basic_info.is_3d:
        spot_zyx = spot_zyx[:, 1:]
    spot_gene = nb.call_spots.gene_names[nbp.gene_no]

    im_dims = np.rint(spot_zyx.max(axis=0)).astype(int)
    background_image = np.zeros(im_dims, dtype=np.int8)
    # viewer = napari.view_image(background_image)
    viewer = napari.Viewer()
    iss_legend.add_legend(viewer, genes=gene_color, cells=cell_color, celltype=False, gene=True)

    # Add all points as white dots
    point_size = 10  # with size=4, spots are too small to see
    viewer.add_points(spot_zyx, name='GeneSpots', face_color='w', size=point_size+2, opacity=0, shown=qual_ok)
    all_gene_layer_ind = 0

    @viewer.bind_key('c')
    def call_to_view_codes(viewer):
        # on key press
        # TODO: make function for different letters to view_code / view_spot / view_omp of selected spot
        if len(viewer.layers[all_gene_layer_ind].selected_data) == 1:
            view_codes(nb, list(viewer.layers[all_gene_layer_ind].selected_data)[0])
        else:
            viewer.status = 'No spot selected :('

    @viewer.bind_key('s')
    def call_to_view_spot(viewer):
        if len(viewer.layers[all_gene_layer_ind].selected_data) == 1:
            view_spot(nb, list(viewer.layers[all_gene_layer_ind].selected_data)[0])
        else:
            viewer.status = 'No spot selected :('

    @viewer.bind_key('b')
    def call_to_view_bm(viewer):
        view_bleed_matrix(nb)

    @viewer.bind_key('g')
    def call_to_view_bm(viewer):
        view_bled_codes(nb)

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
                              name=f'GeneSpots_Symbol_{s}', size=point_size, shown=qual_ok[spots_to_plot])
    # viewer.layers.pop(0)  # delete all gene white spot layer
    viewer_status_on_select(viewer.layers[0], nb, viewer)  # so indicates when a spot is selected
    viewer.layers.selection.active = viewer.layers[all_gene_layer_ind]
    napari.run()


def viewer_status_on_select(pointsLayer, nb, viewer):
    """
       indicate selected data in viewer status.
    """

    def indicate_selected(selectedData):
        if selectedData is not None:
            if len(selectedData) == 1:
                spot_no = list(selectedData)[0]
                # TODO: make option for either omp or dot product spots
                spot_gene = nb.call_spots.gene_names[nb.omp.gene_no[spot_no]]
                viewer.status = f'Spot {spot_no}, {spot_gene} Selected'

    """
    Listen to selected data changes
    """

    @thread_worker(connect={'yielded': indicate_selected})
    def _watchSelectedData(pointsLayer):
        selectedData = None
        while True:
            time.sleep(1/10)
            oldSelectedData = selectedData
            selectedData = pointsLayer.selected_data
            if oldSelectedData != selectedData:
                yield selectedData
            yield None

    return (_watchSelectedData(pointsLayer))
