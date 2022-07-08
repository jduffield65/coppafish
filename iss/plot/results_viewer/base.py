import os
import pandas as pd
import numpy as np
from ...call_spots.base import quality_threshold
from .legend import iss_legend
from ..call_spots import view_codes, view_bleed_matrix, view_bled_codes, view_spot, omp_spot_score
from ..omp import view_omp
import napari
from napari.qt import thread_worker
import time
from qtpy.QtCore import Qt
from superqt import QLabeledDoubleRangeSlider, QDoubleRangeSlider
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow


def iss_plot(nb):
    # TODO: get rid of button if is no omp page
    legend_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'legend')
    gene_color = pd.read_csv(os.path.join(legend_folder, 'gene_color.csv'))
    gene_color['GeneNo'] = np.ones((len(gene_color['GeneNames']), 1), dtype=int) * -1

    # Add indices of genes in notebook to gene_color data - quicker to look up integers than names in change_threshold
    gene_no = np.ones((len(gene_color['GeneNames']), 1), dtype=int) * -1
    for i in range(len(gene_color['GeneNames'])):
        gene_ind = np.where(nb.call_spots.gene_names == gene_color['GeneNames'][i])[0]
        if len(gene_ind) > 0:
            gene_no[i] = gene_ind[0]
    gene_color['GeneNo'] = gene_no

    cell_color = pd.read_csv(os.path.join(legend_folder, 'cell_color.csv'))
    # combine anchor and omp spots so can use button to switch between them.
    n_anchor_spots = nb.ref_spots.tile.size
    n_spots = nb.ref_spots.tile.size + nb.omp.tile.size
    omp_0_ind = n_anchor_spots
    qual_ok = np.zeros(n_spots, dtype=bool)
    # initially show omp spots
    qual_ok[omp_0_ind:] = quality_threshold(nb.omp)
    spot_zyx = np.zeros((n_spots, 3))
    spot_zyx[:omp_0_ind] = (nb.ref_spots.local_yxz + nb.stitch.tile_origin[nb.ref_spots.tile])[:, [2, 0, 1]]
    spot_zyx[omp_0_ind:] = (nb.omp.local_yxz + nb.stitch.tile_origin[nb.omp.tile])[:, [2, 0, 1]]
    if not nb.basic_info.is_3d:
        spot_zyx = spot_zyx[:, 1:]

    im_dims = np.rint(spot_zyx.max(axis=0)).astype(int)
    background_image = np.zeros(im_dims, dtype=np.int8)
    # viewer = napari.view_image(background_image)
    viewer = napari.Viewer()
    iss_legend.add_legend(viewer, genes=gene_color, cells=cell_color, celltype=False, gene=True)

    # Add all points as white dots
    point_size = 10  # with size=4, spots are too small to see
    viewer.add_points(spot_zyx, name='Diagnostic', face_color='w', size=point_size+2, opacity=0, shown=qual_ok)
    all_gene_layer_ind = 0

    gene_color_dict = dict()
    for g in gene_color.index:
        gene_color_dict[gene_color.loc[g, 'GeneNames']] = (
        gene_color.loc[g, 'ColorR'], gene_color.loc[g, 'ColorG'], gene_color.loc[g, 'ColorB'])
    # color of all genes in the notebook
    gene_rgb_nb = np.ones((len(nb.call_spots.gene_names), 3))
    for i in gene_color.index:
        if gene_color.loc[i, 'GeneNo'] != -1:
            gene_rgb_nb[gene_color.loc[i, 'GeneNo']] = [gene_color.loc[i, 'ColorR'], gene_color.loc[i, 'ColorG'],
                                                          gene_color.loc[i, 'ColorB']]

    # Add gene spots with ISS color code
    all_gene_no = np.hstack((nb.ref_spots.gene_no, nb.omp.gene_no))
    for s in np.unique(gene_color['Symbols']):
        # TODO: set transparency based on spot score
        spots_to_plot = np.isin(all_gene_no, gene_color[gene_color['Symbols'] == s]['GeneNo'])
        if spots_to_plot.any():
            coords_to_plot = spot_zyx[spots_to_plot]
            spotcolor_to_plot = gene_rgb_nb[all_gene_no[spots_to_plot]]
            symb_to_plot = np.unique(gene_color[gene_color['Symbols'] == s]['napari_symbol'])[0]
            viewer.add_points(coords_to_plot, face_color=spotcolor_to_plot, symbol=symb_to_plot,
                              name=f'Gene Symbol: {s}', size=point_size, shown=qual_ok[spots_to_plot])

    my_buttons = Window()
    viewer_status_on_select(viewer.layers[all_gene_layer_ind], nb, viewer, omp_0_ind)  # so indicates when a spot is selected
    viewer.layers.selection.active = viewer.layers[all_gene_layer_ind]

    my_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
    my_slider.setSingleStep(0.01)
    my_slider.setValue((nb.omp.score_thresh, 1))
    my_slider.setRange(0, 1)
    my_slider.setMaximum(1)
    # # Below is if used QLabeledDoubleRangeSlider, but issue is that widget does not fill entire range so looks weird.
    # my_slider.setDecimals(2)
    # # set width of labels so shows all decimal places
    # for label in [my_slider._min_label, my_slider._max_label, my_slider._handle_labels[0], my_slider._handle_labels[1]]:
    #     label.setFixedWidth(55)

    def show_score_thresh(low_value, high_value):
        viewer.status = 'Score Range = [{:.2f}, {:.2f}]'.format(low_value, high_value)

    def update_plot_no_args():
        if my_buttons.method == 'OMP':
            update_plot(viewer, nb.omp, gene_color, all_gene_no, omp_0_ind, all_gene_layer_ind,
                        my_slider.value()[0], my_slider.value()[1])
        else:
            update_plot(viewer, nb.ref_spots, gene_color, all_gene_no, omp_0_ind, all_gene_layer_ind,
                        my_slider.value()[0], my_slider.value()[1])

    def button_anchor_clicked():
        # Only allow one button pressed
        if my_buttons.method == 'Anchor':
            my_buttons.button_anchor.setChecked(True)
            my_buttons.button_omp.setChecked(False)
        else:
            viewer.status = 'Now showing anchor spots'
            my_buttons.button_anchor.setChecked(True)
            my_buttons.button_omp.setChecked(False)
            my_buttons.method = 'Anchor'
            update_plot_no_args()

    def button_omp_clicked():
        # Only allow one button pressed
        if my_buttons.method == 'OMP':
            my_buttons.button_omp.setChecked(True)
            my_buttons.button_anchor.setChecked(False)
        else:
            viewer.status = 'Now showing omp spots'
            my_buttons.button_omp.setChecked(True)
            my_buttons.button_anchor.setChecked(False)
            my_buttons.method = 'OMP'
            update_plot_no_args()

    my_buttons.button_anchor.clicked.connect(button_anchor_clicked)
    my_buttons.button_omp.clicked.connect(button_omp_clicked)
    my_slider.valueChanged.connect(lambda x: show_score_thresh(x[0], x[1]))  # When dragging, status will show thresh.
    my_slider.sliderReleased.connect(update_plot_no_args)  # On release of slider, genes shown will change
    viewer.window.add_dock_widget(my_slider, area="left", name='Score Range')
    viewer.window.add_dock_widget(my_buttons, area="left", name='Method')

    @viewer.bind_key('c')
    def call_to_view_codes(viewer):
        # on key press
        n_selected = len(viewer.layers[all_gene_layer_ind].selected_data)
        if n_selected == 1:
            spot_no = list(viewer.layers[all_gene_layer_ind].selected_data)[0]
            if my_buttons.method == 'OMP':
                spot_no = spot_no - omp_0_ind
            view_codes(nb, spot_no, my_buttons.method)
        elif n_selected > 1:
            viewer.status = f'{n_selected} spots selected - need 1 to run diagnostic'
        else:
            viewer.status = 'No spot selected :('

    @viewer.bind_key('s')
    def call_to_view_spot(viewer):
        n_selected = len(viewer.layers[all_gene_layer_ind].selected_data)
        if n_selected == 1:
            spot_no = list(viewer.layers[all_gene_layer_ind].selected_data)[0]
            if my_buttons.method == 'OMP':
                spot_no = spot_no - omp_0_ind
            view_spot(nb, spot_no, my_buttons.method)
        elif n_selected > 1:
            viewer.status = f'{n_selected} spots selected - need 1 to run diagnostic'
        else:
            viewer.status = 'No spot selected :('

    @viewer.bind_key('o')
    def call_to_view_omp(viewer):
        n_selected = len(viewer.layers[all_gene_layer_ind].selected_data)
        if n_selected == 1:
            spot_no = list(viewer.layers[all_gene_layer_ind].selected_data)[0]
            if my_buttons.method == 'OMP':
                spot_no = spot_no - omp_0_ind
            if os.path.isfile(str(nb._config_file)):
                # Need to access properties in omp section of config file
                view_omp(nb, spot_no, my_buttons.method)
            else:
                viewer.status = 'Notebook config file not valid :('
        elif n_selected > 1:
            viewer.status = f'{n_selected} spots selected - need 1 to run diagnostic'
        else:
            viewer.status = 'No spot selected :('

    @viewer.bind_key('b')
    def call_to_view_bm(viewer):
        view_bleed_matrix(nb)

    @viewer.bind_key('g')
    def call_to_view_bm(viewer):
        view_bled_codes(nb)

    # TODO: widget when press key, window pops up which allows you to select genes to plot
    napari.run()


def viewer_status_on_select(pointsLayer, nb, viewer, omp_0_ind):
    """
       indicate selected data in viewer status.
    """

    def indicate_selected(selectedData):
        if selectedData is not None:
            if len(selectedData) == 1:
                spot_no = list(selectedData)[0]
                if spot_no >= omp_0_ind:
                    method = 'omp'
                else:
                    method = 'anchor'
                if method == 'omp':
                    spot_no = spot_no - omp_0_ind
                    spot_gene = nb.call_spots.gene_names[nb.omp.gene_no[spot_no]]
                else:
                    spot_gene = nb.call_spots.gene_names[nb.ref_spots.gene_no[spot_no]]
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


def update_plot(viewer, nbp, gene_color, all_gene_no, omp_0_ind, all_gene_layer_ind, min_score, max_score):
    n_spots = viewer.layers[all_gene_layer_ind].shown.size
    if nbp.name == 'omp':
        score = omp_spot_score(nbp)
        array_ind = np.arange(omp_0_ind, n_spots)
    else:
        score = nbp.score
        array_ind = np.arange(omp_0_ind)
    qual_ok_crop = np.array([score > min_score, score <= max_score, nbp.intensity > nbp.intensity_thresh]).all(axis=0)
    qual_ok = np.zeros(n_spots, dtype=bool)
    qual_ok[array_ind] = qual_ok_crop
    for i in range(len(viewer.layers)):
        if 'Gene Symbol' in viewer.layers[i].name:
            s = viewer.layers[i].name[-1]
            spots_to_plot = np.isin(all_gene_no, gene_color[gene_color['Symbols'] == s]['GeneNo'])
            viewer.layers[i].shown = qual_ok[spots_to_plot]
        elif viewer.layers[i].name == 'Diagnostic':
            viewer.layers[i].shown = qual_ok


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.button_anchor = QPushButton('Anchor', self)
        self.button_anchor.setCheckable(True)
        self.button_anchor.setGeometry(75, 2, 50, 28)  # left, top, width, height

        self.button_omp = QPushButton('OMP', self)
        self.button_omp.setCheckable(True)
        self.button_omp.setGeometry(140, 2, 50, 28)  # left, top, width, height
        # Initially, show OMP spots
        self.button_omp.setChecked(True)
        self.method = 'OMP'

