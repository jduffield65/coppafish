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
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow



def iss_plot(nb, method='omp'):
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
    if method == 'omp':
        nbp = nb.omp
    else:
        nbp = nb.ref_spots
    qual_ok = quality_threshold(nbp)
    spot_zyx = (nbp.local_yxz + nb.stitch.tile_origin[nbp.tile])[:, [2, 0, 1]]
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

    @viewer.bind_key('c')
    def call_to_view_codes(viewer):
        # on key press
        n_selected = len(viewer.layers[all_gene_layer_ind].selected_data)
        if n_selected == 1:
            view_codes(nb, list(viewer.layers[all_gene_layer_ind].selected_data)[0])
        elif n_selected > 1:
            viewer.status = f'{n_selected} spots selected - need 1 to run diagnostic'
        else:
            viewer.status = 'No spot selected :('

    @viewer.bind_key('s')
    def call_to_view_spot(viewer):
        n_selected = len(viewer.layers[all_gene_layer_ind].selected_data)
        if n_selected == 1:
            view_spot(nb, list(viewer.layers[all_gene_layer_ind].selected_data)[0])
        elif n_selected > 1:
            viewer.status = f'{n_selected} spots selected - need 1 to run diagnostic'
        else:
            viewer.status = 'No spot selected :('

    @viewer.bind_key('o')
    def call_to_view_omp(viewer):
        n_selected = len(viewer.layers[all_gene_layer_ind].selected_data)
        if n_selected == 1:
            if os.path.isfile(str(nb._config_file)):
                view_omp(nb, list(viewer.layers[all_gene_layer_ind].selected_data)[0])
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
    for s in np.unique(gene_color['Symbols']):
        # TODO: set transparency based on spot score
        spots_to_plot = np.isin(nb.omp.gene_no, gene_color[gene_color['Symbols'] == s]['GeneNo'])
        if spots_to_plot.any():
            coords_to_plot = spot_zyx[spots_to_plot]
            spotcolor_to_plot = gene_rgb_nb[nb.omp.gene_no[spots_to_plot]]
            symb_to_plot = np.unique(gene_color[gene_color['Symbols'] == s]['napari_symbol'])[0]
            viewer.add_points(coords_to_plot, face_color=spotcolor_to_plot, symbol=symb_to_plot,
                              name=f'Gene Symbol: {s}', size=point_size, shown=qual_ok[spots_to_plot])
    viewer_status_on_select(viewer.layers[all_gene_layer_ind], nb, viewer)  # so indicates when a spot is selected
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
        update_plot(viewer, nbp, gene_color, my_slider.value()[0], my_slider.value()[1])

    my_slider.valueChanged.connect(lambda x: show_score_thresh(x[0], x[1]))  # When dragging, status will show thresh.
    my_slider.sliderReleased.connect(update_plot_no_args)  # On release of slider, genes shown will change
    viewer.window.add_dock_widget(my_slider, area="left", name='Score Range')
    my_buttons = Window()
    viewer.window.add_dock_widget(my_buttons, area="left", name='Method')

    # TODO: widget when press key, window pops up which allows you to select genes to plot
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


def update_plot(viewer, nbp, gene_color, min_score, max_score):
    viewer.status = 'Score Range = [{:.2f}, {:.2f}]'.format(min_score, max_score)
    if nbp.name == 'omp':
        score = omp_spot_score(nbp)
    else:
        score = nbp.score
    qual_ok = np.array([score > min_score, score <= max_score, nbp.intensity > nbp.intensity_thresh]).all(axis=0)
    for i in range(len(viewer.layers)):
        if 'Gene Symbol' in viewer.layers[i].name:
            s = viewer.layers[i].name[-1]
            spots_to_plot = np.isin(nbp.gene_no, gene_color[gene_color['Symbols'] == s]['GeneNo'])
            viewer.layers[i].shown = qual_ok[spots_to_plot]
        elif viewer.layers[i].name == 'Diagnostic':
            viewer.layers[i].shown = qual_ok


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # creating a push button
        self.active_button = 1
        self.methods = ['OMP', 'Anchor']
        self.button1 = QPushButton(self.methods[0], self)
        # self.button1.move(20, 10)
        self.button1.setCheckable(True)
        # setting calling method by button
        self.button1.clicked.connect(self.button1_clicked)
        self.button1.setChecked(True)
        self.button1.setGeometry(75, 2, 50, 28)  # left, top, width, height

        self.button2 = QPushButton(self.methods[1], self)
        # self.button2.move(135, 10)
        self.button2.setCheckable(True)
        # setting calling method by button
        self.button2.clicked.connect(self.button2_clicked)
        self.button2.setGeometry(140, 2, 50, 28)  # left, top, width, height

    def button1_clicked(self):
        # Only allow one button pressed
        if self.active_button == 1:
            self.button1.setChecked(True)
            self.button2.setChecked(False)
        else:
            self.button1.setChecked(True)
            self.button2.setChecked(False)
            self.active_button = 1
        self.button_clicked()

    def button2_clicked(self):
        # Only allow one button pressed
        if self.active_button == 2:
            self.button2.setChecked(True)
            self.button1.setChecked(False)
        else:
            self.button2.setChecked(True)
            self.button1.setChecked(False)
            self.active_button = 2
        self.button_clicked()

    def button_clicked(self):
        print(f"Button {self.active_button} clicked")
