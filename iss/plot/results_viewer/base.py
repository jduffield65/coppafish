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
from superqt import QDoubleRangeSlider, QDoubleSlider
from PyQt5.QtWidgets import QPushButton, QMainWindow


class iss_plot:
    def __init__(self, nb):
        self.nb = nb
        legend_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'legend')
        gene_legend_info = pd.read_csv(os.path.join(legend_folder, 'gene_color.csv'))
        cell_legend_info = pd.read_csv(os.path.join(legend_folder, 'cell_color.csv'))

        # indices of genes in notebook to gene_color data - quicker to look up integers than names
        # in change_threshold
        n_legend_genes = len(gene_legend_info['GeneNames'])
        self.legend_gene_symbol = np.asarray(gene_legend_info['Symbols'])
        self.legend_gene_no = np.ones(n_legend_genes, dtype=int) * -1
        for i in range(n_legend_genes):
            gene_ind = np.where(self.nb.call_spots.gene_names == gene_legend_info['GeneNames'][i])[0]
            if len(gene_ind) > 0:
                self.legend_gene_no[i] = gene_ind[0]

        # concatenate anchor and omp spots so can use button to switch between them.
        self.omp_0_ind = self.nb.ref_spots.tile.size  # number of anchor spots
        self.n_spots = self.omp_0_ind + self.nb.omp.tile.size  # number of anchor + number of omp spots
        spot_zyx = np.zeros((self.n_spots, 3))
        spot_zyx[:self.omp_0_ind] = (self.nb.ref_spots.local_yxz + self.nb.stitch.tile_origin[self.nb.ref_spots.tile]
                                     )[:, [2, 0, 1]]
        spot_zyx[self.omp_0_ind:] = (self.nb.omp.local_yxz + self.nb.stitch.tile_origin[self.nb.omp.tile])[:, [2, 0, 1]]
        if not self.nb.basic_info.is_3d:
            spot_zyx = spot_zyx[:, 1:]

        show_spots = np.zeros(self.n_spots, dtype=bool)  # indicates spots shown when plot first opened
        show_spots[self.omp_0_ind:] = quality_threshold(self.nb.omp)  # initially show omp spots which passed threshold

        # color to plot for all genes in the notebook
        gene_color = np.ones((len(self.nb.call_spots.gene_names), 3))
        for i in range(n_legend_genes):
            if self.legend_gene_no[i] != -1:
                gene_color[self.legend_gene_no[i]] = [gene_legend_info.loc[i, 'ColorR'],
                                                      gene_legend_info.loc[i, 'ColorG'],
                                                      gene_legend_info.loc[i, 'ColorB']]

        self.viewer = napari.Viewer()
        iss_legend.add_legend(self.viewer, genes=gene_legend_info, cells=cell_legend_info, celltype=False,
                              gene=True)

        # Add all spots in layer as transparent white spots.
        point_size = 10  # with size=4, spots are too small to see
        self.viewer.add_points(spot_zyx, name='Diagnostic', face_color='w', size=point_size + 2, opacity=0,
                               shown=show_spots)
        self.diagnostic_layer_ind = 0

        # Add gene spots with ISS color code - different layer for each symbol
        self.spot_gene_no = np.hstack((self.nb.ref_spots.gene_no, self.nb.omp.gene_no))
        self.label_prefix = 'Gene Symbol'  # prefix of label for layers showing spots
        for s in np.unique(self.legend_gene_symbol):
            # TODO: set transparency based on spot score
            spots_correct_gene = np.isin(self.spot_gene_no, self.legend_gene_no[self.legend_gene_symbol == s])
            if spots_correct_gene.any():
                coords_to_plot = spot_zyx[spots_correct_gene]
                spotcolor_to_plot = gene_color[self.spot_gene_no[spots_correct_gene]]
                symb_to_plot = np.unique(gene_legend_info[self.legend_gene_symbol == s]['napari_symbol'])[0]
                self.viewer.add_points(coords_to_plot, face_color=spotcolor_to_plot, symbol=symb_to_plot,
                                       name=f'{self.label_prefix}: {s}', size=point_size,
                                       shown=show_spots[spots_correct_gene])

        self.viewer.layers.selection.active = self.viewer.layers[self.diagnostic_layer_ind]
        # so indicates when a spot is selected in viewer status
        # It is needed because layer is transparent so can't see when select spot.
        self.viewer_status_on_select()

        # Scores for anchor/omp are different so reset score range when change method
        self.score_range = {'anchor': [self.nb.ref_spots.score_thresh, 1], 'omp': [self.nb.omp.score_thresh, 1]}
        self.score_thresh_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)  # Slider to change score_thresh
        self.score_thresh_slider.setValue(self.score_range['omp'])
        self.score_thresh_slider.setRange(0, 1)
        # When dragging, status will show thresh.
        self.score_thresh_slider.valueChanged.connect(lambda x: self.show_score_thresh(x[0], x[1]))
        # On release of slider, genes shown will change
        self.score_thresh_slider.sliderReleased.connect(self.update_plot)
        self.viewer.window.add_dock_widget(self.score_thresh_slider, area="left", name='Score Range')

        # intensity is calculated same way for anchor / omp method so do not reset intensity threshold
        # when change method.
        self.intensity_thresh_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.intensity_thresh_slider.setRange(0, 1)
        self.intensity_thresh_slider.setValue(self.nb.omp.intensity_thresh)
        # When dragging, status will show thresh.
        self.intensity_thresh_slider.valueChanged.connect(lambda x: self.show_intensity_thresh(x))
        # On release of slider, genes shown will change
        self.intensity_thresh_slider.sliderReleased.connect(self.update_plot)
        self.viewer.window.add_dock_widget(self.intensity_thresh_slider, area="left", name='Intensity Threshold')

        self.method_buttons = ButtonMethodWindow()  # Buttons to change between Anchor and OMP spots showing.
        self.method_buttons.button_anchor.clicked.connect(self.button_anchor_clicked)
        self.method_buttons.button_omp.clicked.connect(self.button_omp_clicked)
        self.viewer.window.add_dock_widget(self.method_buttons, area="left", name='Method')

        self.key_call_functions()
        napari.run()

    def viewer_status_on_select(self):
        """
           indicate selected data in viewer status.
        """

        def indicate_selected(selectedData):
            if selectedData is not None:
                n_selected = len(selectedData)
                if n_selected == 1:
                    spot_no = list(selectedData)[0]
                    if self.method_buttons.method == 'OMP':
                        spot_no = spot_no - self.omp_0_ind
                        spot_gene = self.nb.call_spots.gene_names[self.nb.omp.gene_no[spot_no]]
                    else:
                        spot_gene = self.nb.call_spots.gene_names[self.nb.ref_spots.gene_no[spot_no]]
                    self.viewer.status = f'Spot {spot_no}, {spot_gene} Selected'
                elif n_selected > 1:
                    self.viewer.status = f'{n_selected} spots selected'

        """
        Listen to selected data changes
        """

        @thread_worker(connect={'yielded': indicate_selected})
        def _watchSelectedData(pointsLayer):
            selectedData = None
            while True:
                time.sleep(1 / 10)
                oldSelectedData = selectedData
                selectedData = pointsLayer.selected_data
                if oldSelectedData != selectedData:
                    yield selectedData
                yield None

        return (_watchSelectedData(self.viewer.layers[self.diagnostic_layer_ind]))

    def update_plot(self):
        if self.method_buttons.method == 'OMP':
            score = omp_spot_score(self.nb.omp)
            method_ind = np.arange(self.omp_0_ind, self.n_spots)
            intensity_ok = self.nb.omp.intensity > self.intensity_thresh_slider.value()
        else:
            score = self.nb.ref_spots.score
            method_ind = np.arange(self.omp_0_ind)
            intensity_ok = self.nb.ref_spots.intensity > self.intensity_thresh_slider.value()
        # Keep record of last score range set for each method
        self.score_range[self.method_buttons.method.lower()] = self.score_thresh_slider.value()
        qual_ok = np.array([score > self.score_thresh_slider.value()[0], score <= self.score_thresh_slider.value()[1],
                            intensity_ok]).all(axis=0)
        spots_shown = np.zeros(self.n_spots, dtype=bool)
        spots_shown[method_ind] = qual_ok
        for i in range(len(self.viewer.layers)):
            if i == self.diagnostic_layer_ind:
                self.viewer.layers[i].shown = spots_shown
            elif self.label_prefix in self.viewer.layers[i].name:
                s = self.viewer.layers[i].name[-1]
                spots_correct_gene = np.isin(self.spot_gene_no,
                                             self.legend_gene_no[self.legend_gene_symbol == s])
                self.viewer.layers[i].shown = spots_shown[spots_correct_gene]

    def show_score_thresh(self, low_value, high_value):
        self.viewer.status = self.method_buttons.method + ': Score Range = [{:.2f}, {:.2f}]'.format(low_value,
                                                                                                    high_value)

    def show_intensity_thresh(self, value):
        self.viewer.status = 'Intensity Threshold = {:.3f}'.format(value)

    def button_anchor_clicked(self):
        # Only allow one button pressed
        if self.method_buttons.method == 'Anchor':
            self.method_buttons.button_anchor.setChecked(True)
            self.method_buttons.button_omp.setChecked(False)
        else:
            self.method_buttons.button_anchor.setChecked(True)
            self.method_buttons.button_omp.setChecked(False)
            self.method_buttons.method = 'Anchor'
            # Because method has changed, also need to change score range
            self.score_thresh_slider.setValue(self.score_range['anchor'])
            self.update_plot()

    def button_omp_clicked(self):
        # Only allow one button pressed
        if self.method_buttons.method == 'OMP':
            self.method_buttons.button_omp.setChecked(True)
            self.method_buttons.button_anchor.setChecked(False)
        else:
            self.method_buttons.button_omp.setChecked(True)
            self.method_buttons.button_anchor.setChecked(False)
            self.method_buttons.method = 'OMP'
            # Because method has changed, also need to change score range
            self.score_thresh_slider.setValue(self.score_range['omp'])
            self.update_plot()

    def get_selected_spot(self):
        """
        Returns spot_no selected if only one selected (this is the spot_no relavent to the Notebook i.e.
        if omp, the index of the spot in nb.omp is returned).
        Otherwise, returns None and indicates why in viewer status.
        """
        n_selected = len(self.viewer.layers[self.diagnostic_layer_ind].selected_data)
        if n_selected == 1:
            spot_no = list(self.viewer.layers[self.diagnostic_layer_ind].selected_data)[0]
            if self.method_buttons.method == 'OMP':
                spot_no = spot_no - self.omp_0_ind  # return spot_no as saved in self.nb for current method.
        elif n_selected > 1:
            self.viewer.status = f'{n_selected} spots selected - need 1 to run diagnostic'
            spot_no = None
        else:
            self.viewer.status = 'No spot selected :('
            spot_no = None
        return spot_no

    def key_call_functions(self):
        """
        Contains all functions which can be called by pressing a key with napari viewer open
        """
        @self.viewer.bind_key('b')
        def call_to_view_bm(viewer):
            view_bleed_matrix(self.nb)

        @self.viewer.bind_key('g')
        def call_to_view_bm(viewer):
            view_bled_codes(self.nb)

        @self.viewer.bind_key('c')
        def call_to_view_codes(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_codes(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key('s')
        def call_to_view_spot(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_spot(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key('o')
        def call_to_view_omp(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                if os.path.isfile(str(self.nb._config_file)):
                    view_omp(self.nb, spot_no, self.method_buttons.method)
                else:
                    self.viewer.status = 'Notebook config file not valid :('


class ButtonMethodWindow(QMainWindow):
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

