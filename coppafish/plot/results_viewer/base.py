import os
import pandas as pd
import numpy as np
from ...call_spots.qual_check import quality_threshold
from .legend import add_legend
from ..call_spots import view_codes, view_bleed_matrix, view_bled_codes, view_spot, view_intensity, gene_counts, \
    view_scaled_k_means
from ...call_spots import omp_spot_score, get_intensity_thresh
from ..omp import view_omp, view_omp_fit, view_omp_score, histogram_score, histogram_2d_score
from ..omp.coefs import view_score  # gives import error if call from call_spots.dot_product
from ...setup import Notebook
import napari
from napari.qt import thread_worker
import time
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider, QDoubleSlider, QRangeSlider
from PyQt5.QtWidgets import QPushButton, QMainWindow
from napari.layers.points import Points
from napari.layers.points._points_constants import Mode
import warnings
from typing import Optional, Union


class Viewer:
    def __init__(self, nb: Notebook, background_image: Optional[Union[str, np.ndarray]] = 'dapi',
                 gene_marker_file: Optional[str] = None):
        """
        This is the function to view the results of the pipeline
        i.e. the spots found and which genes they were assigned to.

        Args:
            nb: Notebook containing at least the `ref_spots` page.
            background_image: Optional file_name or image that will be plotted as the background image.
                If image, z dimension needs to be first i.e. `n_z x n_y x n_x` if 3D or `n_y x n_x` if 2D.
                If pass *2D* image for *3D* data, will show same image as background on each z-plane.
            gene_marker_file: Path to csv file containing marker and color for each gene. There must be 6 columns
                in the csv file with the following headers:

                * GeneNames - str, name of gene with first letter capital
                * ColorR - float, Rgb color for plotting
                * ColorG - float, rGb color for plotting
                * ColorB - float, rgB color for plotting
                * napari_symbol - str, symbol used to plot in napari
                * mpl_symbol - str, equivalent of napari symbol in matplotlib.

                If it is not provided, then the default file *coppafish/plot/results_viewer/legend.gene_color.csv*
                will be used.
        """
        # TODO: flip y axis so origin bottom left
        self.nb = nb
        if gene_marker_file is None:
            gene_marker_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gene_color.csv')
        gene_legend_info = pd.read_csv(gene_marker_file)

        # indices of genes in notebook to gene_color data - quicker to look up integers than names
        # in change_threshold
        n_legend_genes = len(gene_legend_info['GeneNames'])
        self.legend_gene_symbol = np.asarray(gene_legend_info['mpl_symbol'])
        self.legend_gene_no = np.ones(n_legend_genes, dtype=int) * -1
        for i in range(n_legend_genes):
            # TODO: maybe guard against different cases in this comparison
            gene_ind = np.where(self.nb.call_spots.gene_names == gene_legend_info['GeneNames'][i])[0]
            if len(gene_ind) > 0:
                self.legend_gene_no[i] = gene_ind[0]

        # concatenate anchor and omp spots so can use button to switch between them.
        self.omp_0_ind = self.nb.ref_spots.tile.size  # number of anchor spots
        if self.nb.has_page('omp'):
            self.n_spots = self.omp_0_ind + self.nb.omp.tile.size  # number of anchor + number of omp spots
        else:
            self.n_spots = self.omp_0_ind
        spot_zyx = np.zeros((self.n_spots, 3))
        spot_zyx[:self.omp_0_ind] = (self.nb.ref_spots.local_yxz + self.nb.stitch.tile_origin[self.nb.ref_spots.tile]
                                     )[:, [2, 0, 1]]
        if self.nb.has_page('omp'):
            spot_zyx[self.omp_0_ind:] = (self.nb.omp.local_yxz + self.nb.stitch.tile_origin[self.nb.omp.tile]
                                         )[:, [2, 0, 1]]
        if not self.nb.basic_info.is_3d:
            spot_zyx = spot_zyx[:, 1:]

        # indicate spots shown when plot first opened - omp if exists, else anchor
        if self.nb.has_page('omp'):
            show_spots = np.zeros(self.n_spots, dtype=bool)
            show_spots[self.omp_0_ind:] = quality_threshold(self.nb, 'omp')
        else:
            show_spots = quality_threshold(self.nb, 'anchor')

        # color to plot for all genes in the notebook
        gene_color = np.ones((len(self.nb.call_spots.gene_names), 3))
        for i in range(n_legend_genes):
            if self.legend_gene_no[i] != -1:
                gene_color[self.legend_gene_no[i]] = [gene_legend_info.loc[i, 'ColorR'],
                                                      gene_legend_info.loc[i, 'ColorG'],
                                                      gene_legend_info.loc[i, 'ColorB']]

        self.viewer = napari.Viewer()
        self.viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)

        # Add background image if given
        self.diagnostic_layer_ind = 0
        self.image_layer_ind = None
        if background_image is not None:
            if isinstance(background_image, str):
                if background_image.lower() == 'dapi':
                    file_name = nb.file_names.big_dapi_image
                elif background_image.lower() == 'anchor':
                    file_name = nb.file_names.big_anchor_image
                else:
                    file_name = background_image
                if file_name is not None and os.path.isfile(file_name):
                    background_image = np.load(file_name)
                    if file_name.endswith('.npz'):
                        # Assume image is first array if .npz file
                        background_image = background_image[background_image._files[0]]
                else:
                    background_image = None
                    warnings.warn(f'No file exists with address =\n{file_name}\nso plotting with no background.')
            if background_image is not None:
                self.viewer.add_image(background_image)
                self.diagnostic_layer_ind = 1
                self.image_layer_ind = 0
                self.viewer.layers[self.image_layer_ind].contrast_limits_range = [background_image.min(),
                                                                                  background_image.max()]
                self.image_contrast_slider = QRangeSlider(Qt.Orientation.Horizontal)  # Slider to change score_thresh
                self.image_contrast_slider.setRange(background_image.min(), background_image.max())
                # Make starting lower bound contrast the 95th percentile value so most appears black
                # Use mid_z to quicken up calculation
                mid_z = int(background_image.shape[0]/2)
                start_contrast = np.percentile(background_image[mid_z], [95, 99.99]).astype(int).tolist()
                self.image_contrast_slider.setValue(start_contrast)
                self.change_image_contrast()
                # When dragging, status will show contrast values.
                self.image_contrast_slider.valueChanged.connect(lambda x: self.show_image_contrast(x[0], x[1]))
                # On release of slider, genes shown will change
                self.image_contrast_slider.sliderReleased.connect(self.change_image_contrast)

        # Add legend indicating genes plotted
        self.legend = {'fig': None, 'ax': None}
        self.legend['fig'], self.legend['ax'], n_gene_label_letters = \
            add_legend(gene_legend_info=gene_legend_info, genes=nb.call_spots.gene_names)
        # xy is position of each symbol in legend, need to see which gene clicked on.
        self.legend['xy'] = np.zeros((len(self.legend['ax'].collections), 2), dtype=float)
        self.legend['gene_no'] = np.zeros(len(self.legend['ax'].collections), dtype=int)
        # In legend, each gene name label has at most n_gene_label_letters letters so need to crop
        # gene_names in notebook when looking for corresponding gene in legend.
        gene_names_crop = np.asarray([gene_name[:n_gene_label_letters] for gene_name in nb.call_spots.gene_names])
        for i in range(self.legend['xy'].shape[0]):
            # Position of label for each gene in legend window
            self.legend['xy'][i] = np.asarray(self.legend['ax'].collections[i].get_offsets())
            # gene no in notebook that each position in the legend corresponds to
            self.legend['gene_no'][i] = \
                np.where(gene_names_crop == self.legend['ax'].texts[i].get_text())[0][0]
        self.legend['fig'].mpl_connect('button_press_event', self.update_genes)
        self.viewer.window.add_dock_widget(self.legend['fig'], area='left', name='Genes')
        self.active_genes = np.arange(len(nb.call_spots.gene_names))  # start with all genes shown

        if background_image is not None:
            # Slider to change background image contrast
            self.viewer.window.add_dock_widget(self.image_contrast_slider, area="left", name='Image Contrast')

        # Add all spots in layer as transparent white spots.
        point_size = 10  # with size=4, spots are too small to see
        self.viewer.add_points(spot_zyx, name='Diagnostic', face_color='w', size=point_size + 2, opacity=0,
                               shown=show_spots)

        # Add gene spots with coppafish color code - different layer for each symbol
        if self.nb.has_page('omp'):
            self.spot_gene_no = np.hstack((self.nb.ref_spots.gene_no, self.nb.omp.gene_no))
        else:
            self.spot_gene_no = self.nb.ref_spots.gene_no
        self.label_prefix = 'Gene Symbol:'  # prefix of label for layers showing spots
        for s in np.unique(self.legend_gene_symbol):
            spots_correct_gene = np.isin(self.spot_gene_no, self.legend_gene_no[self.legend_gene_symbol == s])
            if spots_correct_gene.any():
                coords_to_plot = spot_zyx[spots_correct_gene]
                spotcolor_to_plot = gene_color[self.spot_gene_no[spots_correct_gene]]
                symb_to_plot = np.unique(gene_legend_info[self.legend_gene_symbol == s]['napari_symbol'])[0]
                self.viewer.add_points(coords_to_plot, face_color=spotcolor_to_plot, symbol=symb_to_plot,
                                       name=f'{self.label_prefix}{s}', size=point_size,
                                       shown=show_spots[spots_correct_gene])
                # TODO: showing multiple z-planes at once is possible using out_of_slice_display=True,
                #  but at the moment cannot use at same time as show.
                #  When this works, can change n_z shown by changing z-dimension of point_size i.e.
                #  point_size = [z_size, 10, 10] and z_size changes.
                #  On next napari, release should be able to change thickness of spots in z-direction i.e. control what
                #  number of z-planes can be seen at any one time:
                #  https://github.com/napari/napari/issues/4816#issuecomment-1186600574.
                #  Then see find_spots viewer for how I added a z-thick slider

        self.viewer.layers.selection.active = self.viewer.layers[self.diagnostic_layer_ind]
        # so indicates when a spot is selected in viewer status
        # It is needed because layer is transparent so can't see when select spot.
        self.viewer_status_on_select()

        config = self.nb.get_config()['thresholds']
        self.score_omp_multiplier = config['score_omp_multiplier']
        self.score_thresh_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)  # Slider to change score_thresh
        # Scores for anchor/omp are different so reset score range when change method
        self.score_range = {'anchor': [config['score_ref'], 1]}
        if self.nb.has_page('omp'):
            self.score_range['omp'] = [config['score_omp'], 1]
            self.score_thresh_slider.setValue(self.score_range['omp'])
        else:
            self.score_thresh_slider.setValue(self.score_range['anchor'])
        self.score_thresh_slider.setRange(0, 1)
        # When dragging, status will show thresh.
        self.score_thresh_slider.valueChanged.connect(lambda x: self.show_score_thresh(x[0], x[1]))
        # On release of slider, genes shown will change
        self.score_thresh_slider.sliderReleased.connect(self.update_plot)
        self.viewer.window.add_dock_widget(self.score_thresh_slider, area="left", name='Score Range')

        # OMP Score Multiplier Slider
        self.omp_score_multiplier_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.omp_score_multiplier_slider.setValue(self.score_omp_multiplier)
        self.omp_score_multiplier_slider.setRange(0, 50)
        self.omp_score_multiplier_slider.valueChanged.connect(lambda x: self.show_omp_score_multiplier(x))
        self.omp_score_multiplier_slider.sliderReleased.connect(self.update_plot)

        # intensity is calculated same way for anchor / omp method so do not reset intensity threshold
        # when change method.
        self.intensity_thresh_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.intensity_thresh_slider.setRange(0, 1)
        intensity_thresh = get_intensity_thresh(nb)
        self.intensity_thresh_slider.setValue(intensity_thresh)
        # When dragging, status will show thresh.
        self.intensity_thresh_slider.valueChanged.connect(lambda x: self.show_intensity_thresh(x))
        # On release of slider, genes shown will change
        self.intensity_thresh_slider.sliderReleased.connect(self.update_plot)
        self.viewer.window.add_dock_widget(self.intensity_thresh_slider, area="left", name='Intensity Threshold')

        if self.nb.has_page('omp'):
            self.method_buttons = ButtonMethodWindow('OMP')  # Buttons to change between Anchor and OMP spots showing.
        else:
            self.method_buttons = ButtonMethodWindow('Anchor')
        self.method_buttons.button_anchor.clicked.connect(self.button_anchor_clicked)
        self.method_buttons.button_omp.clicked.connect(self.button_omp_clicked)
        if self.nb.has_page('omp'):
            self.viewer.window.add_dock_widget(self.omp_score_multiplier_slider, area="left",
                                               name='OMP Score Multiplier')
            # Only have button to change method if have omp page too.
            self.viewer.window.add_dock_widget(self.method_buttons, area="left", name='Method')

        self.key_call_functions()
        if self.nb.basic_info.is_3d:
            self.viewer.dims.axis_labels = ['z', 'y', 'x']
        else:
            self.viewer.dims.axis_labels = ['y', 'x']

        napari.run()

    def viewer_status_on_select(self):
        # indicate selected data in viewer status.

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
        # This updates the spots plotted to reflect score_range and intensity threshold selected by sliders,
        # method selected by button and genes selected through clicking on the legend.
        if self.method_buttons.method == 'OMP':
            score = omp_spot_score(self.nb.omp, self.omp_score_multiplier_slider.value())
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
        # Only show spots which belong to a gene that is active and that passes quality threshold
        genes_shown = np.isin(self.spot_gene_no[method_ind], self.active_genes)
        spots_shown[method_ind[genes_shown]] = qual_ok[genes_shown]
        for i in range(len(self.viewer.layers)):
            if i == self.diagnostic_layer_ind:
                self.viewer.layers[i].shown = spots_shown
            elif self.label_prefix in self.viewer.layers[i].name:
                s = self.viewer.layers[i].name.replace(self.label_prefix,'')
                spots_correct_gene = np.isin(self.spot_gene_no,
                                             self.legend_gene_no[self.legend_gene_symbol == s])
                self.viewer.layers[i].shown = spots_shown[spots_correct_gene]

    def update_genes(self, event):
        # When click on a gene in the legend will remove/add that gene to plot.
        # When right-click on a gene, it will only show that gene.
        # When click on a gene which is the only selected gene, it will return to showing all genes.
        xy_clicked = np.array([event.xdata, event.ydata])
        xy_gene = np.zeros(2)
        for i in range(2):
            xy_gene[i] = self.legend['xy'][np.abs(xy_clicked[i] - self.legend['xy'][:, i]).argmin(), i]
        gene_clicked = np.where((self.legend['xy'] == xy_gene).all(axis=1))[0][0]
        gene_no = self.legend['gene_no'][gene_clicked]
        n_active = self.active_genes.size
        is_active = np.isin(gene_no, self.active_genes)
        active_genes_last = self.active_genes.copy()
        if is_active and n_active == 1:
            # If gene is only one selected, any click on it will return to all genes
            self.active_genes = np.sort(self.legend['gene_no'])
            # 1st argument in setdiff1d is always the larger array
            changed_genes = np.setdiff1d(self.active_genes, active_genes_last)
        elif event.button.name == 'RIGHT':
            # If right-click on a gene, will select only that gene
            self.active_genes = np.asarray([gene_no])
            # 1st argument in setdiff1d is always the larger array
            changed_genes = np.setdiff1d(active_genes_last, self.active_genes)
            if not is_active:
                # also need to changed clicked gene if was not already active
                changed_genes = np.append(changed_genes, gene_no)
        elif is_active:
            # If single-click on a gene which is selected, will remove that gene
            self.active_genes = np.setdiff1d(self.active_genes, gene_no)
            changed_genes = np.asarray([gene_no])
        elif not is_active:
            # If single-click on a gene which is not selected, it will be removed
            self.active_genes = np.append(self.active_genes, gene_no)
            changed_genes = np.asarray([gene_no])

        # Change formatting
        for g in changed_genes:
            i = np.where(self.legend['gene_no'] == g)[0][0]
            if np.isin(g, self.active_genes):
                alpha = 1
            else:
                alpha = 0.5  # If not selected, make transparent
            self.legend['ax'].collections[i].set_alpha(alpha)
            self.legend['ax'].texts[i].set_alpha(alpha)
        self.legend['fig'].draw()
        self.update_plot()

    def show_score_thresh(self, low_value, high_value):
        self.viewer.status = self.method_buttons.method + ': Score Range = [{:.2f}, {:.2f}]'.format(low_value,
                                                                                                    high_value)

    def show_omp_score_multiplier(self, value):
        self.viewer.status = 'OMP Score Multiplier = {:.2f}'.format(value)

    def show_intensity_thresh(self, value):
        self.viewer.status = 'Intensity Threshold = {:.3f}'.format(value)

    def show_image_contrast(self, low_value, high_value):
        # Show contrast of background image while dragging
        self.viewer.status = 'Image Contrast Limits: [{:.0f}, {:.0f}]'.format(low_value, high_value)

    def change_image_contrast(self):
        # Change contrast of background image
        self.viewer.layers[self.image_layer_ind].contrast_limits = [self.image_contrast_slider.value()[0],
                                                                    self.image_contrast_slider.value()[1]]

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
        # Returns spot_no selected if only one selected (this is the spot_no relavent to the Notebook i.e.
        # if omp, the index of the spot in nb.omp is returned).
        # Otherwise, returns None and indicates why in viewer status.
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
        # Contains all functions which can be called by pressing a key with napari viewer open
        @Points.bind_key('Space', overwrite=True)
        def change_zoom_select_mode(layer):
            if layer.mode == Mode.PAN_ZOOM:
                layer.mode = Mode.SELECT
                self.viewer.help = 'Mode: Select'
            elif layer.mode == Mode.SELECT:
                layer.mode = Mode.PAN_ZOOM
                self.viewer.help = 'Mode: Pan/Zoom'

        @self.viewer.bind_key('i')
        def remove_background_image(viewer):
            # Make background image visible / remove it
            if self.image_layer_ind is not None:
                if viewer.layers[self.image_layer_ind].visible:
                    viewer.layers[self.image_layer_ind].visible = False
                else:
                    viewer.layers[self.image_layer_ind].visible = True

        @self.viewer.bind_key('b')
        def call_to_view_bm(viewer):
            view_bleed_matrix(self.nb)

        @self.viewer.bind_key('g')
        def call_to_view_bm(viewer):
            view_bled_codes(self.nb)

        @self.viewer.bind_key('Shift-g')
        def call_to_gene_counts(viewer):
            if self.nb.has_page('omp'):
                score_multiplier = self.omp_score_multiplier_slider.value()
                score_omp_thresh = self.score_range['omp'][0]
            else:
                score_multiplier = None
                score_omp_thresh = None
            score_thresh = self.score_range['anchor'][0]
            intensity_thresh = self.intensity_thresh_slider.value()
            gene_counts(self.nb, None, None, score_thresh, intensity_thresh, score_omp_thresh, score_multiplier)

        @self.viewer.bind_key('h')
        def call_to_view_omp_score(viewer):
            if self.nb.has_page('omp'):
                score_multiplier = self.omp_score_multiplier_slider.value()
            else:
                score_multiplier = None
            histogram_score(self.nb, self.method_buttons.method, score_multiplier)

        @self.viewer.bind_key('Shift-h')
        def call_to_view_omp_score(viewer):
            if self.nb.has_page('omp'):
                histogram_2d_score(self.nb, self.omp_score_multiplier_slider.value())

        @self.viewer.bind_key('k')
        def call_to_view_omp_score(viewer):
            view_scaled_k_means(self.nb)

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

        @self.viewer.bind_key('d')
        def call_to_view_omp_score(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_score(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key('Shift-i')
        def call_to_view_omp_score(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_intensity(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key('o')
        def call_to_view_omp(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_omp(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key('Shift-o')
        def call_to_view_omp(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_omp_fit(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key('Shift-s')
        def call_to_view_omp_score(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_omp_score(self.nb, spot_no, self.method_buttons.method, self.omp_score_multiplier_slider.value())


class ButtonMethodWindow(QMainWindow):
    def __init__(self, active_button: str = 'OMP'):
        super().__init__()
        self.button_anchor = QPushButton('Anchor', self)
        self.button_anchor.setCheckable(True)
        self.button_anchor.setGeometry(75, 2, 50, 28)  # left, top, width, height

        self.button_omp = QPushButton('OMP', self)
        self.button_omp.setCheckable(True)
        self.button_omp.setGeometry(140, 2, 50, 28)  # left, top, width, height
        if active_button.lower() == 'omp':
            # Initially, show OMP spots
            self.button_omp.setChecked(True)
            self.method = 'OMP'
        elif active_button.lower() == 'anchor':
            # Initially, show Anchor spots
            self.button_anchor.setChecked(True)
            self.method = 'Anchor'
        else:
            raise ValueError(f"active_button should be 'Anchor' or 'OMP' but {active_button} was given.")
