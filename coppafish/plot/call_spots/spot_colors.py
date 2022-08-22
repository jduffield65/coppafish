import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RangeSlider
from ...call_spots.qual_check import omp_spot_score, get_intensity_thresh
from ...call_spots.background import fit_background
from ...setup import Notebook
from ...spot_colors.base import get_spot_colors
import matplotlib
from typing import List, Optional, Tuple, Union
plt.style.use('dark_background')
try:
    # So matplotlib plots pop out
    # Put in try so don't get error when unit testing in continuous integration
    # which is in headless mode
    matplotlib.use('qtagg')
except ImportError:
    pass


class ColorPlotBase:
    def __init__(self, images: List, norm_factor: Optional[Union[np.ndarray, List]],
                 subplot_row_columns: Optional[List] = None,
                 fig_size: Optional[Tuple] = None, subplot_adjust: Optional[List] = None,
                 cbar_pos: Optional[List] = None, slider_pos: Optional[List] = None,
                 button_pos: Optional[List] = None):
        """
        This is the base class for plots with multiple subplots and with a slider to change the color axis and a button
        to change the normalisation.
        After initialising, the function `change_norm()` should be run to plot normalised images.
        This will change `self.method` from `'raw'` to `'norm'`.

        Args:
            images: `float [n_images]`
                Each image is `n_y x n_x (x n_z)`. This is the normalised image.
                There will be a subplot for each image and if it is 3D, the first z-plane will be set as the
                starting data, `self.im_data` while the full 3d data will be saved as `self.im_data_3d`.
            norm_factor: `float [n_images]`
                `norm_factor[i]` is the value to multiply `images[i]` to give raw image.
                `norm_factor[i]` is either an integer or an array of same dimensions as `image[i]`.
                If a single `norm_factor` given, assume same for each image.
            subplot_row_columns: `[n_rows, n_columns]`
                The subplots will be arranged into `n_rows` and `n_columns`.
                If not given, `n_columns` will be 1.
            fig_size: `[width, height]`
                Size of figure to plot in inches.
                If not given, will be set to `(9, 5)`.
            subplot_adjust: `[left, right, bottom, top]`
                The position of the sides of the subplots in the figure.
                I.e., we don't want subplot to overlap with cbar, slider or buttom and this ensures that.
                If not given, will be set to `[0.07, 0.775, 0.095, 0.94]`.
            cbar_pos: `[left, bottom, width, height]`
                Position of color axis.
                If not given, will be set to `[0.9, 0.15, 0.03, 0.8]`.
            slider_pos: `[left, bottom, width, height]`
                Position of slider that controls color axis.
                If not given, will be set to `[0.85, 0.15, 0.01, 0.8]`.
            button_pos: `[left, bottom, width, height]`
                Position of button which triggers change of normalisation.
                If not given, will be set to `[0.85, 0.02, 0.1, 0.05]`.
        """
        # When bled code of a gene more than this, that particular round/channel will be highlighted in plots
        self.intense_gene_thresh = 0.2
        self.n_images = len(images)
        if subplot_row_columns is None:
            subplot_row_columns = [self.n_images, 1]
        # Default positions
        if fig_size is None:
            fig_size = (9, 5)
        if subplot_adjust is None:
            subplot_adjust = [0.07, 0.775, 0.095, 0.94]
        if cbar_pos is None:
            cbar_pos = [0.9, 0.15, 0.03, 0.8]
        if slider_pos is None:
            self.slider_pos = [0.85, 0.15, 0.01, 0.8]
        else:
            self.slider_pos = slider_pos
        if button_pos is None:
            button_pos = [0.85, 0.02, 0.1, 0.05]
        if not isinstance(norm_factor, list):
            # allow for different norm for each image
            if norm_factor is None:
                self.color_norm = None
            else:
                self.color_norm = [norm_factor, ] * self.n_images
        else:
            self.color_norm = norm_factor
        self.im_data = [val for val in images]  # put in order channels, rounds
        self.method = 'raw' if self.color_norm is not None else 'norm'
        if self.color_norm is None:
            self.caxis_info = {'norm': {}}
        else:
            self.caxis_info = {'norm': {}, 'raw': {}}
        for key in self.caxis_info:
            if key == 'norm':
                im_data = self.im_data
                self.caxis_info[key]['format'] = '%.2f'
            else:
                im_data = [self.im_data[i] * self.color_norm[i] for i in range(self.n_images)]
                self.caxis_info[key]['format'] = '%.0f'
            self.caxis_info[key]['min'] = np.min([im.min() for im in im_data] + [-1e-20])
            self.caxis_info[key]['max'] = np.max([im.max() for im in im_data] + [1e-20])
            self.caxis_info[key]['max'] = np.max([self.caxis_info[key]['max'], -self.caxis_info[key]['min']])
            # have equal either side of zero so small negatives don't look large
            self.caxis_info[key]['min'] = -self.caxis_info[key]['max']
            self.caxis_info[key]['clims'] = [self.caxis_info[key]['min'], self.caxis_info[key]['max']]
            # cmap_norm is so cmap is white at 0.
            self.caxis_info[key]['cmap_norm'] = \
                matplotlib.colors.TwoSlopeNorm(vmin=self.caxis_info[key]['min'],
                                               vcenter=0, vmax=self.caxis_info[key]['max'])

        self.fig, self.ax = plt.subplots(subplot_row_columns[0], subplot_row_columns[1], figsize=fig_size,
                                         sharex=True, sharey=True)
        if self.n_images == 1:
            self.ax = [self.ax]  # need it to be a list
        elif subplot_row_columns[0] > 1 and subplot_row_columns[1] > 1:
            self.ax = self.ax.flatten()  # only have 1 ax index
        oob_axes = np.arange(self.n_images, subplot_row_columns[0] * subplot_row_columns[1])
        if oob_axes.size > 0:
            for i in oob_axes:
                self.fig.delaxes(self.ax[i])  # delete excess subplots
            self.ax = self.ax[:self.n_images]
        self.fig.subplots_adjust(left=subplot_adjust[0], right=subplot_adjust[1], bottom=subplot_adjust[2],
                                 top=subplot_adjust[3])
        self.im = [None] * self.n_images
        if self.im_data[0].ndim == 3:
            # For 3D data, start by showing just the first plane
            self.im_data_3d = self.im_data.copy()
            self.im_data = [val[:, :, 0] for val in self.im_data_3d]
            if self.color_norm is not None:
                self.color_norm_3d = self.color_norm.copy()
                self.color_norm = [val[:, :, 0] for val in self.color_norm_3d]
        else:
            self.im_data_3d = None
            self.color_norm_3d = None
        # initialise plots with a zero array
        for i in range(self.n_images):
            self.im[i] = self.ax[i].imshow(np.zeros(self.im_data[0].shape[:2]), cmap="seismic", aspect='auto',
                                           norm=self.caxis_info[self.method]['cmap_norm'])
        cbar_ax = self.fig.add_axes(cbar_pos)  # left, bottom, width, height
        self.fig.colorbar(self.im[0], cax=cbar_ax)

        self.slider_ax = self.fig.add_axes(self.slider_pos)
        self.color_slider = None
        if self.color_norm is not None:
            self.norm_button_color = 'white'
            self.norm_button_color_press = 'red'
            if self.method == 'raw':
                current_color = self.norm_button_color_press
            else:
                current_color = self.norm_button_color
            self.norm_button_ax = self.fig.add_axes(button_pos)
            self.norm_button = Button(self.norm_button_ax, 'Norm', hovercolor='0.275')
            self.norm_button.label.set_color(current_color)
            self.norm_button.on_clicked(self.change_norm)


    def change_clim(self, val: List):
        """
        Function triggered on change of color axis slider.

        Args:
            val: `[min_caxis, max_caxis]`
                Color axis of plots will be changed to these values.
        """
        if val[0] >= 0:
            # cannot have positive lower bound with diverging colormap
            val[0] = -1e-20
        if val[1] <= 0:
            # cannot have negative upper bound with diverging colormap
            val[1] = 1e-20
        self.caxis_info[self.method]['clims'] = val
        for im in self.im:
            im.set_clim(val[0], val[1])
        self.im[-1].axes.figure.canvas.draw()

    def change_norm(self, event=None):
        """
        Function triggered on press of normalisation button.
        Will either multiply or divide each image by the relevant `color_norm`.
        """
        # need to make new slider at each button press because min/max will change
        self.slider_ax.remove()
        self.slider_ax = self.fig.add_axes(self.slider_pos)
        if self.color_norm is not None:
            self.method = 'norm' if self.method == 'raw' else 'raw'  # change to the other method
            if self.method == 'raw':
                # Change color of text when button pressed
                self.norm_button.label.set_color(self.norm_button_color_press)
            else:
                self.norm_button.label.set_color(self.norm_button_color)
        for i in range(self.n_images):
            # change image to different normalisation and change clim
            self.im[i].set_data(self.im_data[i] * self.color_norm[i] if self.method == 'raw' else self.im_data[i])
            self.im[i].set_norm(self.caxis_info[self.method]['cmap_norm'])
            self.im[i].set_clim(self.caxis_info[self.method]['clims'][0],
                                self.caxis_info[self.method]['clims'][1])

        self.color_slider = RangeSlider(self.slider_ax, "Clim", self.caxis_info[self.method]['min'],
                                        self.caxis_info[self.method]['max'],
                                        self.caxis_info[self.method]['clims'],
                                        orientation='vertical', valfmt=self.caxis_info[self.method]['format'])
        self.color_slider.on_changed(self.change_clim)
        self.im[-1].axes.figure.canvas.draw()


class view_codes(ColorPlotBase):
    def __init__(self, nb: Notebook, spot_no: int, method: str = 'anchor'):
        """
        Diagnostic to compare `spot_color` to `bled_code` of predicted gene.

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
        """
        color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                            nb.basic_info.use_channels)].transpose()
        if method.lower() == 'omp':
            page_name = 'omp'
            config = nb.get_config()['thresholds']
            spot_score = omp_spot_score(nb.omp, config['score_omp_multiplier'], spot_no)
        else:
            page_name = 'ref_spots'
            spot_score = nb.ref_spots.score[spot_no]
        self.spot_color = nb.__getattribute__(page_name).colors[spot_no][
                              np.ix_(nb.basic_info.use_rounds, nb.basic_info.use_channels)].transpose() / color_norm
        # Get spot color after background fitting
        self.background_removed = False
        self.spot_color_pb = fit_background(self.spot_color.T[np.newaxis],
                                            nb.call_spots.background_weight_shift)[0][0].T
        gene_no = nb.__getattribute__(page_name).gene_no[spot_no]

        gene_name = nb.call_spots.gene_names[gene_no]
        gene_color = nb.call_spots.bled_codes_ge[gene_no][np.ix_(nb.basic_info.use_rounds,
                                                                 nb.basic_info.use_channels)].transpose()
        super().__init__([self.spot_color, gene_color], color_norm, slider_pos=[0.85, 0.2, 0.01, 0.75],
                         cbar_pos=[0.9, 0.2, 0.03, 0.75])
        self.ax[0].set_title(f'Spot {spot_no}: match {str(np.around(spot_score, 2))} '
                             f'to {gene_name}')
        self.ax[1].set_title(f'Predicted code for Gene {gene_no}: {gene_name}')
        self.ax[0].set_yticks(ticks=np.arange(self.im_data[0].shape[0]), labels=nb.basic_info.use_channels)
        self.ax[1].set_xticks(ticks=np.arange(self.im_data[0].shape[1]))
        self.ax[1].set_xticklabels(['{:.0f} ({:.2f})'.format(r, nb.call_spots.gene_efficiency[gene_no, r])
                                    for r in nb.basic_info.use_rounds])
        self.ax[1].set_xlabel('Round (Gene Efficiency)')
        self.fig.supylabel('Color Channel')
        intense_gene_cr = np.where(gene_color > self.intense_gene_thresh)
        for i in range(len(intense_gene_cr[0])):
            for j in range(2):
                # can't add rectangle to multiple axes hence second for loop
                rectangle = plt.Rectangle((intense_gene_cr[1][i]-0.5, intense_gene_cr[0][i]-0.5), 1, 1,
                                          fill=False, ec="lime", linestyle=':', lw=2)
                self.ax[j].add_patch(rectangle)

        self.background_button_ax = self.fig.add_axes([0.85, 0.1, 0.1, 0.05])
        self.background_button = Button(self.background_button_ax, 'Background', hovercolor='0.275')
        self.background_button.label.set_color(self.norm_button_color)
        self.background_button.on_clicked(self.change_background)

        self.change_norm()  # initialise with method = 'norm'
        plt.show()

    def change_background(self, event=None):
        """
        Function triggered on press of background button.
        Will either remove/add background contribution to spot_color
        """
        # need to make new slider at each button press because min/max will change
        if not self.background_removed:
            self.im_data[0] = self.spot_color_pb
            self.background_removed = True
            # Change color when pressed
            self.background_button.label.set_color(self.norm_button_color_press)
        else:
            self.im_data[0] = self.spot_color
            self.background_removed = False
            self.background_button.label.set_color(self.norm_button_color)
        # Change norm method before call change_norm so overall it does not change
        if self.color_norm is not None:
            self.method = 'norm' if self.method == 'raw' else 'raw'  # change to the other method
        self.change_norm()


class view_spot(ColorPlotBase):
    def __init__(self, nb: Notebook, spot_no: int, method: str = 'anchor', im_size: int = 8):
        """
        Diagnostic to show intensity of each color channel / round in neighbourhood of spot.
        Will show a grid of `n_use_channels x n_use_rounds` subplots.

        !!! warning "Requires access to `nb.file_names.tile_dir`"

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
            im_size: Radius of image to be plotted for each channel/round.
        """
        color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                            nb.basic_info.use_channels)].transpose()
        if method.lower() == 'omp':
            config = nb.get_config()['thresholds']
            page_name = 'omp'
            spot_score = omp_spot_score(nb.omp, config['score_omp_multiplier'], spot_no)
        else:
            page_name = 'ref_spots'
            spot_score = nb.ref_spots.score[spot_no]
        gene_no = nb.__getattribute__(page_name).gene_no[spot_no]
        t = nb.__getattribute__(page_name).tile[spot_no]
        spot_yxz = nb.__getattribute__(page_name).local_yxz[spot_no]

        gene_name = nb.call_spots.gene_names[gene_no]
        gene_color = nb.call_spots.bled_codes_ge[gene_no][np.ix_(nb.basic_info.use_rounds,
                                                                 nb.basic_info.use_channels)].transpose().flatten()
        n_use_channels, n_use_rounds = color_norm.shape
        color_norm = [val for val in color_norm.flatten()]
        spot_yxz_global = spot_yxz + nb.stitch.tile_origin[t]
        im_size = [im_size, im_size]  # Useful for debugging to have different im_size_y, im_size_x.
        # Subtlety here, may have y-axis flipped, but I think it is correct:
        # note im_yxz[1] refers to point at max_y, min_x+1, z. So when reshape and set plot_extent, should be correct.
        # I.e. im = np.zeros(49); im[1] = 1; im = im.reshape(7,7); plt.imshow(im, extent=[-0.5, 6.5, -0.5, 6.5])
        # will show the value 1 at max_y, min_x+1.
        im_yxz = np.array(np.meshgrid(np.arange(spot_yxz[0]-im_size[0], spot_yxz[0]+im_size[0]+1)[::-1],
                                      np.arange(spot_yxz[1]-im_size[1], spot_yxz[1]+im_size[1]+1), spot_yxz[2]),
                          dtype=np.int16).T.reshape(-1, 3)
        im_diameter = [2*im_size[0]+1, 2*im_size[1]+1]
        spot_colors = get_spot_colors(im_yxz, t, nb.register.transform, nb.file_names, nb.basic_info)
        spot_colors = np.moveaxis(spot_colors, 1, 2)  # put round as the last axis to match color_norm
        spot_colors = spot_colors.reshape(im_yxz.shape[0], -1)
        # reshape
        cr_images = [spot_colors[:, i].reshape(im_diameter[0], im_diameter[1]) / color_norm[i]
                     for i in range(spot_colors.shape[1])]
        subplot_adjust = [0.07, 0.775, 0.075, 0.92]
        super().__init__(cr_images, color_norm, subplot_row_columns=[n_use_channels, n_use_rounds],
                         subplot_adjust=subplot_adjust, fig_size=(13, 8))
        # set x, y coordinates to be those of the global coordinate system
        plot_extent = [im_yxz[:, 1].min()-0.5+nb.stitch.tile_origin[t, 1],
                       im_yxz[:, 1].max()+0.5+nb.stitch.tile_origin[t, 1],
                       im_yxz[:, 0].min()-0.5+nb.stitch.tile_origin[t, 0],
                       im_yxz[:, 0].max()+0.5+nb.stitch.tile_origin[t, 0]]
        for i in range(self.n_images):
            # Add cross-hair
            if gene_color[i] > self.intense_gene_thresh:
                cross_hair_color = 'lime'  # different color if expected large intensity
                linestyle = '--'
                self.ax[i].tick_params(color='lime', labelcolor='lime')
                for spine in self.ax[i].spines.values():
                    spine.set_edgecolor('lime')
            else:
                cross_hair_color = 'k'
                linestyle = ':'
            self.ax[i].axes.plot([spot_yxz_global[1], spot_yxz_global[1]], [plot_extent[2], plot_extent[3]],
                                 cross_hair_color, linestyle=linestyle, lw=1)
            self.ax[i].axes.plot([plot_extent[0], plot_extent[1]], [spot_yxz_global[0], spot_yxz_global[0]],
                                 cross_hair_color, linestyle=linestyle, lw=1)
            self.im[i].set_extent(plot_extent)
            self.ax[i].tick_params(labelbottom=False, labelleft=False)
            # Add axis labels to subplots of far left column or bottom row
            if i % n_use_rounds == 0:
                self.ax[i].set_ylabel(f'{nb.basic_info.use_channels[int(i/n_use_rounds)]}')
            if i >= self.n_images - n_use_rounds:
                r = nb.basic_info.use_rounds[i-(self.n_images - n_use_rounds)]
                self.ax[i].set_xlabel('{:.0f} ({:.2f})'.format(r, nb.call_spots.gene_efficiency[gene_no, r]))


        self.ax[0].set_xticks([spot_yxz_global[1]])
        self.ax[0].set_yticks([spot_yxz_global[0]])
        self.fig.supylabel('Color Channel', size=14)
        self.fig.supxlabel('Round (Gene Efficiency)', size=14, x=(subplot_adjust[0] + subplot_adjust[1]) / 2)
        plt.suptitle(f'Spot {spot_no}: match {str(np.around(spot_score, decimals=2))} '
                     f'to {gene_name}', x=(subplot_adjust[0] + subplot_adjust[1]) / 2, size=16)
        self.change_norm()
        plt.show()


class view_intensity(ColorPlotBase):
    def __init__(self, nb: Notebook, spot_no: int, method: str = 'anchor'):
        """
        Diagnostic to show how intensity is computed from `spot_color`.

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
        """
        color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                            nb.basic_info.use_channels)].transpose()
        if method.lower() == 'omp':
            page_name = 'omp'
            config = nb.get_config()['thresholds']
        else:
            page_name = 'ref_spots'
        intensity_saved = nb.__getattribute__(page_name).intensity[spot_no]
        intensity_thresh = get_intensity_thresh(nb)
        spot_color = nb.__getattribute__(page_name).colors[spot_no][
                         np.ix_(nb.basic_info.use_rounds, nb.basic_info.use_channels)].transpose() / color_norm
        subplot_adjust = [0.07, 0.775, 0.1, 0.91]
        super().__init__([spot_color], color_norm, subplot_adjust=subplot_adjust)
        if intensity_saved > intensity_thresh:
            color = 'w'
        else:
            color = 'r'
        spot_color_symbol = r"$\mathbf{\zeta_s}$"
        intensity_symbol = r"$\chi_s$, (median of $\max_c\zeta_{s_{rc}}$ indicated in green)"
        self.ax[0].set_title(f'Spot Color, {spot_color_symbol}, for spot {spot_no}\n'
                             f'Intensity, {intensity_symbol} = {str(np.around(intensity_saved, 3))}', color=color)
        self.ax[0].set_yticks(ticks=np.arange(self.im_data[0].shape[0]), labels=nb.basic_info.use_channels)
        self.ax[0].set_xticks(ticks=np.arange(self.im_data[0].shape[1]), labels=nb.basic_info.use_rounds)
        self.ax[0].set_xlabel('Round')
        self.fig.supylabel('Color Channel')
        # Highlight max channel in each round which contributes to intensity
        max_channels = np.argmax(self.im_data[0], axis=0)
        for r in range(len(nb.basic_info.use_rounds)):
            # can't add rectangle to multiple axes hence second for loop
            rectangle = plt.Rectangle((r-0.5, max_channels[r]-0.5), 1, 1,
                                      fill=False, ec='lime', linestyle=':', lw=4)
            self.ax[0].add_patch(rectangle)
        self.change_norm()  # initialise with method = 'norm'
        plt.show()
