import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RangeSlider
from ..call_spots import omp_spot_score
from ..setup import Notebook
from ..spot_colors import get_spot_colors
import matplotlib
from typing import List, Optional, Tuple, Union

matplotlib.use('qtagg')


class ColorPlotBase:
    def __init__(self, images: List, norm_factor: Optional[Union[np.ndarray, List]],
                 subplot_row_columns: Optional[List] = None,
                 fig_size: Optional[Tuple] = None, subplot_adjust: Optional[List] = None,
                 cbar_pos: Optional[List] = None, slider_pos: Optional[List] = None,
                 button_pos: Optional[List] = None):
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
            self.color_norm = [norm_factor, ] * self.n_images
        else:
            self.color_norm = norm_factor
        self.im_data = [val for val in images]  # put in order channels, rounds
        self.method = 'raw'
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
        self.im = [None, ] * self.n_images
        if self.im_data[0].ndim == 3:
            # For 3D data, start by showing just the first plane
            self.im_data_3d = self.im_data.copy()
            self.im_data = [val[:, :, 0] for val in self.im_data_3d]
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
        self.norm_button_ax = self.fig.add_axes(button_pos)
        self.norm_button = Button(self.norm_button_ax, 'Norm', hovercolor='0.275')
        self.norm_button.on_clicked(self.change_norm)

    def change_clim(self, val):
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
        # need to make new slider at each button press because min/max will change
        self.slider_ax.remove()
        self.slider_ax = self.fig.add_axes(self.slider_pos)
        self.method = 'norm' if self.method == 'raw' else 'raw'  # change to the other method

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
    def __init__(self, nb: Notebook, spot_no: int):
        color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                            nb.basic_info.use_channels)].transpose()
        spot_color = nb.omp.colors[spot_no][np.ix_(nb.basic_info.use_rounds,
                                                   nb.basic_info.use_channels)].transpose() / color_norm
        gene_no = nb.omp.gene_no[spot_no]
        gene_name = nb.call_spots.gene_names[gene_no]
        gene_color = nb.call_spots.bled_codes_ge[gene_no][np.ix_(nb.basic_info.use_rounds,
                                                                 nb.basic_info.use_channels)].transpose()
        super().__init__([spot_color, gene_color], color_norm)
        self.ax[0].set_title(f'Spot {spot_no}: match {np.round(omp_spot_score(nb.omp, spot_no), decimals=2)} '
                             f'to {gene_name}')
        self.ax[1].set_title(f'Predicted code for {gene_name}')
        self.ax[0].set_yticks(ticks=np.arange(self.im_data[0].shape[0]), labels=nb.basic_info.use_channels)
        self.ax[1].set_xticks(ticks=np.arange(self.im_data[0].shape[1]), labels=nb.basic_info.use_rounds)
        self.ax[1].set_xlabel('Round')
        self.fig.supylabel('Color Channel')
        self.change_norm()  # initialise with method = 'norm'
        plt.show()


class view_bleed_matrix(ColorPlotBase):
    def __init__(self, nb: Notebook):
        color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                            nb.basic_info.use_channels)]
        n_use_rounds, n_use_channels = color_norm.shape
        single_bm = (color_norm == color_norm[0]).all()
        if single_bm:
            bleed_matrix = [nb.call_spots.bleed_matrix[0][np.ix_(nb.basic_info.use_channels,
                                                                 nb.basic_info.use_dyes)]]
            subplot_row_columns = [1, 1]
            subplot_adjust = [0.07, 0.775, 0.095, 0.94]
            fig_size = (9, 5)
        else:
            bleed_matrix = [nb.call_spots.bleed_matrix[r][np.ix_(nb.basic_info.use_channels,
                                                                 nb.basic_info.use_dyes)]
                            for r in range(n_use_rounds)]
            if n_use_rounds <= 3:
                subplot_row_columns = [n_use_rounds, 1]
            else:
                n_cols = int(np.ceil(n_use_rounds / 4))  # at most 4 rows
                subplot_row_columns = [int(np.ceil(n_use_rounds / n_cols)), n_cols]
            subplot_adjust = [0.07, 0.775, 0.095, 0.92]
            fig_size = (12, 7)
        n_use_dyes = bleed_matrix[0].shape[1]
        # different norm for each round, each has dims n_use_channels x 1 whereas BM dims is n_use_channels x n_dyes
        # i.e. normalisation just affected by channel not by dye.
        color_norm = [np.expand_dims(color_norm[r], 1) for r in range(n_use_rounds)]
        if nb.basic_info.dye_names is not None:
            subplot_adjust[2] = 0.14  # need more space at bottom for dye labels
        super().__init__(bleed_matrix, color_norm, subplot_row_columns, subplot_adjust=subplot_adjust,
                         fig_size=fig_size)
        self.ax[0].set_yticks(ticks=np.arange(n_use_channels), labels=nb.basic_info.use_channels)
        if nb.basic_info.dye_names is None:
            self.ax[-1].set_xticks(ticks=np.arange(n_use_dyes), labels=nb.basic_info.use_dyes)
        else:
            self.ax[-1].set_xticks(ticks=np.arange(n_use_dyes), labels=nb.basic_info.dye_names[nb.basic_info.use_dyes],
                                   rotation=45)
        if single_bm:
            self.ax[0].set_title('Bleed Matrix')
            self.ax[0].set_ylabel('Color Channel')
            self.ax[0].set_xlabel('Dyes')
        else:
            for i in range(n_use_rounds):
                self.ax[i].set_title(f'Round {nb.basic_info.use_rounds[i]}', size=8)
                plt.suptitle("Bleed Matrices", size=12, x=(subplot_adjust[0] + subplot_adjust[1]) / 2)
            self.fig.supylabel('Color Channel', size=12)
            self.fig.supxlabel('Dyes', size=12, x=(subplot_adjust[0] + subplot_adjust[1]) / 2)
        self.change_norm()  # initialise with method = 'norm'
        plt.show()


class view_bled_codes(ColorPlotBase):
    def __init__(self, nb: Notebook):
        color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                            nb.basic_info.use_channels)].transpose()[:, :, np.newaxis]
        self.n_genes = nb.call_spots.bled_codes_ge.shape[0]
        self.gene_names = nb.call_spots.gene_names
        self.gene_efficiency = nb.call_spots.gene_efficiency
        self.use_rounds = nb.basic_info.use_rounds
        bled_codes_ge = nb.call_spots.bled_codes_ge[np.ix_(np.arange(self.n_genes), nb.basic_info.use_rounds,
                                                           nb.basic_info.use_channels)]
        bled_codes = nb.call_spots.bled_codes[np.ix_(np.arange(self.n_genes), nb.basic_info.use_rounds,
                                                          nb.basic_info.use_channels)]
        bled_codes_ge = np.moveaxis(bled_codes_ge, 0, -1)  # move gene index to last so scroll over
        bled_codes = np.moveaxis(bled_codes, 0, -1)
        bled_codes_ge = np.moveaxis(bled_codes_ge, 0, 1)  # move channel index to first for plotting
        bled_codes = np.moveaxis(bled_codes, 0, 1)
        subplot_adjust = [0.07, 0.775, 0.095, 0.9]  # larger top adjust for super title
        super().__init__([bled_codes_ge, bled_codes], color_norm, subplot_adjust=subplot_adjust)
        self.gene_no = 0
        self.ax[0].set_title('With Gene Efficiency', size=10)
        self.ax[1].set_title('Without Gene Efficiency', size=10)
        self.ax[0].set_yticks(ticks=np.arange(self.im_data[0].shape[0]), labels=nb.basic_info.use_channels)
        self.ax[1].set_xticks(ticks=np.arange(self.im_data[0].shape[1]))
        self.ax[1].set_xlabel('Round (Gene Efficiency)')
        self.fig.supylabel('Color Channel')
        self.main_title = plt.suptitle('', x=(subplot_adjust[0] + subplot_adjust[1]) / 2)
        self.update_title()
        self.fig.canvas.mpl_connect('scroll_event', self.change_gene)
        self.change_norm()
        plt.show()

    def change_gene(self, event):
        if event.button == 'up':
            self.gene_no = (self.gene_no + 1) % self.n_genes
        else:
            self.gene_no = (self.gene_no - 1) % self.n_genes
        self.im_data = [val[:, :, self.gene_no] for val in self.im_data_3d]
        for i in range(self.n_images):
            # change image to different normalisation and change clim
            self.im[i].set_data(self.im_data[i] * self.color_norm[i] if self.method == 'raw' else self.im_data[i])
        self.update_title()
        self.im[-1].axes.figure.canvas.draw()

    def update_title(self):
        self.main_title.set_text(f'Gene {self.gene_no}, {self.gene_names[self.gene_no]} Bled Code')
        self.ax[1].set_xticklabels(['{:.0f} ({:.2f})'.format(r, self.gene_efficiency[self.gene_no, r])
                                    for r in self.use_rounds])


class view_spot(ColorPlotBase):
    def __init__(self, nb: Notebook, spot_no: int, im_size: int = 8):
        gene_no = nb.omp.gene_no[spot_no]
        gene_name = nb.call_spots.gene_names[gene_no]
        color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                            nb.basic_info.use_channels)].transpose()
        n_use_channels, n_use_rounds = color_norm.shape
        color_norm = [val for val in color_norm.flatten()]
        t = nb.omp.tile[spot_no]
        spot_yxz = nb.omp.local_yxz[spot_no]
        spot_yxz_global = spot_yxz + nb.stitch.tile_origin[t]
        im_size = [im_size, im_size]  # Useful for debugging to have different im_size_y, im_size_x.
        # note im_yxz[1] refers to point at min_y, min_x+1, z. So when reshape, should be correct.
        im_yxz = np.array(np.meshgrid(np.arange(spot_yxz[0]-im_size[0], spot_yxz[0]+im_size[0]+1),
                                      np.arange(spot_yxz[1]-im_size[1], spot_yxz[1]+im_size[1]+1), spot_yxz[2]),
                          dtype=np.int16).T.reshape(-1, 3)
        im_diameter = [2*im_size[0]+1, 2*im_size[1]+1]
        spot_colors = get_spot_colors(im_yxz, t, nb.register.transform, nb.file_names, nb.basic_info)
        spot_colors = spot_colors[np.ix_(np.arange(im_yxz.shape[0]),
                                         nb.basic_info.use_rounds, nb.basic_info.use_channels)]
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
            self.ax[i].axes.plot([spot_yxz_global[1], spot_yxz_global[1]], [plot_extent[2], plot_extent[3]],
                                 'k', linestyle=":", lw=1)
            self.ax[i].axes.plot([plot_extent[0], plot_extent[1]], [spot_yxz_global[0], spot_yxz_global[0]],
                                 'k', linestyle=":", lw=1)
            self.im[i].set_extent(plot_extent)
            self.ax[i].tick_params(labelbottom=False, labelleft=False)
            if i % n_use_rounds == 0:
                self.ax[i].set_ylabel(f'{nb.basic_info.use_channels[int(i/n_use_rounds)]}')
            if i >= self.n_images - n_use_rounds:
                self.ax[i].set_xlabel(f'{nb.basic_info.use_rounds[i-(self.n_images - n_use_rounds)]}')


        self.ax[0].set_xticks([spot_yxz_global[1]])
        self.ax[0].set_yticks([spot_yxz_global[0]])
        self.fig.supylabel('Color Channel', size=14)
        self.fig.supxlabel('Round', size=14, x=(subplot_adjust[0] + subplot_adjust[1]) / 2)
        plt.suptitle(f'Spot {spot_no}: match {np.round(omp_spot_score(nb.omp, spot_no), decimals=2)} '
                     f'to {gene_name}', x=(subplot_adjust[0] + subplot_adjust[1]) / 2, size=16)
        self.change_norm()
        plt.show()
