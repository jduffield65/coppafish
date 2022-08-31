import numpy as np
from matplotlib import pyplot as plt
from ...setup import Notebook
from .spot_colors import ColorPlotBase
plt.style.use('dark_background')


class view_bleed_matrix(ColorPlotBase):
    def __init__(self, nb: Notebook):
        """
        Diagnostic to plot `bleed_matrix`. If `config['call_spots']['bleed_matrix_method']` is `'single'`,
        a single `bleed_matrix` will be plotted. If it is `'separate'`, one will be shown for each round.

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
        """
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
        super().__init__(bleed_matrix, color_norm, subplot_row_columns, subplot_adjust=subplot_adjust,
                         fig_size=fig_size)
        self.ax[0].set_yticks(ticks=np.arange(n_use_channels), labels=nb.basic_info.use_channels)
        if nb.basic_info.dye_names is None:
            self.ax[-1].set_xticks(ticks=np.arange(n_use_dyes), labels=nb.basic_info.use_dyes)
        else:
            self.fig.subplots_adjust(bottom=0.15)
            self.ax[-1].set_xticks(ticks=np.arange(n_use_dyes),
                                   labels=np.asarray(nb.basic_info.dye_names)[nb.basic_info.use_dyes], rotation=45)
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
        """
        Diagnostic to show `bled_codes` with and without `gene_efficiency` applied for all genes.
        Change gene by scrolling with mouse.

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
        """
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
        self.change_gene()  # plot rectangles
        plt.show()

    def change_gene(self, event=None):
        if event is not None:
            if event.button == 'up':
                self.gene_no = (self.gene_no + 1) % self.n_genes
            else:
                self.gene_no = (self.gene_no - 1) % self.n_genes
        self.im_data = [val[:, :, self.gene_no] for val in self.im_data_3d]
        for i in range(self.n_images):
            # change image to different normalisation and change clim
            self.im[i].set_data(self.im_data[i] * self.color_norm[i] if self.method == 'raw' else self.im_data[i])
            intense_gene_cr = np.where(self.im_data[i] > self.intense_gene_thresh)
            [p.remove() for p in reversed(self.ax[i].patches)]  # remove current rectangles
            for j in range(len(intense_gene_cr[0])):
                rectangle = plt.Rectangle((intense_gene_cr[1][j] - 0.5, intense_gene_cr[0][j] - 0.5), 1, 1,
                                          fill=False, ec="g", linestyle=':', lw=2)
                self.ax[i].add_patch(rectangle)
        self.update_title()
        self.im[-1].axes.figure.canvas.draw()

    def update_title(self):
        self.main_title.set_text(f'Gene {self.gene_no}, {self.gene_names[self.gene_no]} Bled Code')
        self.ax[1].set_xticklabels(['{:.0f} ({:.2f})'.format(r, self.gene_efficiency[self.gene_no, r])
                                    for r in self.use_rounds])
