import os

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


class ViewBleedCalc:
    def __init__(self, nb: Notebook):
        self.nb = nb
        color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds, nb.basic_info.use_channels)]
        color_norm = np.repeat(color_norm[np.newaxis, :, :], np.sum(nb.ref_spots.isolated), axis=0)
        self.isolated_spots = nb.ref_spots.colors[nb.ref_spots.isolated][:, :, nb.basic_info.use_channels] / color_norm
        # Get current working directory and load default bleed matrix
        self.default_bleed = np.load(os.path.join(os.getcwd(), 'coppafish/setup/default_bleed.npy'))[nb.basic_info.use_channels]
        # Swap columns 2 and 3 in default bleed and dye names
        self.default_bleed[:, [2, 3]] = self.default_bleed[:, [3, 2]]
        self.dye_names = self.nb.basic_info.dye_names
        self.dye_names[2], self.dye_names[3] = self.dye_names[3], self.dye_names[2]
        # Normalise each column of default_bleed to have L2 norm of 1
        self.dye_template = self.default_bleed / np.linalg.norm(self.default_bleed, axis=0)
        # Now we are going to loop through all isolated spots, convert these to n_rounds colour vectors and then
        # assign each colour vector to a dye
        self.colour_vectors = self.isolated_spots.reshape((self.isolated_spots.shape[0] * self.nb.basic_info.n_rounds,
                                                           len(self.nb.basic_info.use_channels)))
        self.all_dye_score = np.zeros((self.colour_vectors.shape[0], self.dye_template.shape[1]))
        self.dye_score = np.zeros(self.colour_vectors.shape[0])
        self.dye_assignment = np.zeros(self.colour_vectors.shape[0], dtype=int)
        for i in range(self.colour_vectors.shape[0]):
            # Assign each vector to dye which maximises dot product with vector
            self.all_dye_score[i] = self.colour_vectors[i] @ self.dye_template
            self.dye_assignment[i] = np.argmax(self.all_dye_score[i])
            self.dye_score[i] = np.max(self.all_dye_score[i])

        # Now we have assigned each colour vector to a dye, we can view all colour vectors assigned to each dye
        max_intensity = np.max(self.colour_vectors)
        max_score = np.max(self.dye_score)
        fig, ax = plt.subplots(2, self.dye_template.shape[1], figsize=(10, 10))
        for i in range(self.dye_template.shape[1]):
            # Plot the colour vectors assigned to each dye
            dye_vectors = self.colour_vectors[self.dye_assignment == i]
            # Order these vectors by dye score in descending order
            dye_vectors = dye_vectors[np.argsort(self.dye_score[self.dye_assignment == i])[::-1]]
            ax[0, i].imshow(dye_vectors, vmin=0, vmax=max_intensity/2, aspect='auto', interpolation='none')
            ax[0, i].set_title(self.nb.basic_info.dye_names[i])
            ax[0, i].set_yticks([])
            ax[0, i].set_xticks([])

            # Add a horizontal red line at a dye score of 1. To do this we must first find the index of the dye
            # vector with score closest to 1
            first_index = np.argmin(np.abs(self.dye_score[self.dye_assignment == i] - 1))
            ax[0, i].axhline(first_index, color='r', linestyle='--')

            # Plot a histogram of the dye scores
            ax[1, i].hist(self.dye_score[self.dye_assignment == i], bins=np.linspace(0, max_score / 2, 200))
            ax[1, i].set_title(self.dye_names[i])
            ax[1, i].set_xlabel('Dye Score')
            ax[1, i].set_ylabel('Frequency')
            ax[1, i].set_yticks([])
            mean = np.mean(self.dye_score[self.dye_assignment == i])
            ax[1, i].axvline(mean, color='r')
            # Add label in top right corner of each plot with median dye score
            ax[1, i].text(0.95, 0.95, '{:.2f}'.format(mean), color='r',
                          horizontalalignment='right', verticalalignment='top', transform=ax[1, i].transAxes)

        # Add a single colour bar for all plots on the right. Label this as spot intensity
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ax[0, 0].images[0], cax=cbar_ax)
        cbar_ax.set_ylabel('Spot Intensity (normalised)')

        # Add super title
        fig.suptitle('Dye Assignment', fontsize=16)

        plt.show()
