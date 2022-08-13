import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from ...setup import Notebook
from ...call_spots import omp_spot_score
from typing import Optional, Union, List
import warnings


class histogram_omp_score:
    ylim_tol = 0.2  # If fractional change in y limit is less than this, then leave it the same
    def __init__(self, nb: Notebook, score_multiplier: Optional[float] = None):
        # Add data
        self.nbp_omp = nb.omp
        if score_multiplier is None:
            config = nb.get_config()['thresholds']
            score_multiplier = config['score_omp_multiplier']
        self.score_multiplier = score_multiplier
        self.gene_names = nb.call_spots.gene_names
        self.n_genes = self.gene_names.size
        # Use all genes by default
        self.genes_use = np.arange(self.n_genes)
        self.gene_no = nb.omp.gene_no
        self.use = np.isin(nb.omp.gene_no, self.genes_use)  # which spots to plot

        # Initialise plot
        self.fig, self.ax = plt.subplots(1, 1, figsize=(11, 5))
        self.subplot_adjust = [0.07, 0.9, 0.1, 0.93]
        self.fig.subplots_adjust(left=self.subplot_adjust[0], right=self.subplot_adjust[1],
                                 bottom=self.subplot_adjust[2], top=self.subplot_adjust[3])

        self.hist_spacing = 0.001
        hist_bins = np.arange(0, 1+self.hist_spacing/2, self.hist_spacing)
        score = omp_spot_score(self.nbp_omp, self.score_multiplier, self.use)
        y, x = np.histogram(score, hist_bins)
        x = x[:-1] + self.hist_spacing/2  # so same length as x
        self.ax.plot(x, y)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, None)
        self.ax.set_ylabel(r"Number of Spots")
        self.ax.set_xlabel(r"Score, $\gamma_s$")
        self.ax.set_title(r"Distribution of OMP Score, $\gamma_s$, for All Genes")

        # Add text box to change score multiplier
        text_box_labels = ['Gene', 'Score\n'+r'Multiplier, $\rho$', 'Histogram\nSpacing']
        text_box_values = ['all', np.around(self.score_multiplier, 2), self.hist_spacing]
        text_box_funcs = [self.update_genes, self.update_score_multiplier, self.update_hist_spacing]
        self.text_boxes = [None] * len(text_box_labels)
        for i in range(len(text_box_labels)):
            text_ax = self.fig.add_axes([self.subplot_adjust[1] + 0.02, self.subplot_adjust[3] - 0.2 * (i + 1),
                                         0.05, 0.04])
            self.text_boxes[i] = TextBox(text_ax, text_box_labels[i], text_box_values[i], color='k',
                                         hovercolor=[0.2, 0.2, 0.2])
            self.text_boxes[i].cursor.set_color('r')
            label = text_ax.get_children()[0]  # label is a child of the TextBox axis
            label.set_position([0.5, 2.75])  # [x,y] - change here to set the position
            # centering the text
            label.set_verticalalignment('top')
            label.set_horizontalalignment('center')
            self.text_boxes[i].on_submit(text_box_funcs[i])

        plt.show(block=True)

    def update(self):
        ylim_old = self.ax.get_ylim()[1]  # To check whether we need to change y limit
        self.ax.clear()
        score = omp_spot_score(self.nbp_omp, self.score_multiplier, self.use)
        hist_bins = np.arange(0, 1 + self.hist_spacing / 2, self.hist_spacing)
        y, x = np.histogram(score, hist_bins)
        x = x[:-1] + self.hist_spacing/2  # so same length as x
        self.ax.plot(x, y)
        self.ax.set_xlim(0, 1)
        ylim_new = self.ax.get_ylim()[1]
        if np.abs(ylim_new - ylim_old) / np.max([ylim_new, ylim_old]) < self.ylim_tol:
            ylim_new = ylim_old
        self.ax.set_ylim(0, ylim_new)
        self.ax.set_ylabel(r"Number of Spots")
        self.ax.set_xlabel(r"Score, $\gamma_s$")
        if isinstance(self.genes_use, int):
            gene_label = self.gene_names[self.genes_use]
        else:
            gene_label = "All Genes"
        self.ax.set_title(r"Distribution of $\gamma_s$ for "+gene_label)
        self.ax.figure.canvas.draw()

    def update_genes(self, text):
        # Can select to view histogram of one gene or all genes
        if text.lower() == 'all':
            g = 'all'
        else:
            try:
                g = int(text)
                if g >= self.n_genes or g < 0:
                    warnings.warn(f'\nGene index needs to be between 0 and {self.n_genes_all}')
                    g = self.genes_use
            except (ValueError, TypeError):
                # if a string, check if is name of gene
                gene_names = list(map(str.lower, self.gene_names))
                try:
                    g = gene_names.index(text.lower())
                except ValueError:
                    # default to the best gene at this iteration
                    warnings.warn(f"\nGene given, {text}, is not valid")
                    g = self.genes_use
        if g == 'all':
            self.genes_use = np.arange(self.n_genes)
        else:
            self.genes_use = g
        self.use = np.isin(self.gene_no, self.genes_use)  # which spots to plot
        self.text_boxes[0].set_val(g)
        self.update()

    def update_score_multiplier(self, text):
        # Can see how changing score_multiplier affects distribution
        try:
            score_multiplier = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nScore multiplier given, {text}, is not valid")
            score_multiplier = self.score_multiplier
        if score_multiplier < 0:
            warnings.warn("Score multiplier cannot be negative")
            score_multiplier = self.score_multiplier
        self.score_multiplier = score_multiplier
        self.text_boxes[1].set_val(np.around(score_multiplier, 2))
        self.update()

    def update_hist_spacing(self, text):
        # Can select spacing of histogram
        try:
            hist_spacing = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nScore multiplier given, {text}, is not valid")
            hist_spacing = self.hist_spacing
        if hist_spacing < 0:
            warnings.warn("Histogram spacing cannot be negative")
            hist_spacing = self.hist_spacing
        if hist_spacing < 0:
            warnings.warn("Histogram spacing cannot be negative")
            hist_spacing = self.hist_spacing
        if hist_spacing >= 1:
            warnings.warn("Histogram spacing cannot be >= 1")
            hist_spacing = self.hist_spacing
        self.hist_spacing = hist_spacing
        self.text_boxes[2].set_val(hist_spacing)
        self.update()
