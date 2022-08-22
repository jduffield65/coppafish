import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, CheckButtons
from ..call_spots.score_calc import background_fitting, get_dot_product_score
from ...setup import Notebook
from ...call_spots import omp_spot_score
from typing import Optional, List
import warnings
plt.style.use('dark_background')


class histogram_score:
    ylim_tol = 0.2  # If fractional change in y limit is less than this, then leave it the same
    check_tol = 1e-3  # If saved spot_score and that computed here differs by more than this and check=True, get error.

    def __init__(self, nb: Notebook, method: str = 'omp', score_omp_multiplier: Optional[float] = None,
                 check: bool = False, hist_spacing: float = 0.001, show_plot: bool = True):
        """
        If method is anchor, this will show the histogram of `nb.ref_spots.score` with the option to
        view the histogram of the score computed using various other configurations of `background` fitting
        and `gene_efficiency`. This allows one to see how the these affect the score.

        If `method` is omp, this will show the histogram of omp score, computed with
        `coppafish.call_spots.omp_spot_score`.
        There will also be the option to view the histograms shown for the anchor method.
        I.e. we compute the dot product score for the omp spots.

        Args:
            nb: *Notebook* containing at least `call_spots` page.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
            score_omp_multiplier: Can specify the value of score_omp_multiplier to use to compute omp score.
                If `None`, will use value in config file.
            check: If `True`, and `method='anchor'`, will check that scores computed here match those saved to Notebook.
            hist_spacing: Initial width of bin in histogram.
            show_plot: Whether to run `plt.show()` or not.
        """
        # Add data
        if score_omp_multiplier is None:
            config = nb.get_config()['thresholds']
            score_omp_multiplier = config['score_omp_multiplier']
        self.score_multiplier = score_omp_multiplier
        self.gene_names = nb.call_spots.gene_names
        self.n_genes = self.gene_names.size
        # Use all genes by default
        self.genes_use = np.arange(self.n_genes)

        # For computing score_dp
        spot_colors, spot_colors_pb, background_var = background_fitting(nb, method)
        grc_ind = np.ix_(np.arange(self.n_genes), nb.basic_info.use_rounds, nb.basic_info.use_channels)
        # Bled codes saved to Notebook should already have L2 norm = 1 over used_channels and rounds
        bled_codes = nb.call_spots.bled_codes[grc_ind]
        bled_codes_ge = nb.call_spots.bled_codes_ge[grc_ind]

        # Save score_dp for each permutation of with/without background/gene_efficiency
        self.n_plots = 5
        if method.lower() == 'omp':
            self.n_plots += 1
            self.nbp_omp = nb.omp
            self.gene_no = self.nbp_omp.gene_no
            self.score = np.zeros((self.gene_no.size, self.n_plots), dtype=np.float32)
            self.score[:, -1] = omp_spot_score(self.nbp_omp, self.score_multiplier)
            self.method = 'OMP'
        else:
            self.gene_no = nb.ref_spots.gene_no
            self.score = np.zeros((self.gene_no.size, self.n_plots), dtype=np.float32)
            self.method = 'Anchor'
        self.use = np.isin(self.gene_no, self.genes_use)  # which spots to plot
        # DP score
        self.score[:, 0] = get_dot_product_score(spot_colors_pb, bled_codes_ge, self.gene_no,
                                                 nb.call_spots.dp_norm_shift, background_var)[0]
        if method.lower() != 'omp' and check:
            if np.max(np.abs(self.score[:, 0] - nb.ref_spots.score)) > self.check_tol:
                raise ValueError(f"nb.ref_spots.score differs to that computed here\n"
                                 f"Set check=False to get past this error")

        # DP score no weight
        self.score[:, 1] = get_dot_product_score(spot_colors_pb, bled_codes_ge, self.gene_no,
                                                 nb.call_spots.dp_norm_shift, None)[0]
        # DP score no background
        self.score[:, 2] = get_dot_product_score(spot_colors, bled_codes_ge, self.gene_no,
                                                 nb.call_spots.dp_norm_shift, None)[0]
        # DP score no gene efficiency
        self.score[:, 3] = get_dot_product_score(spot_colors_pb, bled_codes, self.gene_no,
                                                 nb.call_spots.dp_norm_shift, background_var)[0]
        # DP score no background or gene efficiency
        self.score[:, 4] = get_dot_product_score(spot_colors, bled_codes, self.gene_no,
                                                 nb.call_spots.dp_norm_shift, None)[0]

        # Initialise plot
        self.fig, self.ax = plt.subplots(1, 1, figsize=(11, 5))
        self.subplot_adjust = [0.07, 0.85, 0.1, 0.93]
        self.fig.subplots_adjust(left=self.subplot_adjust[0], right=self.subplot_adjust[1],
                                 bottom=self.subplot_adjust[2], top=self.subplot_adjust[3])
        self.ax.set_ylabel(r"Number of Spots")
        if method.lower() == 'omp':
            self.ax.set_xlabel(r"Score, $\gamma_s$ or $\Delta_s$")
        else:
            self.ax.set_xlabel(r"Score, $\Delta_s$")
        self.ax.set_title(f"Distribution of Scores for all {self.method} spots")

        # Set lower bound based on dot product score with no GE/no background as likely to be lowest
        self.hist_min = np.percentile(self.score[:, 4], 0.1)
        # Set upper bound based on dot product score with GE and background because this is likely to be highest
        self.hist_max = np.clip(np.percentile(self.score[:, 0], 99.9), 1, 2)
        self.hist_spacing = hist_spacing
        hist_bins = np.arange(self.hist_min, self.hist_max + self.hist_spacing / 2, self.hist_spacing)
        self.plots = [None] * self.n_plots
        default_colors = plt.rcParams['axes.prop_cycle']._left
        for i in range(self.n_plots):
            y, x = np.histogram(self.score[self.use, i], hist_bins)
            x = x[:-1] + self.hist_spacing / 2  # so same length as x
            self.plots[i], = self.ax.plot(x, y, color=default_colors[i]['color'])
            if method.lower() == 'omp' and i < self.n_plots - 1:
                self.plots[i].set_visible(False)
            elif i > 0 and method.lower() != 'omp':
                self.plots[i].set_visible(False)

        self.ax.set_xlim(self.hist_min, self.hist_max)
        self.ax.set_ylim(0, None)

        # Add text box to change score multiplier
        text_box_labels = ['Gene', 'Histogram\nSpacing', 'Score\n' + r'Multiplier, $\rho$']
        text_box_values = ['all', self.hist_spacing, np.around(self.score_multiplier, 2)]
        text_box_funcs = [self.update_genes, self.update_hist_spacing, self.update_score_multiplier]
        if method.lower() != 'omp':
            text_box_labels = text_box_labels[:2]
            text_box_values = text_box_values[:2]
            text_box_funcs = text_box_funcs[:2]
        self.text_boxes = [None] * len(text_box_labels)
        for i in range(len(text_box_labels)):
            text_ax = self.fig.add_axes([self.subplot_adjust[1] + 0.05,
                                         self.subplot_adjust[2] + 0.15 * (len(text_box_labels) - i - 1), 0.05, 0.04])
            self.text_boxes[i] = TextBox(text_ax, text_box_labels[i], text_box_values[i], color='k',
                                         hovercolor=[0.2, 0.2, 0.2])
            self.text_boxes[i].cursor.set_color('r')
            label = text_ax.get_children()[0]  # label is a child of the TextBox axis
            if i == 0:
                label.set_position([0.5, 1.77])  # [x,y] - change here to set the position
            else:
                label.set_position([0.5, 2.75])
                # centering the text
            label.set_verticalalignment('top')
            label.set_horizontalalignment('center')
            self.text_boxes[i].on_submit(text_box_funcs[i])

        # Add buttons to add/remove score_dp histograms
        self.buttons_ax = self.fig.add_axes([self.subplot_adjust[1] + 0.02, self.subplot_adjust[3] - 0.45, 0.15, 0.5])
        plt.axis('off')
        self.button_labels = [r"$\Delta_s$" + "\nDot Product Score",
                              r"$\Delta_s$" + "\nNo Weighting",
                              r"$\Delta_s$" + "\nNo Background",
                              r"$\Delta_s$" + "\nNo Gene Efficiency",
                              r"$\Delta_s$" + "\nNo Background\nNo Gene Efficiency"]
        label_checked = [True, False, False, False, False]
        if method.lower() == 'omp':
            self.button_labels += [r"$\gamma_s$" + "\nOMP Score"]
            label_checked += [True]
            label_checked[0] = False
        self.buttons = CheckButtons(self.buttons_ax, self.button_labels, label_checked)

        for i in range(self.n_plots):
            self.buttons.labels[i].set_fontsize(7)
            self.buttons.labels[i].set_color(default_colors[i]['color'])
            self.buttons.rectangles[i].set_color('w')
        self.buttons.on_clicked(self.choose_plots)
        if show_plot:
            plt.show()

    def update(self, inds_update: Optional[List[int]] = None):
        ylim_old = self.ax.get_ylim()[1]  # To check whether we need to change y limit
        hist_bins = np.arange(self.hist_min, self.hist_max + self.hist_spacing / 2, self.hist_spacing)
        if inds_update is None:
            inds_update = np.arange(len(self.plots))  # By default update all plots
        ylim_new = 0
        for i in np.arange(len(self.plots)):
            if i in inds_update:
                y, x = np.histogram(self.score[self.use, i], hist_bins)
                x = x[:-1] + self.hist_spacing / 2  # so same length as x
                self.plots[i].set_data(x, y)
            ylim_new = np.max([ylim_new, self.plots[i].get_ydata().max()])
        # self.ax.set_xlim(0, 1)
        if np.abs(ylim_new - ylim_old) / np.max([ylim_new, ylim_old]) < self.ylim_tol:
            ylim_new = ylim_old
        self.ax.set_ylim(0, ylim_new)
        if isinstance(self.genes_use, int):
            gene_label = f" matched to {self.gene_names[self.genes_use]}"
        else:
            gene_label = ""
        self.ax.set_title(f"Distribution of Scores for all {self.method} spots" + gene_label)
        self.ax.figure.canvas.draw()

    def update_genes(self, text):
        # TODO: If give 2+45 then show distribution for gene 2 and 45 on the same plot
        # Can select to view histogram of one gene or all genes
        if text.lower() == 'all':
            g = 'all'
        else:
            try:
                g = int(text)
                if g >= self.n_genes or g < 0:
                    warnings.warn(f'\nGene index needs to be between 0 and {self.n_genes}')
                    g = self.genes_use
            except (ValueError, TypeError):
                # if a string, check if is name of gene
                gene_names = list(map(str.lower, self.gene_names))
                try:
                    g = gene_names.index(text.lower())
                except ValueError:
                    # default to the best gene at this iteration
                    warnings.warn(f"\nGene given, {text}, is not valid")
                    if isinstance(self.genes_use, int):
                        g = self.genes_use
                    else:
                        g = 'all'
        if g == 'all':
            self.genes_use = np.arange(self.n_genes)
        else:
            self.genes_use = g
        self.use = np.isin(self.gene_no, self.genes_use)  # which spots to plot
        self.text_boxes[0].set_val(g)
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
        self.text_boxes[1].set_val(hist_spacing)
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
        self.score[:, self.n_plots - 1] = omp_spot_score(self.nbp_omp, self.score_multiplier)
        self.text_boxes[2].set_val(np.around(score_multiplier, 2))
        self.update(inds_update=[self.n_plots - 1])

    def choose_plots(self, label):
        index = self.button_labels.index(label)
        self.plots[index].set_visible(not self.plots[index].get_visible())
        self.ax.figure.canvas.draw()


class histogram_2d_score(histogram_score):
    def __init__(self, nb: Notebook, score_omp_multiplier: Optional[float] = None):
        """
        This plots the bivariate histogram to see the correlation between the omp spot score, $\gamma_s$ and
        the dot product score $\Delta_s$.

        Args:
            nb: *Notebook* containing at least `call_spots` page.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
        """
        # large hist_spacing so quick as we change it anway
        super().__init__(nb, 'omp', score_omp_multiplier, False, 0.5, False)
        self.ax.clear()
        # Get rid of buttons - only use actual dot product score
        self.buttons_ax.clear()
        plt.axis('off')
        self.score = self.score[:, [0, self.n_plots - 1]]
        self.n_plots = 2
        del self.plots
        hist_bins = np.arange(self.hist_min, self.hist_max + self.hist_spacing / 2, self.hist_spacing)
        self.x_score_ind = 0
        self.hist_spacing = 0.01
        self.plot = self.ax.hist2d(self.score[:, self.x_score_ind], self.score[:, -1], hist_bins)[3]
        self.cbar = self.fig.colorbar(self.plot, ax=self.ax)
        self.ax.set_xlim(self.hist_min, self.hist_max)
        self.ax.set_ylim(self.hist_min, self.hist_max)
        self.text_boxes[1].set_val(self.hist_spacing)
        self.ax.set_xlabel(self.button_labels[0].replace('\n', ', '))
        self.ax.set_ylabel(self.button_labels[-1].replace('\n', ', '))
        plt.show()

    def update(self, inds_update: Optional[List[int]] = None):
        ylim_old = self.plot.get_clim()[1]  # To check whether we need to change y limit
        hist_bins = np.arange(self.hist_min, self.hist_max + self.hist_spacing / 2, self.hist_spacing)
        self.plot = self.ax.hist2d(self.score[self.use, self.x_score_ind], self.score[self.use, -1], hist_bins)[3]
        ylim_new = self.plot.get_clim()[1]
        self.ax.set_xlim(self.hist_min, self.hist_max)
        self.ax.set_ylim(self.hist_min, self.hist_max)
        if np.abs(ylim_new - ylim_old) / np.max([ylim_new, ylim_old]) < self.ylim_tol:
            ylim_new = ylim_old
        self.plot.set_clim(0, ylim_new)
        self.cbar.update_normal(self.plot)
        if isinstance(self.genes_use, int):
            gene_label = f" matched to {self.gene_names[self.genes_use]}"
        else:
            gene_label = ""
        self.ax.set_title(f"Distribution of Scores for all {self.method} spots" + gene_label)
        self.ax.figure.canvas.draw()
