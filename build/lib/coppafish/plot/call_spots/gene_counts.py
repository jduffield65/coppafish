import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, CheckButtons
from .score_calc import background_fitting, get_dot_product_score
from ...setup import Notebook
from ...call_spots import omp_spot_score
from typing import Optional, List
import warnings
plt.style.use('dark_background')


class gene_counts:
    ylim_tol = 0.2  # If fractional change in y limit is less than this, then leave it the same
    check_tol = 1e-3  # If saved spot_score and that computed here differs by more than this and check=True, get error.

    def __init__(self, nb: Notebook, fake_bled_codes: Optional[np.ndarray] = None,
                 fake_gene_names: Optional[List[str]] = None,
                 score_thresh: Optional[float] = None, intensity_thresh: Optional[float] = None,
                 score_omp_thresh: Optional[float] = None, score_omp_multiplier: Optional[float] = None):
        """
        This shows the number of reference spots assigned to each gene which pass the quality thresholding based on
        the parameters `score_thresh` and `intensity_thresh`.

        If `nb` has the *OMP* page, then the number of omp spots will also be shown, where the quality thresholding is
        based on `score_omp_thresh`, `score_omp_multiplier` and `intensity_thresh`.

        There will also be a second reference spots histogram, the difference with this is that the spots
        were allowed to be assigned to some fake genes with `bled_codes` specified through `fake_bled_codes`.

        !!! note
            `fake_bled_codes` have dimension `n_fake x  nbp_basic.n_rounds x nbp_basic.n_channels` not
            `n_fake x len(nbp_basic.use_rounds) x len(nbp_basic.use_channels)`.

        Args:
            nb: *Notebook* containing at least `call_spots` page.
            fake_bled_codes: `float [n_fake_genes x n_rounds x n_channels]`.
                colors of fake genes to find dot product with
                Will find new gene assignment of anchor spots to new set of `bled_codes` which include these in
                addition to `bled_codes_ge`.
                By default, will have a fake gene for each round and channel such that it is $1$ in round r,
                channel $c$ and 0 everywhere else.
            fake_gene_names: `str [n_fake_genes]`.
                Can give name of each fake gene. If `None`, fake gene $i$ will be called FAKE:$i$.
            score_thresh: Threshold for score for ref_spots. Can be changed with text box.
            intensity_thresh: Threshold for intensity. Can be changed with text box.
            score_omp_thresh: Threshold for score for omp_spots. Can be changed with text box.
            score_omp_multiplier: Can specify the value of score_omp_multiplier to use to compute omp score.
                If `None`, will use value in config file. Can be changed with text box.
        """
        # Add fake genes
        if fake_bled_codes is None:
            # Default have binary fake gene for each used round and channel
            n_fake = len(nb.basic_info.use_rounds) * len(nb.basic_info.use_channels)
            fake_bled_codes = np.zeros((n_fake, nb.basic_info.n_rounds, nb.basic_info.n_channels))
            i = 0
            # Cluster fake genes by channel because more likely to be a faulty channel
            for c in nb.basic_info.use_channels:
                for r in nb.basic_info.use_rounds:
                    fake_bled_codes[i, r, c] = 1
                    i += 1
            fake_gene_names = [f'r{r}c{c}' for c in nb.basic_info.use_channels for r in nb.basic_info.use_rounds]
        n_fake = fake_bled_codes.shape[0]
        if fake_gene_names is None:
            fake_gene_names = [f'FAKE:{i}' for i in range(n_fake)]
        self.gene_names = nb.call_spots.gene_names.tolist() + fake_gene_names
        self.n_genes = len(self.gene_names)
        self.n_genes_real = self.n_genes - n_fake

        # Do new gene assignment for anchor spots when fake genes are included
        spot_colors_pb, background_var = background_fitting(nb, 'anchor')[1:]  # Fit background before dot product score
        grc_ind = np.ix_(np.arange(self.n_genes), nb.basic_info.use_rounds, nb.basic_info.use_channels)
        # Bled codes saved to Notebook should already have L2 norm = 1 over used_channels and rounds
        bled_codes = np.append(nb.call_spots.bled_codes_ge, fake_bled_codes, axis=0)
        bled_codes = bled_codes[grc_ind]
        # Ensure L2 norm of 1 for each gene
        norm_factor = np.expand_dims(np.linalg.norm(bled_codes, axis=(1, 2)), (1, 2))
        norm_factor[norm_factor == 0] = 1  # For genes with no dye in use_dye, this avoids blow up on next line
        bled_codes = bled_codes / norm_factor
        score, gene_no = get_dot_product_score(spot_colors_pb, bled_codes, None, nb.call_spots.dp_norm_shift,
                                               background_var)

        # Add quality thresholding info and gene assigned to for method with no fake genes and method with fake genes
        self.intensity = [nb.ref_spots.intensity.astype(np.float16)] * 2
        self.score = [nb.ref_spots.score.astype(np.float16), score.astype(np.float16)]
        self.gene_no = [nb.ref_spots.gene_no, gene_no]

        # Add current thresholds
        config = nb.get_config()['thresholds']
        if config['intensity'] is None:
            config['intensity'] = nb.call_spots.gene_efficiency_intensity_thresh
        if score_thresh is None:
            score_thresh = config['score_ref']
        if intensity_thresh is None:
            intensity_thresh = config['intensity']
        self.score_thresh = [score_thresh] * 2
        self.intensity_thresh = intensity_thresh

        # Add omp gene assignment if have page
        if nb.has_page('omp'):
            if score_omp_multiplier is None:
                score_omp_multiplier = config['score_omp_multiplier']
            self.score_multiplier = score_omp_multiplier
            self.nbp_omp = nb.omp
            if score_omp_thresh is None:
                score_omp_thresh = config['score_omp']
            self.score_thresh += [score_omp_thresh]
            self.score += [omp_spot_score(self.nbp_omp, self.score_multiplier).astype(np.float16)]
            self.intensity += [self.nbp_omp.intensity.astype(np.float16)]
            self.gene_no += [self.nbp_omp.gene_no]
            self.omp = True
        else:
            self.omp = False
        self.n_plots = len(self.score)
        self.use = None
        self.update_use()

        # Initialise plot
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 7))
        self.subplot_adjust = [0.07, 0.85, 0.12, 0.93]
        self.fig.subplots_adjust(left=self.subplot_adjust[0], right=self.subplot_adjust[1],
                                 bottom=self.subplot_adjust[2], top=self.subplot_adjust[3])
        self.ax.set_ylabel(r"Number of Spots")
        self.ax.set_xlabel('Gene')
        self.ax.set_title(f"Number of Spots assigned to each Gene")

        # record min and max for textbox input
        self.score_min = np.around(self.score[0].min(), 2)  # Min score of score[1] cannot be less than this
        self.score_max = np.around(self.score[1].max(), 2)  # Max score of score[0] cannot be more than this
        self.intensity_min = np.around(self.intensity[0].min(), 2)
        self.intensity_max = np.around(self.intensity[0].max(), 2)
        self.plots = [None] * self.n_plots
        default_colors = plt.rcParams['axes.prop_cycle']._left
        default_colors = default_colors[:2] + default_colors[3:4]  # default 3 is better than default 2 for omp plot
        for i in range(self.n_plots):
            self.plots[i], = self.ax.plot(np.arange(self.n_genes),
                                          np.histogram(self.gene_no[i][self.use[i]],
                                                       np.arange(self.n_genes + 1) - 0.5)[0],
                                          color=default_colors[i]['color'])
            if i == 1:
                self.plots[i].set_visible(False)
        self.ax.set_ylim(0, None)
        self.ax.set_xticks(np.arange(self.n_genes))
        self.ax.set_xticklabels(self.gene_names, rotation=90, size=7)
        self.ax.set_xlim(-0.5, self.n_genes_real - 0.5)

        # Add text box to change score multiplier
        text_box_labels = [r'Score, $\Delta_s$' + '\nThreshold', r'Intensity, $\chi_s$' + '\nThreshold']
        text_box_values = [np.around(self.score_thresh[0], 2), np.around(self.intensity_thresh, 2)]
        text_box_funcs = [self.update_score_thresh, self.update_intensity_thresh]
        if self.omp:
            text_box_labels += [r'OMP Score, $\gamma_s$' + '\nThreshold', 'Score\n' + r'Multiplier, $\rho$']
            text_box_values += [np.around(self.score_thresh[2], 2), np.around(self.score_multiplier, 2)]
            text_box_funcs += [self.update_score_omp_thresh, self.update_score_multiplier]
        self.text_boxes = [None] * len(text_box_labels)
        for i in range(len(text_box_labels)):
            text_ax = self.fig.add_axes([self.subplot_adjust[1] + 0.05,
                                         self.subplot_adjust[2] + 0.15 * (len(text_box_labels) - i - 1), 0.05, 0.04])
            self.text_boxes[i] = TextBox(text_ax, text_box_labels[i], text_box_values[i], color='k',
                                         hovercolor=[0.2, 0.2, 0.2])
            self.text_boxes[i].cursor.set_color('r')
            label = text_ax.get_children()[0]  # label is a child of the TextBox axis
            label.set_position([0.5, 2.75])
            # centering the text
            label.set_verticalalignment('top')
            label.set_horizontalalignment('center')
            self.text_boxes[i].on_submit(text_box_funcs[i])

        # Add buttons to add/remove score_dp histograms
        self.buttons_ax = self.fig.add_axes([self.subplot_adjust[1] + 0.02, self.subplot_adjust[3] - 0.25, 0.15, 0.3])
        plt.axis('off')
        self.button_labels = ["Ref Spots",
                              "Ref Spots - Fake Genes"]
        label_checked = [True, False]
        if self.omp:
            self.button_labels += ["OMP Spots"]
            label_checked += [True]
        self.buttons = CheckButtons(self.buttons_ax, self.button_labels, label_checked)

        for i in range(self.n_plots):
            self.buttons.labels[i].set_fontsize(7)
            self.buttons.labels[i].set_color(default_colors[i]['color'])
            self.buttons.rectangles[i].set_color('w')
        self.buttons.on_clicked(self.choose_plots)
        plt.show()

    def update_use(self):
        # use[i][s] indicates whether spot s for gene assignment method i passes the score and intensity thresholding
        self.use = [np.array([self.score[i] > self.score_thresh[i],
                              self.intensity[i] > self.intensity_thresh]).all(axis=0) for i in range(self.n_plots)]

    def update(self, inds_update: Optional[List[int]] = None):
        if inds_update is None:
            inds_update = np.arange(len(self.plots))  # By default update all plots
        ylim_old = self.ax.get_ylim()[1]  # To check whether we need to change y limit
        ylim_new = 0
        self.update_use()
        for i in np.arange(self.n_plots):
            if i in inds_update:
                self.plots[i].set_data(np.arange(self.n_genes),
                                       np.histogram(self.gene_no[i][self.use[i]],
                                                    np.arange(self.n_genes + 1) - 0.5)[0])
            ylim_new = np.max([ylim_new, self.plots[i].get_ydata().max()])
        if np.abs(ylim_new - ylim_old) / np.max([ylim_new, ylim_old]) < self.ylim_tol:
            ylim_new = ylim_old
        self.ax.set_ylim(0, ylim_new)
        self.ax.figure.canvas.draw()

    def update_score_thresh(self, text):
        # Change the number of ref spots shown to be consistent with new threshold
        try:
            score_thresh = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nScore Threshold given, {text}, is not valid")
            score_thresh = self.score_thresh[0]
        if score_thresh < self.score_min:
            warnings.warn(f"Min Score is {self.score_min} so clipping threshold at {self.score_min}")
            score_thresh = self.score_min
        if score_thresh > self.score_max:
            warnings.warn(f"Max Score is {self.score_max} so clipping threshold at {self.score_max}")
            score_thresh = self.score_max
        self.score_thresh[0] = score_thresh
        self.score_thresh[1] = score_thresh
        self.text_boxes[0].set_val(np.around(score_thresh, 2))
        self.update([0, 1])

    def update_intensity_thresh(self, text):
        # Change the number of spots shown to be consistent with new intensity threshold
        try:
            intensity_thresh = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nIntensity Threshold given, {text}, is not valid")
            intensity_thresh = self.intensity_thresh
        if intensity_thresh < self.intensity_min:
            warnings.warn(f"Min Intensity is {self.intensity_min} so clipping threshold at {self.intensity_min}")
            intensity_thresh = self.intensity_min
        if intensity_thresh > self.intensity_max:
            warnings.warn(f"Max Intensity is {self.intensity_max} so clipping threshold at {self.intensity_max}")
            intensity_thresh = self.intensity_max
        self.intensity_thresh = intensity_thresh
        self.text_boxes[1].set_val(np.around(intensity_thresh, 2))
        self.update()

    def update_score_omp_thresh(self, text):
        # Change the number of omp spots shown to be consistent with new threshold
        try:
            score_omp_thresh = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nOMP Score Threshold given, {text}, is not valid")
            score_omp_thresh = self.score_thresh[2]
        if score_omp_thresh < 0:
            warnings.warn("Min OMP Score is 0 so clipping threshold at 0")
            score_omp_thresh = 0
        if score_omp_thresh > 1:
            warnings.warn("Max OMP Score is 1 so clipping threshold at 1")
            score_omp_thresh = 1
        self.score_thresh[2] = score_omp_thresh
        self.text_boxes[2].set_val(np.around(score_omp_thresh, 2))
        self.update([2])

    def update_score_multiplier(self, text):
        # Can see how changing score_multiplier affects number of spots over threshold
        try:
            score_multiplier = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nScore multiplier given, {text}, is not valid")
            score_multiplier = self.score_multiplier
        if score_multiplier < 0:
            warnings.warn("Score multiplier cannot be negative")
            score_multiplier = self.score_multiplier
        self.score_multiplier = score_multiplier
        self.score[2] = omp_spot_score(self.nbp_omp, self.score_multiplier)
        self.text_boxes[3].set_val(np.around(score_multiplier, 2))
        self.update([2])

    def choose_plots(self, label):
        index = self.button_labels.index(label)
        if self.buttons.get_status()[1]:
            self.ax.set_xlim(-0.5, self.n_genes - 0.5)
        else:
            self.ax.set_xlim(-0.5, self.n_genes_real - 0.5)
        self.plots[index].set_visible(not self.plots[index].get_visible())
        self.ax.figure.canvas.draw()
