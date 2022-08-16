import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from ...setup import Notebook
from ..omp.track_fit import get_track_info
from .background import view_background
from typing import Optional, List
import warnings
plt.style.use('dark_background')


class view_weight:
    intense_gene_thresh = 0.2  # Crosshair will be plotted for rounds/channels where gene
    # bled code more intense than this
    check_tol = 1e-4  # Weights and dot products are deamed the same if closer than this.
    n_plots_top_row = 4  # Number of plots on first row
    change_var_cbar_tol = 50  # variance colorbar will change if difference between vmax and max value more than this

    def __init__(self, nb: Notebook, spot_no: int, method: str = 'omp',
                 iter: int = 0, alpha: Optional[float] = None, beta: Optional[float] = None,
                 score_info: Optional[List] = None, check_weight: bool = True):
        """
        This produces at least 5 plots which show how the weight used in the dot product score was calculated.

        The iteration as well as the alpha and beta parameters used to compute the weight can be changed
        with the text boxes.

        Args:
            nb: *Notebook* containing at least the *call_spots* and *ref_spots* pages.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
            iter: Iteration in OMP to view the dot product calculation for i.e. the number of genes
                which have already been fitted (`iter=0` will have only background fit,
                `iter=1` will have background + 1 gene etc.).
                The score saved as `nb.ref_spots.score` can be viewed with `iter=0`.
            score_info: This is a list containing `[track_info, bled_codes, weight_vmax]`.
                It is only ever used to call this function from `view_score`.
            check_weight: When this is `True`, we raise an error if weight computed here is different
                to that computed with `get_track_info`.
        """
        self.spot_no = spot_no
        if score_info is None:
            self.track_info, self.bled_codes = get_track_info(nb, spot_no, method)[:2]
            self.weight_vmax = None
        else:
            self.track_info, self.bled_codes, self.weight_vmax = score_info
        self.n_genes, self.n_rounds_use, self.n_channels_use = self.bled_codes.shape
        # allow to view dot product with background
        self.bled_codes = np.append(self.bled_codes, self.track_info['background_codes'], axis=0)
        self.n_genes_all = self.bled_codes.shape[0]
        self.spot_color = self.track_info['residual'][0]

        if method.lower() == 'omp':
            config_name = 'omp'
        else:
            config_name = 'call_spots'
        # Get default params
        config = nb.get_config()
        if alpha is None:
            alpha = config[config_name]['alpha']
        if beta is None:
            beta = config[config_name]['beta']
        self.alpha = alpha
        self.beta = beta
        self.check_weight = check_weight

        self.n_iter = self.track_info['residual'].shape[0] - 2  # first two indices in track is not added gene
        if iter >= self.n_iter or iter < 0:
            warnings.warn(f"Only {self.n_iter} iterations for this pixel but iter={iter}, "
                          f"setting iter = {self.n_iter - 1}.")
            iter = self.n_iter - 1
        self.iter = iter
        self.gene_names = list(nb.call_spots.gene_names) + [f'BG{i}' for i in nb.basic_info.use_channels]
        self.use_channels = nb.basic_info.use_channels
        self.use_rounds = nb.basic_info.use_rounds

        # Initialize data
        self.n_plots = self.n_plots_top_row + self.n_iter
        self.update_data()

        # Initialize plots
        self.vmax = None
        self.get_cax_lim()
        n_rows = 2
        n_cols = int(np.max([self.n_plots_top_row, self.n_iter]))
        self.fig = plt.figure(figsize=(16, 7))
        self.ax = []
        for i in range(self.n_plots):
            if i >= self.n_plots_top_row:
                # So goes to next row
                self.ax += [self.fig.add_subplot(n_rows, n_cols, n_cols + i + 1 - self.n_plots_top_row)]
            else:
                self.ax += [self.fig.add_subplot(n_rows, n_cols, i + 1)]
        # Y and X axis are the same for all plots hence share
        self.ax[0].get_shared_x_axes().join(self.ax[0], *self.ax[1:])
        self.ax[0].get_shared_y_axes().join(self.ax[0], *self.ax[1:])
        self.subplot_adjust = [0.05, 0.87, 0.07, 0.9]
        self.fig.subplots_adjust(left=self.subplot_adjust[0], right=self.subplot_adjust[1],
                                 bottom=self.subplot_adjust[2], top=self.subplot_adjust[3])
        self.im = [None] * self.n_plots
        self.variance_cbar = None
        self.set_up_plots()
        self.set_titles()
        self.add_rectangles()

        # Text boxes to change parameters
        text_box_labels = ['Iteration', r'$\alpha$', r'$\beta$']
        text_box_values = [self.iter, self.alpha, self.beta]
        text_box_funcs = [self.update_iter, self.update_alpha, self.update_beta]
        self.text_boxes = [None] * len(text_box_labels)
        for i in range(len(text_box_labels)):
            text_ax = self.fig.add_axes([self.subplot_adjust[1] + 0.05, self.subplot_adjust[3] - 0.15 * (i+1),
                                         0.05, 0.04])
            self.text_boxes[i] = TextBox(text_ax, text_box_labels[i], text_box_values[i], color='k',
                                         hovercolor=[0.2, 0.2, 0.2])
            self.text_boxes[i].cursor.set_color('r')
            # change text box title to be above not to the left of box
            label = text_ax.get_children()[0]  # label is a child of the TextBox axis
            label.set_position([0.5, 1.75])  # [x,y] - change here to set the position
            # centering the text
            label.set_verticalalignment('top')
            label.set_horizontalalignment('center')
            self.text_boxes[i].on_submit(text_box_funcs[i])

        self.nb = nb
        self.method = method
        self.fig.canvas.mpl_connect('button_press_event', self.show_background)
        plt.show()

    def update_data(self):
        residual = self.track_info['residual'][1 + self.iter]  # want to find dot product at this iteration
        background_var = self.track_info['background_coefs'] ** 2 @ \
                         self.bled_codes[-self.n_channels_use:].reshape(self.n_channels_use, -1) ** 2 * \
                         self.alpha + self.beta ** 2
        gene_var = self.track_info['coef'][self.iter + 1, :self.n_genes] ** 2 @ \
                   self.bled_codes[:self.n_genes].reshape(self.n_genes, -1) ** 2 * self.alpha  # will be 0 if iter=0
        variance = (gene_var + background_var).reshape(self.n_rounds_use, self.n_channels_use)
        weight = (1 / (gene_var + background_var)).reshape(self.n_rounds_use, self.n_channels_use)
        if self.check_weight:
            # Sanity check that calculation of weight here matches that in get_track_info
            if np.abs(weight - self.track_info['inverse_var'][2 + self.iter]).max() > self.check_tol:
                raise ValueError("Weight calculated is not the same as that from get_track_info\n"
                                 "Set check_weight=False to skip this error.")
        # Normalise weight so max possible dot_product_weight.sum() is 1.
        weight = weight / np.sum(weight) * self.n_rounds_use * self.n_channels_use

        self.im_data = [self.spot_color.T, residual.T, variance.T, weight.T]

        # Add the background image
        background_image = np.zeros((self.n_rounds_use, self.n_channels_use))
        for c in range(self.n_channels_use):
            background_image += self.track_info['background_codes'][c] * self.track_info['background_coefs'][c]
        self.im_data += [background_image.T]

        # Add coef * bled_code for all genes added - Don't add last gene because we don't use last gene
        # because to compute the weight for last gene, we use all genes fitted prior to it.
        for i in range(self.n_iter - 1):
            g = self.track_info['gene_added'][i + 2]
            coef = self.track_info['coef'][self.iter + 1, g]
            self.im_data += [self.bled_codes[g].T * coef]

    def get_cax_lim(self):
        bled_code_max = np.max(np.abs(self.track_info['residual']))
        var_max = np.max(1 / self.track_info['inverse_var'][2+self.iter])
        weights = [val / np.sum(val) * self.n_rounds_use * self.n_channels_use for val in
                   self.track_info['inverse_var'][2:]]
        weights_max = np.max(weights)
        self.vmax = np.full(self.n_plots, bled_code_max + 0.05)
        self.vmax[2] = var_max + 5
        if self.weight_vmax is None:
            self.vmax[3] = weights_max + 0.5
        else:
            self.vmax[3] = self.weight_vmax

    def set_up_plots(self):
        for i in range(self.n_plots):
            if np.isin(i, [2, 3]):
                self.im[i] = self.ax[i].imshow(self.im_data[i], cmap="viridis",
                                               vmin=0, vmax=self.vmax[i])
            else:
                self.im[i] = self.ax[i].imshow(self.im_data[i], cmap="bwr",
                                               vmin=-self.vmax[i], vmax=self.vmax[i])
            self.ax[i].set_xticks(np.arange(self.n_rounds_use))
            self.ax[i].set_yticks(np.arange(self.n_channels_use))
            if np.isin(i, [0, self.n_plots_top_row]):
                self.ax[i].set_yticklabels(self.use_channels)
            else:
                self.ax[i].set_yticklabels([])
            if i >= self.n_plots_top_row:
                self.ax[i].set_xticklabels(self.use_rounds)
            else:
                self.ax[i].set_xticklabels([])
        self.fig.colorbar(self.im[self.n_plots_top_row], ax=self.ax[self.n_plots_top_row:],
                          fraction=0.046, pad=0.01, aspect=40)
        self.variance_cbar = self.fig.colorbar(self.im[2], ax=self.ax[2], fraction=0.023, pad=0.04, aspect=40)
        self.fig.colorbar(self.im[3], ax=self.ax[3], fraction=0.023, pad=0.04, aspect=40)
        self.fig.supylabel('Channel')
        self.fig.supxlabel('Round', size=12, x=(self.subplot_adjust[0] + self.subplot_adjust[1]) / 2)
        self.im[-1].axes.figure.canvas.draw()

    def set_titles(self):
        self.title = [r'Spot Color, $\mathbf{\zeta_s}$', f'Iter {self.iter} Residual, ' + r'$\mathbf{\zeta_{si}}$',
                      r"$\sigma^2_{{si}_{rc}} = \beta^2 + \alpha\sum_g\mu^2_{sig}b^2_{g_{rc}}$",
                      r"$\omega^2_{{si}_{rc}} = n_{r}n_{c}"
                      r"\frac{\sigma^{-2}_{{si}_{rc}}}{\sum_{rc}\sigma^{-2}_{{si}_{rc}}}$",
                      r"Background, $\mathbf{B_s}=\sum_{g=" + str(self.n_genes) + "}^{" + str(self.n_genes_all - 1) +
                      "}\mu_{sg}\mathbf{b}_g$"]
        for i in range(self.n_iter - 1):
            g = self.track_info['gene_added'][i + 2]
            gene_name = self.gene_names[g]
            coef = str(np.around(self.track_info['coef'][self.iter + 1, g], 2))
            gi_str = f"i={self.iter},g={g}"
            title = gene_name +r", $\mu_{sig}\mathbf{b}_{"+str(g)+"}$\n$\mu_{"+gi_str+"}="+coef+"$"
            self.title += [title]
        for i in range(len(self.ax)):
            self.ax[i].set_title(self.title[i], size=10, color='w')
        plt.suptitle(r"Weight Squared, $\mathbf{\omega}_{si}$, Calculation, at iteration "
                     f"{self.iter} of OMP for spot {self.spot_no}",
                     x=(self.subplot_adjust[0] + self.subplot_adjust[1]) / 2)

    def add_rectangles(self):
        # Add green rectangle wherever bled code > 0.2
        for i in range(self.n_iter-1):
            [p.remove() for p in reversed(self.ax[5 + i].patches)]  # remove current rectangles
        for i in range(self.iter):
            g = self.track_info['gene_added'][i + 2]
            intense_gene_cr = np.where(self.bled_codes[g].T > self.intense_gene_thresh)
            for j in range(len(intense_gene_cr[0])):
                rectangle = plt.Rectangle((intense_gene_cr[1][j] - 0.5, intense_gene_cr[0][j] - 0.5), 1, 1,
                                          fill=False, ec="g", linestyle=':', lw=2)
                self.ax[5+i].add_patch(rectangle)

    def update(self):
        self.check_weight = False  # Don't check anymore after updated once
        self.update_data()
        for i in range(len(self.ax)):
            self.im[i].set_data(self.im_data[i])
        self.set_titles()
        self.add_rectangles()
        if np.abs(self.im_data[2].max() - self.variance_cbar.vmax) > self.change_var_cbar_tol:
            # Change clims if variance changes by large amount
            self.im[2].set_clim(0, self.im_data[2].max())
            self.variance_cbar.update_normal(self.im[2])
        self.im[-1].axes.figure.canvas.draw()

    def update_iter(self, text):
        try:
            iter = int(text)
            if iter >= self.n_iter or iter < 0:
                warnings.warn(f"\nOnly {self.n_iter} iterations for this pixel but iter={iter}.\n"
                              f"Setting to last iteration, {self.n_iter - 1}.")
                iter = self.n_iter - 1
        except (ValueError, TypeError):
            warnings.warn(f"\nIteration given, {text}, is not valid")
            iter = self.iter
        self.iter = iter
        self.text_boxes[0].set_val(iter)
        self.update()

    def update_alpha(self, text):
        try:
            alpha = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nAlpha given, {text}, is not valid")
            alpha = self.alpha
        self.alpha = alpha
        self.text_boxes[1].set_val(alpha)
        self.update()

    def update_beta(self, text):
        try:
            beta = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nBeta given, {text}, is not valid")
            beta = self.beta
        self.beta = beta
        self.text_boxes[2].set_val(beta)
        self.update()

    def show_background(self, event):
        # If click on background plot, it will show background calculation
        x_click = event.x
        y_click = event.y
        if y_click < self.ax[0].bbox.extents[1] and x_click < self.ax[1].bbox.extents[0]:
            view_background(self.nb, self.spot_no, self.method,
                            track_info=[self.track_info, self.vmax[0]])