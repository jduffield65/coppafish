import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from ..omp.track_fit import get_track_info
from ...setup import Notebook
from ...utils.base import round_any
from .weight import view_weight
from typing import Optional, List
import warnings
plt.style.use('dark_background')


class view_score:
    intense_gene_thresh = 0.2   # Crosshair will be plotted for rounds/channels where gene
    # bled code more intense than this
    check_tol = 1e-4  # Weights and dot products are deemed the same if closer than this.
    weight_plot_ind = 4  # When click on weight plot, will open view_weight plot

    def __init__(self, nb: Notebook, spot_no: int, method: str = 'omp', g: Optional[int] = None,
                 iter: int = 0, omp_fit_info: Optional[List] = None, check_weight: bool = False):
        """
        This produces 4 plots on the first row, showing spot_color, residual, variance and weight squared (basically
        the normalised inverse variance).

        The bottom row shows the contribution from background and genes to the variance.

        The iteration as well as the alpha and beta parameters used to compute the weight can be changed through
        the text boxes.

        If the weight plot is clicked on, the `view_weight` plot will open for the current iteration.

        Args:
            nb: *Notebook* containing at least the *call_spots* and *ref_spots* pages.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
            g: Gene to view dot product calculation for.
                If left as `None`, will show the gene with the largest dot product score.
            iter: Iteration in OMP to view the dot product calculation for i.e. the number of genes
                which have already been fitted (`iter=0` will have only background fit,
                `iter=1` will have background + 1 gene etc.).
                The score saved as `nb.ref_spots.score` can be viewed with `iter=0`.
            omp_fit_info: This is a list containing `[track_info, bled_codes, dp_thresh]`.
                It is only ever used to call this function from `view_omp_fit`.
            check_weight: When this is `True`, we raise an error if weight computed here is different
                to that computed with `get_track_info`.
        """
        self.spot_no = spot_no
        if omp_fit_info is None:
            self.track_info, self.bled_codes, self.dp_thresh = get_track_info(nb, spot_no, method)
        else:
            self.track_info, self.bled_codes, self.dp_thresh = omp_fit_info
        self.n_genes, self.n_rounds_use, self.n_channels_use = self.bled_codes.shape
        # allow to view dot product with background
        self.bled_codes = np.append(self.bled_codes, self.track_info['background_codes'], axis=0)
        self.n_genes_all = self.bled_codes.shape[0]
        self.spot_color = self.track_info['residual'][0]

        # Get saved values if anchor method
        if method.lower() != 'omp':
            self.g_saved = nb.ref_spots.gene_no[spot_no]
            if self.track_info['gene_added'][2] < self.n_genes:
                # Possibility best gene will be background here, but impossible for saved best gene to be background
                if self.track_info['gene_added'][2] != self.g_saved and check_weight:
                    raise ValueError(f"\nBest gene saved was {self.g_saved} but with parameters used here, it "
                                     f"was {self.track_info['gene_added'][2]}.\nEnsure that alpha and beta in "
                                     f"config['call_spots'] have not been changed.\n"
                                     f"Set check_weight=False to skip this error.")
            self.dp_val_saved = nb.ref_spots.score[spot_no]
            config_name = 'call_spots'
        else:
            self.g_saved = None
            self.dp_val_saved = None
            config_name = 'omp'

        # Get default dot product params
        config = nb.get_config()
        self.alpha = config[config_name]['alpha']
        self.beta = config[config_name]['beta']
        self.dp_norm_shift = nb.call_spots.dp_norm_shift
        self.check_weight = check_weight

        self.n_iter = self.track_info['residual'].shape[0] - 2  # first two indices in track is not added gene
        if iter >= self.n_iter or iter < 0:
            warnings.warn(f"Only {self.n_iter} iterations for this pixel but iter={iter}, "
                          f"setting iter = {self.n_iter - 1}.")
            iter = self.n_iter - 1
        self.iter = iter
        if g is None:
            g = self.track_info['gene_added'][2 + iter]
        self.g = g
        self.gene_names = list(nb.call_spots.gene_names) + [f'BG{i}' for i in nb.basic_info.use_channels]
        self.use_channels = nb.basic_info.use_channels
        self.use_rounds = nb.basic_info.use_rounds

        # Initialize data
        self.dp_val = None
        self.dp_weight_val = None
        self.im_data = None
        self.update_data()

        # Initialize plot
        self.title = None
        self.vmax = None
        self.get_cax_lim()
        self.fig = plt.figure(figsize=(16, 7))
        ax1 = self.fig.add_subplot(2, 4, 1)
        ax2 = self.fig.add_subplot(2, 4, 5)
        ax3 = self.fig.add_subplot(2, 4, 2)
        ax4 = self.fig.add_subplot(2, 4, 6)
        ax5 = self.fig.add_subplot(1, 4, 3)
        ax6 = self.fig.add_subplot(2, 4, 4)
        ax7 = self.fig.add_subplot(2, 4, 8)
        self.ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        self.ax[0].get_shared_x_axes().join(self.ax[0], *self.ax[1:])
        self.ax[0].get_shared_y_axes().join(self.ax[0], *self.ax[1:])
        self.subplot_adjust = [0.05, 0.87, 0.05, 0.9]
        self.fig.subplots_adjust(left=self.subplot_adjust[0], right=self.subplot_adjust[1],
                                 bottom=self.subplot_adjust[2], top=self.subplot_adjust[3])
        self.im = [None] * len(self.ax)
        self.set_up_plots()
        self.set_titles()
        self.add_rectangles()

        # Text boxes to change parameters
        text_box_labels = ['Gene', 'Iteration', r'$\alpha$', r'$\beta$', r'dp_shift, $\lambda_d$']
        text_box_values = [self.g, self.iter, self.alpha, self.beta, self.dp_norm_shift]
        text_box_funcs = [self.update_g, self.update_iter, self.update_alpha, self.update_beta,
                          self.update_dp_norm_shift]
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

        # Make so if click on weight plot, it opens view_weight
        self.nb = nb
        self.method = method
        self.fig.canvas.mpl_connect('button_press_event', self.show_weight)
        plt.show()

    def update_data(self):
        bled_code = self.bled_codes[self.g]
        residual = self.track_info['residual'][1 + self.iter]  # want to find dot product at this iteration
        dp_norm_shift = self.dp_norm_shift * np.sqrt(self.n_rounds_use)
        residual_norm = residual / (np.linalg.norm(residual) + dp_norm_shift)
        dot_product = bled_code * residual_norm
        background_var = self.track_info['background_coefs'] ** 2 @ \
                         self.bled_codes[-self.n_channels_use:].reshape(self.n_channels_use, -1) ** 2 * \
                         self.alpha + self.beta ** 2
        gene_var = self.track_info['coef'][self.iter + 1, :self.n_genes] ** 2 @ \
                   self.bled_codes[:self.n_genes].reshape(self.n_genes, -1) ** 2 * self.alpha  # will be 0 if iter=0
        weight = (1 / (gene_var + background_var)).reshape(self.n_rounds_use, self.n_channels_use)
        if self.check_weight:
            # Sanity check that calculation of weight here matches that in get_track_info
            if np.abs(weight - self.track_info['inverse_var'][2 + self.iter]).max() > self.check_tol:
                raise ValueError("Weight calculated is not the same as that from get_track_info\n"
                                 "Set check_weight=False to skip this error.")
        # Normalise weight so max possible dot_product_weight.sum() is 1.
        weight = weight / np.sum(weight) * self.n_rounds_use * self.n_channels_use
        dot_product_weight = dot_product * weight
        if self.check_weight and self.g == self.track_info['gene_added'][2 + self.iter] and self.g < self.n_genes:
            # Sanity check that calculation of weighted dot product here matches that in get_track_info
            # Don't do this if best gene is a background gene because then score set to 0 in get_track_info
            # so would get an error here.
            if np.abs(dot_product_weight.sum() - self.track_info['dot_product'][2 + self.iter]) > self.check_tol:
                raise ValueError("dot_product_weight calculated is not the same as that from get_track_info.\n"
                                 "Set check_weight=False to skip this error.")

        self.im_data = [self.spot_color.T, residual.T, residual_norm.T, bled_code.T,
                        weight.T, dot_product.T, dot_product_weight.T]
        if dot_product_weight.sum() < 0 and dot_product.sum() < 0:
            self.im_data[3] = -self.im_data[3]  # Show negative bled code if negative dot product
        self.dp_val = self.im_data[-2].sum()
        self.dp_weight_val = self.im_data[-1].sum()

        # Sanity check for ref_spots case, where we have saved dot product value for best gene
        # on iteration 0
        if self.check_weight and self.g_saved is not None and self.iter == 0 and self.g == self.g_saved:
            if np.abs(np.float32(self.dp_weight_val) - self.dp_val_saved) > self.check_tol:
                raise ValueError(f"\nDot product score calculated here is {self.dp_weight_val} but saved value "
                                 f"is {self.dp_val_saved}.\nEnsure that alpha and beta in config['call_spots'] "
                                 f"have not been changed.\nSet check_weight=False to skip this error.")

    def get_cax_lim(self):
        self.vmax = np.zeros(len(self.im_data))
        for i in range(len(self.im_data)):
            self.vmax[i] = np.max(np.abs(self.im_data[i]))

        # Plots 0, 1, 2, 3 have same cax
        self.vmax[0] = np.max(self.vmax[:2]) + 0.05
        self.vmax[1] = self.vmax[0]

        # Plots 2 and 3 have same cax
        self.vmax[2] = np.max(self.vmax[2:4]) + 0.1
        self.vmax[3] = self.vmax[2]

        # The higher the iter, the larger the weight and thus the smaller the vmax
        # increase needs to be.
        self.vmax[4] = self.vmax[4] + 1 - np.clip(self.iter * 0.4, 0, 0.9)

        # Plots 5 and 6 have same cax
        self.vmax[5] = np.max(self.vmax[5:]) + 0.1
        self.vmax[6] = self.vmax[5]

    def set_up_plots(self):
        for i in range(len(self.ax)):
            if i == 4:
                self.im[i] = self.ax[i].imshow(self.im_data[i], cmap="viridis",
                                               vmin=0, vmax=self.vmax[i])
            else:
                self.im[i] = self.ax[i].imshow(self.im_data[i], cmap="bwr",
                                               vmin=-self.vmax[i], vmax=self.vmax[i])
            self.ax[i].set_xticks(np.arange(self.n_rounds_use))
            self.ax[i].set_yticks(np.arange(self.n_channels_use))
            if i <= 1:
                self.ax[i].set_yticklabels(self.use_channels)
            else:
                self.ax[i].set_yticklabels([])
            if np.isin(i, [1, 3, 4, 6]):
                self.ax[i].set_xticklabels(self.use_rounds)
            else:
                self.ax[i].set_xticklabels([])
        cbar = self.fig.colorbar(self.im[0], ax=self.ax[:2], fraction=0.046, pad=0.04, aspect=51)
        if self.vmax[0] > 1.2:
            spacing = 0.5
        else:
            spacing = 0.2
        cbar.set_ticks(np.arange(-round_any(self.vmax[0], spacing, 'floor'),
                                 round_any(self.vmax[0], spacing, 'floor') + spacing / 2, spacing))
        cbar = self.fig.colorbar(self.im[2], ax=self.ax[2:4], fraction=0.046, pad=0.04, aspect=51)
        cbar.set_ticks(np.arange(-round_any(self.vmax[2], 0.2, 'floor'),
                                 round_any(self.vmax[2], 0.2, 'floor') + 0.1, 0.2))
        cbar = self.fig.colorbar(self.im[4], ax=self.ax[4], fraction=0.046, pad=0.04)
        cbar.set_ticks(np.arange(0, round_any(self.vmax[4], 0.2, 'floor') + 0.1, 0.2))
        cbar = self.fig.colorbar(self.im[5], ax=self.ax[5:], fraction=0.046, pad=0.04, aspect=51)
        cbar.set_ticks(np.arange(-round_any(self.vmax[5], 0.1, 'floor'),
                                 round_any(self.vmax[5], 0.1, 'floor') + 0.05, 0.1))
        self.fig.supylabel('Channel')
        self.fig.supxlabel('Round', size=12, x=(self.subplot_adjust[0] + self.subplot_adjust[1]) / 2)
        self.im[-1].axes.figure.canvas.draw()

    def set_titles(self):
        self.title = [r'Spot Color, $\mathbf{\zeta_s}$', f'Iter {self.iter} Residual, ' + r'$\mathbf{\zeta_{si}}$',
                      r"Normalised Residual, $\mathbf{\tilde{\zeta}_{si}} = "
                      r"\frac{\mathbf{\zeta_{si}}}{|\mathbf{\zeta_{si}}| + \lambda_d}$",
                      f'Predicted Code for {self.gene_names[self.g]}, ' + r'$\mathbf{b_g}$',
                      r'Weight Squared, $\mathbf{\omega^2_{si}}$',
                      f'Dot Product:' + r' $\sum_{rc}\delta_{{sig}_{rc}}$' + f'= {np.around(self.dp_val, 2)}\n' +
                      r'$\delta_{{sig}_{rc}} = \tilde{\zeta}_{{si}_{rc}}b_{g_{rc}}$',
                      'Score:' + r' $\sum_{rc}\Delta_{{sig}_{rc}}$' + f'= {np.around(self.dp_weight_val, 2)}\n' +
                      r'$\Delta_{{sig}_{rc}} = \omega^2_{{si}_{rc}}\delta_{{sig}_{rc}}$']
        for i in range(len(self.ax)):
            if i == 3:
                is_fail_thresh = self.g >= self.n_genes or \
                                 np.isin(self.g, self.track_info['gene_added'][2:self.iter + 2])
            elif i == 5:
                is_fail_thresh = np.abs(self.dp_val) < self.dp_thresh
            elif i == 6:
                is_fail_thresh = np.abs(self.dp_weight_val) < self.dp_thresh
            else:
                is_fail_thresh = False
            if is_fail_thresh:
                color = 'r'
            else:
                color = 'w'
            self.ax[i].set_title(self.title[i], size=10, color=color)
        if self.g == self.track_info['gene_added'][2 + self.iter]:
            title_extra = " (Best) "
        else:
            title_extra = " "
        plt.suptitle(f"Dot Product Calculation for{title_extra}Gene {self.g}, {self.gene_names[self.g]}, on iteration "
                     f"{self.iter} of OMP for spot {self.spot_no}",
                     x=(self.subplot_adjust[0] + self.subplot_adjust[1]) / 2)

    def add_rectangles(self):
        intense_gene_cr = np.where(np.abs(self.im_data[3]) > self.intense_gene_thresh)
        for i in [0, 1, 2, 3, 5, 6]:
            [p.remove() for p in reversed(self.ax[i].patches)]  # remove current rectangles
            for j in range(len(intense_gene_cr[0])):
                rectangle = plt.Rectangle((intense_gene_cr[1][j] - 0.5, intense_gene_cr[0][j] - 0.5), 1, 1,
                                          fill=False, ec="g", linestyle=':', lw=2)
                self.ax[i].add_patch(rectangle)


    def update(self):
        self.check_weight = False  # Don't check anymore after updated once
        self.update_data()
        for i in range(len(self.ax)):
            self.im[i].set_data(self.im_data[i])
        self.set_titles()
        self.add_rectangles()
        self.im[-1].axes.figure.canvas.draw()

    def update_g(self, text):
        g_best = self.track_info['gene_added'][2 + self.iter]
        try:
            g = int(text)
        except (ValueError, TypeError):
            # if a string, check if is name of gene
            gene_names = list(map(str.lower, self.gene_names))
            try:
                g = gene_names.index(text.lower())
            except ValueError:
                # default to the best gene at this iteration
                warnings.warn(f"\nGene given, {text}, is not valid so setting to best gene for iteration {self.iter}, "
                              f"{g_best}.")
                g = g_best
        if g >= self.n_genes_all or g < 0:
            warnings.warn(f'\nGene index needs to be between 0 and {self.n_genes_all} but value given is '
                          f'{g}.\nSetting to best gene for iteration {self.iter}, {g_best}')
            g = g_best
        self.g = g
        self.text_boxes[0].set_val(g)
        self.update()

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
        self.text_boxes[1].set_val(iter)
        self.update()

    def update_alpha(self, text):
        try:
            alpha = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nAlpha given, {text}, is not valid")
            alpha = self.alpha
        self.alpha = alpha
        self.text_boxes[2].set_val(alpha)
        self.update()

    def update_beta(self, text):
        try:
            beta = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nBeta given, {text}, is not valid")
            beta = self.beta
        self.beta = beta
        self.text_boxes[3].set_val(beta)
        self.update()

    def update_dp_norm_shift(self, text):
        try:
            dp_norm_shift = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\ndp_norm_shift given, {text}, is not valid")
            dp_norm_shift = self.dp_norm_shift
        self.dp_norm_shift = dp_norm_shift
        self.text_boxes[4].set_val(dp_norm_shift)
        self.update()

    def show_weight(self, event):
        # If click on weight plot, it will show weight calculation for the current iteration
        x_click = event.x
        n_iters = len(self.track_info['gene_added']) - 2
        iter_x_coord = np.zeros(len(self.ax))
        for i in range(len(self.ax)):
            iter_x_coord[i] = np.mean(self.ax[i].bbox.extents[:3:2])
        plot_ind = int(np.argmin(np.abs(iter_x_coord - x_click)))
        if plot_ind == self.weight_plot_ind:
            view_weight(self.nb, self.spot_no, self.method, iter=self.iter,
                        score_info=[self.track_info, self.bled_codes[:self.n_genes], self.vmax[4]])
