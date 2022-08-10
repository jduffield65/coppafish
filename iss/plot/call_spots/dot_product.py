import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from ...omp.coefs import get_all_coefs
from ...setup import Notebook
from ...utils.base import round_any
from typing import Optional, List, Tuple
import warnings


def get_track_info(nb: Notebook, spot_no: int, method: str, dp_thresh: Optional[float] = None,
                   max_genes: Optional[int] = None) -> Tuple[dict, np.ndarray, float]:
    """
    This runs omp while tracking the residual at each stage.

    Args:
        nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
        spot_no: Spot of interest to get track_info for.
        method: `'anchor'` or `'omp'`.
            Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
        dp_thresh: If None, will use value in omp section of config file.
        max_genes: If None, will use value in omp section of config file.

    Returns:
        `track_info` - dictionary containing info about genes added at each step returned:

            - `background_codes` - `float [n_channels x n_rounds x n_channels]`.
                `background_codes[c]` is the background vector for channel `c` with L2 norm of 1.
            - `background_coefs` - `float [n_channels]`.
                `background_coefs[c]` is the coefficient value for `background_codes[c]`.
            - `gene_added` - `int [n_genes_added + 2]`.
                `gene_added[0]` and `gene_added[1]` are -1.
                `gene_added[2+i]` is the `ith` gene that was added.
            - `residual` - `float [(n_genes_added + 2) x n_rounds x n_channels]`.
                `residual[0]` is the initial `pixel_color`.
                `residual[1]` is the post background `pixel_color`.
                `residual[2+i]` is the `pixel_color` after removing gene `gene_added[2+i]`.
            - `coef` - `float [(n_genes_added + 2) x n_genes]`.
                `coef[0]` and `coef[1]` are all 0.
                `coef[2+i]` are the coefficients for all genes after the ith gene has been added.
            - `dot_product` - `float [n_genes_added + 2]`.
                `dot_product[0]` and `dot_product[1]` are 0.
                `dot_product[2+i]` is the dot product for the gene `gene_added[2+i]`.
            - `inverse_var` - `float [(n_genes_added + 2) x n_rounds x n_channels]`.
                `inverse_var[0]` and `inverse_var[1]` are all 0.
                `inverse_var[2+i]` is the weighting used to compute `dot_product[2+i]`,
                 which down-weights rounds/channels for which a gene has already been fitted.
        `bled_codes` - `float [n_genes x n_use_rounds x n_use_channels]`.
            gene `bled_codes` used in omp with L2 norm = 1.
        `dp_thresh` - threshold dot product score, above which gene is fitted.
    """
    color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                        nb.basic_info.use_channels)]
    n_use_rounds, n_use_channels = color_norm.shape
    if method.lower() == 'omp':
        page_name = 'omp'
        config_name = 'omp'
    else:
        page_name = 'ref_spots'
        config_name = 'call_spots'
    spot_color = nb.__getattribute__(page_name).colors[spot_no][
                     np.ix_(nb.basic_info.use_rounds, nb.basic_info.use_channels)] / color_norm
    n_genes = nb.call_spots.bled_codes_ge.shape[0]
    bled_codes = np.asarray(
        nb.call_spots.bled_codes_ge[np.ix_(np.arange(n_genes),
                                           nb.basic_info.use_rounds, nb.basic_info.use_channels)])
    # ensure L2 norm is 1 for bled codes
    norm_factor = np.expand_dims(np.linalg.norm(bled_codes, axis=(1, 2)), (1, 2))
    norm_factor[norm_factor == 0] = 1  # For genes with no dye in use_dye, this avoids blow up on next line
    bled_codes = bled_codes / norm_factor

    # Get info to run omp
    dp_norm_shift = nb.call_spots.dp_norm_shift * np.sqrt(n_use_rounds)
    config = nb.get_config()
    if dp_thresh is None:
        dp_thresh = config['omp']['dp_thresh']
    alpha = config[config_name]['alpha']
    beta = config[config_name]['beta']
    if max_genes is None:
        max_genes = config['omp']['max_genes']
    weight_coef_fit = config['omp']['weight_coef_fit']

    # Run omp with track to get residual at each stage
    track_info = get_all_coefs(spot_color[np.newaxis], bled_codes, nb.call_spots.background_weight_shift,
                               dp_norm_shift, dp_thresh, alpha, beta, max_genes, weight_coef_fit, True)[2]
    return track_info, bled_codes, dp_thresh


class view_dot_product:
    intense_gene_thresh = 0.2   # Crosshair will be plotted for rounds/channels where gene
    # bled code more intense than this

    def __init__(self, nb: Notebook, spot_no: int, method: str = 'omp', g: Optional[int] = None,
                 iter: int = 0, omp_fit_info: Optional[List] = None):
        """

        Args:
            nb:
            spot_no:
            method:
            g:
            iter:
            omp_fit_info:
        """
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
            if self.track_info['gene_added'][2] != self.g_saved:
                raise ValueError(f"\nBest gene saved was {self.g_saved} but with parameters used here, it"
                                 f"was {self.track_info['gene_added'][2]}.\nEnsure that alpha and beta in "
                                 f"config['call_spots'] have not been changed.")
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
        self.check_weight = True
        self.check_tol = 1e-4

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

        text_box_labels = ['Gene', 'Iteration', r'$\alpha$', r'$\beta$', 'dp_norm']
        text_box_values = [self.g, self.iter, int(self.alpha), self.beta, self.dp_norm_shift]
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
                raise ValueError("Weight calculated is not the same as that from get_track_info")
        # Normalise weight so max possible dot_product_weight.sum() is 1.
        weight = weight / np.sum(weight) * self.n_rounds_use * self.n_channels_use
        dot_product_weight = dot_product * weight
        if self.check_weight and self.g == self.track_info['gene_added'][2 + self.iter] and self.g < self.n_genes:
            # Sanity check that calculation of weighted dot product here matches that in get_track_info
            # Don't do this if best gene is a background gene because then score set to 0 in get_track_info
            # so would get an error here.
            if np.abs(dot_product_weight.sum() - self.track_info['dot_product'][2 + self.iter]) > self.check_tol:
                raise ValueError("dot_product_weight calculated is not the same as that from get_track_info")

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
                                 f"have not been changed.")

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
        self.title = ['Spot Color', f'Iter {self.iter} Residual', "Normalised Residual",
                      f'Predicted Code for {self.gene_names[self.g]}',
                      f'Weight Squared',
                      f'Dot Product = {np.around(self.dp_val, 2)}',
                      f'Weighted Dot Product = {np.around(self.dp_weight_val, 2)}']
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
        plt.suptitle(f"Dot Product Calculation for{title_extra}Gene {self.g}, {self.gene_names[self.g]}, at iteration "
                     f"{self.iter} of OMP", x=(self.subplot_adjust[0] + self.subplot_adjust[1]) / 2)

    def add_rectangles(self):
        intense_gene_cr = np.where(self.im_data[3] > self.intense_gene_thresh)
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
