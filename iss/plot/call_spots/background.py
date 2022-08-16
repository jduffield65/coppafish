import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from ...setup import Notebook
from ..omp.track_fit import get_track_info
from ...utils.base import round_any
from typing import Optional, List
import warnings
plt.style.use('dark_background')


class view_background:
    check_tol = 1e-4  # Background coefficients are deemed the same if closer than this.
    spot_color_plot_inds = [0, 1, 7, 8]  # These all share the same caxis because all related to spot_color
    weight_plot_ind = 2  # Index of weight plot
    coef_plot_ind = [5, 6]  # Coefficient plots have dimension n_channels x 1
    change_weight_dp_cbar_tol = 0.1  # Chane weight_dp colorbar if changes by large amount

    def __init__(self, nb: Notebook, spot_no: int, method: str = 'omp', check: bool = True,
                 track_info: Optional[List] = None):
        """
        This shows how the background coefficients were calculated.

        The weighted dot product is equal to weight multiplied by dot product.
        Coefficient for background gene c is the sum over all rounds of weighted dot product in channel c.

        Also shows residual after removing background.

        Args:
            nb: *Notebook* containing at least the *call_spots* and *ref_spots* pages.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
            check: When this is `True`, we raise an error if background coefs computed here is different
                to that computed with `get_track_info`.
            track_info: To use when calling from `view_weight`.
        """
        self.spot_no = spot_no
        if track_info is None:
            self.track_info = get_track_info(nb, spot_no, method)[0]
            self.color_vmax = None
        else:
            self.track_info, self.color_vmax = track_info
        self.n_genes_all = self.track_info['coef'][0].size
        self.spot_color = self.track_info['residual'][0]
        self.n_rounds_use, self.n_channels_use = self.spot_color.shape
        self.n_genes = self.n_genes_all - self.n_channels_use
        self.use_channels = nb.basic_info.use_channels
        self.use_rounds = nb.basic_info.use_rounds

        self.background_weight_shift = nb.call_spots.background_weight_shift
        self.check = check

        # Initialize data
        self.im_data = None
        self.update_data()
        hi = 5

        # Initialize plots
        self.vmax = None
        self.weight_dp_max_initial = None
        self.get_cax_lim()
        self.fig = plt.figure(figsize=(16, 7))
        self.ax = []
        ax1 = self.fig.add_subplot(2, 5, 1)
        ax2 = self.fig.add_subplot(2, 5, 6)
        ax3 = self.fig.add_subplot(1, 5, 2)
        ax4 = self.fig.add_subplot(2, 5, 3)
        ax5 = self.fig.add_subplot(2, 5, 8)
        ax6 = self.fig.add_subplot(2, 5, 4)
        ax7 = self.fig.add_subplot(2, 5, 9)
        ax8 = self.fig.add_subplot(2, 5, 5)
        ax9 = self.fig.add_subplot(2, 5, 10)
        self.ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        self.ax[0].get_shared_y_axes().join(self.ax[0], *self.ax[1:])
        # Coef plots have different x-axis
        self.ax[self.coef_plot_ind[0]].get_shared_x_axes().join(self.ax[self.coef_plot_ind[0]],
                                                                self.ax[self.coef_plot_ind[1]])
        # All other plots have same axis
        self.ax[0].get_shared_x_axes().join(self.ax[0], *self.ax[1:self.coef_plot_ind[0]])
        self.ax[0].get_shared_x_axes().join(self.ax[0], *self.ax[self.coef_plot_ind[1]+1:])
        self.subplot_adjust = [0.05, 0.97, 0.07, 0.9]
        self.fig.subplots_adjust(left=self.subplot_adjust[0], right=self.subplot_adjust[1],
                                 bottom=self.subplot_adjust[2], top=self.subplot_adjust[3])
        self.im = [None] * len(self.ax)
        self.set_up_plots()
        self.set_titles()

        weight_plot_pos = self.ax[self.weight_plot_ind].get_position()
        box_x = np.mean([weight_plot_pos.x0, weight_plot_pos.x1])
        box_y = weight_plot_pos.y0
        text_ax = self.fig.add_axes([box_x, box_y - 0.15, 0.05, 0.04])
        self.text_box = TextBox(text_ax, r'background_shift, $\lambda_b$', self.background_weight_shift, color='k',
                                         hovercolor=[0.2, 0.2, 0.2])
        self.text_box.cursor.set_color('r')
        label = text_ax.get_children()[0]  # label is a child of the TextBox axis
        label.set_position([0.5, 1.75])  # [x,y] - change here to set the position
        # centering the text
        label.set_verticalalignment('top')
        label.set_horizontalalignment('center')
        self.text_box.on_submit(self.update_background_weight_shift)

        plt.show()

    def update_data(self):
        residual = self.track_info['residual'][1]  # want to find dot product at this iteration

        # Add the background image
        background_image = np.zeros((self.n_rounds_use, self.n_channels_use))
        for c in range(self.n_channels_use):
            background_image += self.track_info['background_codes'][c]

        weight = 1 / (np.abs(self.spot_color) + self.background_weight_shift)
        weight_squared = weight ** 2 / np.sum(weight**2 * background_image**2, axis=0, keepdims=True)

        dot_product = self.spot_color * background_image
        dot_product_weight = dot_product * weight_squared
        coef = np.sum(dot_product, axis=0)
        coef_weight = np.sum(dot_product_weight, axis=0)
        if self.check:
            if np.abs(coef_weight - self.track_info['background_coefs']).max() > self.check_tol:
                raise ValueError("Background calculated is not the same as that from get_track_info\n"
                                 "Set check=False to skip this error.")

        self.im_data = [self.spot_color.T, background_image.T, weight_squared.T, dot_product.T,
                        dot_product_weight.T, coef[:, np.newaxis], coef_weight[:, np.newaxis], residual.T,
                        (background_image * coef_weight[np.newaxis]).T]
        self.im_data[-2] = self.im_data[0] - self.im_data[-1]   # update residual to use computed background coefs

    def get_cax_lim(self):
        self.vmax = np.zeros(len(self.im_data))
        for i in range(len(self.im_data)):
            self.vmax[i] = np.max(np.abs(self.im_data[i]))

        # Spot color plots have same cax
        if self.color_vmax is None:
            spot_color_max = np.max(self.vmax[np.array(self.spot_color_plot_inds)]) + 0.05
        else:
            spot_color_max = self.color_vmax
        for i in self.spot_color_plot_inds:
            self.vmax[i] = spot_color_max

        # Weight plot
        self.vmax[self.weight_plot_ind] += 0.1

        # Dot product plots all share same caxis
        dp_inds = np.setdiff1d(np.arange(len(self.im_data)), self.spot_color_plot_inds + [self.weight_plot_ind])
        dp_max = np.max(self.vmax[dp_inds[::2]])
        self.vmax[dp_inds[::2]] = dp_max

        # Dot product weight plots all share same caxis
        dp_weight_max = np.max(self.vmax[dp_inds[1::2]])
        self.vmax[dp_inds[1::2]] = dp_weight_max

    def set_up_plots(self):
        for i in range(len(self.ax)):
            if i == self.weight_plot_ind:
                self.im[i] = self.ax[i].imshow(self.im_data[i], cmap="viridis",
                                               vmin=0, vmax=self.vmax[i])
            else:
                if np.isin(i, self.coef_plot_ind):
                    aspect = 'auto'
                else:
                    aspect = None
                self.im[i] = self.ax[i].imshow(self.im_data[i], cmap="bwr",
                                               vmin=-self.vmax[i], vmax=self.vmax[i], aspect=aspect)
            if np.isin(i, self.coef_plot_ind):
                # Coefficient plots have dimension of 1 in x.
                self.ax[i].set_xticks([])
            else:
                self.ax[i].set_xticks(np.arange(self.n_rounds_use))
            self.ax[i].set_yticks(np.arange(self.n_channels_use))
            if i <= 1:
                self.ax[i].set_yticklabels(self.use_channels)
            else:
                self.ax[i].set_yticklabels([])
            if np.isin(i, [1, 2, 4, 8]):
                self.ax[i].set_xticklabels(self.use_rounds)
            else:
                self.ax[i].set_xticklabels([])
        # Spot Color
        self.fig.colorbar(self.im[7], ax=self.ax[7:], fraction=0.03, pad=0.04, aspect=81)
        # Weight
        self.fig.colorbar(self.im[self.weight_plot_ind], ax=self.ax[self.weight_plot_ind],
                          fraction=0.02, pad=0.04, aspect=51)
        # Dot product
        dp_inds = np.setdiff1d(np.arange(len(self.im_data)), self.spot_color_plot_inds + [self.weight_plot_ind])
        self.fig.colorbar(self.im[dp_inds[2]], ax=self.ax[dp_inds[2]], fraction=0.046, pad=0.04,
                          aspect=51)
        # Dot product weight
        self.weight_dp_cbar = self.fig.colorbar(self.im[dp_inds[3]], ax=self.ax[dp_inds[3]], fraction=0.046, pad=0.04,
                                                aspect=51)
        if self.vmax[dp_inds[3]] > 2.5:
            spacing = 1
        elif self.vmax[dp_inds[3]] > 1.8:
            spacing = 0.5
        elif self.vmax[dp_inds[3]] > 1:
            spacing = 0.2
        else:
            spacing = 0.1
        self.weight_dp_cbar.set_ticks(np.arange(-round_any(self.vmax[dp_inds[3]], spacing, 'floor'),
                                      round_any(self.vmax[dp_inds[3]], spacing, 'floor') + spacing/2, spacing))
        self.fig.supylabel('Channel')
        self.fig.supxlabel('Round', size=12, x=(self.subplot_adjust[0] + self.subplot_adjust[1]) / 2)
        self.im[-1].axes.figure.canvas.draw()

    def set_titles(self):
        self.title = [r'Spot Color, $\mathbf{\zeta_s}$',
                      r"Background Codes, $\mathbf{B} = \sum_{C=" + str(0) + "}^{" + str(self.n_channels_use - 1) +
                      "}\mathbf{B}_C$",
                      "Weight Squared\n"+r"$\Omega^2_{s_{rc}} = \frac{w^2_{rc}}{\sum_rw^2_{rc}B^2_{rc}}$ where "
                                         r"$w_{rc}=\frac{1}{|\zeta_{s_{rc}}|+\lambda_b}$",
                      r"Dot Product, $\zeta_{s_{rc}}B_{rc}$",
                      r"Weighted Dot Product, $\Omega^2_{s_{rc}}\zeta_{s_{rc}}B_{rc}$",
                      r"Coef, $\sum_r\zeta_{s_{rC}}B_{rC}$",
                      r"Weighted Coef, $\mu_{sC}=\sum_r\Omega^2_{s_{rC}}\zeta_{s_{rC}}B_{rC}$",
                      f'Iter 0 Residual, ' + r'$\mathbf{\zeta_{s0}}=\mathbf{\zeta_s}-\mathbf{B_s}$',
                      r"Background, $\mathbf{B_s}=\sum_{C=" + str(0) + "}^{" + str(self.n_channels_use - 1) +
                      "}\mu_{sC}\mathbf{B}_C$"]
        for i in range(len(self.ax)):
            self.ax[i].set_title(self.title[i], size=10, color='w')
        plt.suptitle(f"Background Coefficient Calculation for for spot {self.spot_no}",
                     x=(self.subplot_adjust[0] + self.subplot_adjust[1]) / 2)

    def update(self):
        self.check = False  # Don't check anymore after updated once
        self.update_data()
        for i in range(len(self.ax)):
            self.im[i].set_data(self.im_data[i])
        self.set_titles()
        dp_inds = np.setdiff1d(np.arange(len(self.im_data)), self.spot_color_plot_inds + [self.weight_plot_ind])
        max_weight_dp = np.abs(self.im_data[dp_inds[3]]).max()
        if np.abs(max_weight_dp - self.weight_dp_cbar.vmax) > self.change_weight_dp_cbar_tol:
            # Change clims if variance changes by large amount
            self.im[dp_inds[3]].set_clim(-max_weight_dp, max_weight_dp)
            self.im[dp_inds[1]].set_clim(-max_weight_dp, max_weight_dp)
            self.weight_dp_cbar.update_normal(self.im[dp_inds[3]])
            if self.weight_dp_cbar.vmax > 2.5:
                spacing = 1
            elif self.weight_dp_cbar.vmax > 1.8:
                spacing = 0.5
            elif self.weight_dp_cbar.vmax > 1:
                spacing = 0.2
            else:
                spacing = 0.1
            self.weight_dp_cbar.set_ticks(np.arange(-round_any(self.weight_dp_cbar.vmax, spacing, 'floor'),
                                                    round_any(self.weight_dp_cbar.vmax, spacing, 'floor') + spacing / 2,
                                                    spacing))
        self.im[-1].axes.figure.canvas.draw()

    def update_background_weight_shift(self, text):
        try:
            background_weight_shift = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nBackground weight shift given, {text}, is not valid")
            background_weight_shift = self.background_weight_shift
        self.background_weight_shift = background_weight_shift
        self.text_box.set_val(background_weight_shift)
        self.update()
