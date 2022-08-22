import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import warnings
from ...setup.notebook import Notebook
from .coefs import get_coef_images
from ...call_spots import omp_spot_score
from typing import Optional


class view_omp_score:
    tol = 1e-4   # error if sum calculation different to this

    def __init__(self, nb: Notebook, spot_no: int, method: str = 'omp', score_multiplier: Optional[float] = None,
                 check: bool = False):
        """
        Diagnostic to show how score is computed in the omp method
        Hatched region in top plot shows pixels which contribute to the final score.
        Score is actually equal to the absolute sum of the top plots in the hatched regions.

        Can also see how `score_omp_multiplier` affects the final score. The larger this is, the more
        the positive pixels contribute compared to the negative.

        !!! warning "Requires access to `nb.file_names.tile_dir`"

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
            score_multiplier: Initial value of `score_omp_multiplier`.
            check: If `True`, will compare score found to that saved in *Notebook* and raise error if they differ.
                Will also check that absolute sum of the top plots in the hatched regions is equal to
                score calculated from counting the number of pixels with the correct sign.

        """
        # TODO: The textbox for this plot seems to be much less responsive than in the other diagnostics
        #  for some reason.
        if method.lower() == 'omp':
            page_name = 'omp'
        else:
            page_name = 'ref_spots'
            check = False
        self.check = check
        self.nbp_omp = nb.omp
        self.spot_no = spot_no
        self.gene_no = nb.__getattribute__(page_name).gene_no[spot_no]
        self.gene_names = nb.call_spots.gene_names
        self.expected_sign_image = nb.omp.spot_shape
        self.nz = self.expected_sign_image.shape[2]
        im_radius = ((np.asarray(self.expected_sign_image.shape) - 1) / 2).astype(int)
        self.coef_images, min_global_yxz, max_global_yxz = get_coef_images(nb, spot_no, method, im_radius)
        # Find where sign of coef image matches that of expected sign image
        self.both_positive = np.asarray([self.coef_images[self.gene_no] > 0, self.expected_sign_image > 0]).all(axis=0)
        self.both_negative = np.asarray([self.coef_images[self.gene_no] < 0, self.expected_sign_image < 0]).all(axis=0)
        self.check_neighbours()

        # Start with default multiplier
        config = nb.get_config()['thresholds']
        if score_multiplier is None:
            score_multiplier = config['score_omp_multiplier']
        self.score_multiplier = score_multiplier
        self.score_thresh = config['score_omp']

        # maximum possible value of any one pixel in expected_shape for any score_multiplier
        self.vmax_expected = 1 / np.min([np.sum(self.expected_sign_image > 0), np.sum(self.expected_sign_image < 0)])
        self.vmax_coef = np.abs(self.coef_images[self.gene_no]).max()

        # Initialize plots
        self.plot_extent = [min_global_yxz[1] - 0.5, max_global_yxz[1] + 0.5,
                            min_global_yxz[0] - 0.5, max_global_yxz[0] + 0.5]
        self.fig, self.ax = plt.subplots(2, self.nz, figsize=(14, 5), sharex=True, sharey=True)
        self.subplot_adjust = [0.05, 0.88, 0.09, 0.85]
        self.fig.subplots_adjust(left=self.subplot_adjust[0], right=self.subplot_adjust[1],
                                 bottom=self.subplot_adjust[2], top=self.subplot_adjust[3])
        self.ax = self.ax.flatten()
        self.im = [None] * self.nz * 2
        expected_image_plot = self.expected_image()
        self.score = self.get_score(expected_image_plot)
        for i in range(self.nz):
            self.im[i] = self.ax[i].imshow(expected_image_plot[:, :, i],
                                           vmin=-self.vmax_expected, vmax=self.vmax_expected, cmap='bwr',
                                           extent=self.plot_extent)
            self.im[i+self.nz] = self.ax[i + self.nz].imshow(self.coef_images[self.gene_no, :, :, i],
                                                             vmin=-self.vmax_coef, vmax=self.vmax_coef, cmap='bwr',
                                                             extent=self.plot_extent)
            if i == (self.nz - 1) / 2:
                self.ax[i].set_title(f"Expected Coefficient Sign\nZ={int(np.rint(min_global_yxz[2] + i))}")
            else:
                self.ax[i].set_title(f"Z={int(np.rint(min_global_yxz[2] + i))}")
        self.set_coef_plot_title()
        self.add_hatching()

        # Set up colorbars for each plot
        mid_point = (self.subplot_adjust[2]+self.subplot_adjust[3])/2
        gap_size = 0.08
        cbar_ax = self.fig.add_axes([self.subplot_adjust[1]+0.01, mid_point+gap_size/2,
                                     0.005, self.subplot_adjust[3]-mid_point-gap_size/2])  # left, bottom, width, height
        self.fig.colorbar(self.im[0], cax=cbar_ax)
        cbar_ax = self.fig.add_axes([self.subplot_adjust[1]+0.01, self.subplot_adjust[2]+gap_size/5,
                                     0.005, mid_point-self.subplot_adjust[2]-gap_size/2])  # left, bottom, width, height
        self.fig.colorbar(self.im[self.nz], cax=cbar_ax)

        # Add titles
        self.fig.supylabel('Y')
        self.fig.supxlabel('X', size=12, x=(self.subplot_adjust[0] + self.subplot_adjust[1]) / 2)
        plt.suptitle(f"OMP Score Calculation for Spot {spot_no}, Gene {self.gene_no}: {self.gene_names[self.gene_no]}",
                     x=(self.subplot_adjust[0] + self.subplot_adjust[1]) / 2)

        # Add text box to change score multiplier
        text_ax = self.fig.add_axes([self.subplot_adjust[1] + 0.062, self.subplot_adjust[2]+gap_size/5,
                                     0.05, 0.04])
        self.text_box = TextBox(text_ax, 'Score\n'+r'Multiplier, $\rho$', str(np.around(self.score_multiplier, 2)),
                                color='k', hovercolor=[0.2, 0.2, 0.2])
        self.text_box.cursor.set_color('r')
        label = text_ax.get_children()[0]  # label is a child of the TextBox axis
        label.set_position([0.5, 2.75])  # [x,y] - change here to set the position
        # centering the text
        label.set_verticalalignment('top')
        label.set_horizontalalignment('center')
        self.text_box.on_submit(self.update)

        plt.show()

    def check_neighbours(self):
        # Check same number of pos/neighbours as calculated in the pipeline
        if self.check:
            n_pos_neighb = np.sum(self.both_positive)
            n_neg_neighb = np.sum(self.both_negative)
            n_pos_neighb_save = self.nbp_omp.n_neighbours_pos[self.spot_no]
            n_neg_neighb_save = self.nbp_omp.n_neighbours_neg[self.spot_no]
            message = ""
            if n_pos_neighb_save != n_pos_neighb:
                message += f"\nn_neighbours_pos calculated here was {n_pos_neighb} but value saved in Notebook " \
                           f"for spot {self.spot_no} is {n_pos_neighb_save}."
            if n_pos_neighb_save != n_pos_neighb:
                message += f"\nn_neighbours_neg calculated here was {n_neg_neighb} but value saved in Notebook " \
                           f"for spot {self.spot_no} is {n_neg_neighb_save}."
            if len(message) > 0:
                message += "\nMake sure config parameters and spot_shape have not changed.\n" \
                           "Run with check=False to get past this error."
                raise ValueError(message)

    def get_score(self, expected_image):
        # Gets the omp score by counting where sign of images match and by summing the expected image
        # in regions where sign are correct.
        both_positive = np.asarray([self.coef_images[self.gene_no] > 0, self.expected_sign_image > 0]).all(axis=0)
        both_negative = np.asarray([self.coef_images[self.gene_no] < 0, self.expected_sign_image < 0]).all(axis=0)
        n_pos_neighb = np.sum(both_positive)
        n_neg_neighb = np.sum(both_negative)
        score = omp_spot_score(self.nbp_omp, self.score_multiplier, n_neighbours_pos=n_pos_neighb,
                               n_neighbours_neg=n_neg_neighb)
        score_from_image = np.sum(expected_image[both_positive]) - np.sum(expected_image[both_negative])
        if self.check and np.abs(score - score_from_image) < self.tol:
            # Sanity check that the way we normalised top row is correct. i.e. absolute sum of top row plots equals 1.
            raise ValueError(f"Score computed from counting neighbours differs from that from summing expected_image")
        return score

    def expected_image(self):
        # Get image so that sum of all pixels where sign is correct is equal to omp spot score
        # I.e. absolute sum of hashed values is equal to omp score
        max_score = self.score_multiplier * np.sum(self.expected_sign_image == 1) + \
                    np.sum(self.expected_sign_image == -1)
        image = self.expected_sign_image.astype(float)
        max_pos_score = self.score_multiplier * np.sum(self.expected_sign_image == 1) / max_score
        max_neg_score = np.sum(self.expected_sign_image == -1) / max_score
        image[image > 0] = max_pos_score / np.sum(image == 1)  # Contribution of each pos pixel to max score
        image[image < 0] = -max_neg_score / np.sum(image == -1)  # Contribution of each neg pixel to max score
        if self.check:
            # Sanity check that the way we normalised top row is correct. i.e. absolute sum of top row plots equals 1.
            assert np.abs(np.sum(np.abs(image)) - 1) < self.tol
        return image

    def set_coef_plot_title(self):
        # Changes color if fall below score_thresh
        if self.score > self.score_thresh:
            color = 'w'
        else:
            color = 'r'
        mid_z = int((self.nz - 1) / 2)
        score_str = r"$\gamma_s = \frac{n_{neg_s} + \rho n_{pos_s}}{n_{neg_{max}} + \rho n_{pos_{max}}}$"
        self.ax[mid_z + self.nz].set_title(f"Coefficient for Gene {self.gene_no}: {self.gene_names[self.gene_no]}, "
                                           f"Score, {score_str}, = {str(np.around(self.score, 2))}", color=color)

    def add_hatching(self):
        # Add green hatching to top plot at locations which contribute to score
        for i in range(self.nz):
            pos_yx = np.where(self.both_positive[:,:,i])
            neg_yx = np.where(self.both_negative[:, :, i])
            for j in range(len(pos_yx[0])):
                # y coordinate here a bit weird, but it seems to match up when comparing the two plots
                rectangle = plt.Rectangle((pos_yx[1][j] + self.plot_extent[0], self.plot_extent[3] - pos_yx[0][j] - 1),
                                          1, 1, fill=False, ec="lime", linestyle=':', lw=0, hatch='...')
                self.ax[i].add_patch(rectangle)
            for j in range(len(neg_yx[0])):
                rectangle = plt.Rectangle((neg_yx[1][j] + self.plot_extent[0], self.plot_extent[3] - neg_yx[0][j] - 1),
                                          1, 1, fill=False, ec="lime", lw=0, hatch='...')
                self.ax[i].add_patch(rectangle)

    def update(self, text):
        # Changes the top row of plots to reflect new score multiplier
        try:
            score_multiplier = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nScore multiplier given, {text}, is not valid")
            score_multiplier = self.score_multiplier
        if score_multiplier < 0:
            warnings.warn("Score multiplier cannot be negative")
            score_multiplier = self.score_multiplier
        self.score_multiplier = score_multiplier
        self.text_box.set_val(np.around(score_multiplier, 2))
        # Update expected shape image to reflect new score multiplier
        expected_image_plot = self.expected_image()
        for i in range(self.nz):
            self.im[i].set_data(expected_image_plot[:, :, i])
        self.score = self.get_score(expected_image_plot)
        self.set_coef_plot_title()
        self.im[0].axes.figure.canvas.draw()
