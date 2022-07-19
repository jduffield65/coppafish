from .call_spots import ColorPlotBase
from ..spot_colors import get_spot_colors
from ..call_spots import get_spot_intensity_jax, omp_spot_score
from ..setup import Notebook
from .. import omp, utils
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import os


class view_omp(ColorPlotBase):
    def __init__(self, nb: Notebook, spot_no: int, method: str = 'omp', im_size: int = 8):
        """
        Diagnostic to show omp coefficients of all genes in neighbourhood of spot.
        Only genes for which a significant number of pixels are non-zero will be plotted.

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `omp`.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
            im_size: Radius of image to be plotted for each gene.
        """
        if not os.path.isfile(str(nb._config_file)):
            raise ValueError(f'Need access to config_file to run this diagnostic.\n'
                             f'But nb._config_file = {str(nb._config_file)} does not exist.')

        color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                            nb.basic_info.use_channels)]

        if method.lower() == 'omp':
            page_name = 'omp'
            config = nb.get_config()['thresholds']
            spot_score = omp_spot_score(nb.omp, config['score_omp_multiplier'], spot_no)
        else:
            page_name = 'ref_spots'
            spot_score = nb.ref_spots.score[spot_no]
        gene_no = nb.__getattribute__(page_name).gene_no[spot_no]
        t = nb.__getattribute__(page_name).tile[spot_no]
        spot_yxz = nb.__getattribute__(page_name).local_yxz[spot_no]

        gene_name = nb.call_spots.gene_names[gene_no]
        all_gene_names = list(nb.call_spots.gene_names) + [f'BG{i}' for i in range(nb.basic_info.n_channels)]
        n_use_channels, n_use_rounds = color_norm.shape
        spot_yxz_global = spot_yxz + nb.stitch.tile_origin[t]
        im_size = [im_size, im_size]  # Useful for debugging to have different im_size_y, im_size_x.
        # Subtlety here, may have y-axis flipped, but I think it is correct:
        # note im_yxz[1] refers to point at max_y, min_x+1, z. So when reshape and set plot_extent, should be correct.
        # I.e. im = np.zeros(49); im[1] = 1; im = im.reshape(7,7); plt.imshow(im, extent=[-0.5, 6.5, -0.5, 6.5])
        # will show the value 1 at max_y, min_x+1.
        im_yxz = np.array(np.meshgrid(np.arange(spot_yxz[0]-im_size[0], spot_yxz[0]+im_size[0]+1)[::-1],
                                      np.arange(spot_yxz[1]-im_size[1], spot_yxz[1]+im_size[1]+1), spot_yxz[2]),
                          dtype=np.int16).T.reshape(-1, 3)
        im_diameter = [2*im_size[0]+1, 2*im_size[1]+1]
        spot_colors = get_spot_colors(im_yxz, t, nb.register.transform, nb.file_names, nb.basic_info)
        spot_colors = spot_colors[np.ix_(np.arange(im_yxz.shape[0]),
                                         nb.basic_info.use_rounds, nb.basic_info.use_channels)] / color_norm
        spot_colors = jnp.asarray(spot_colors)

        # Only look at pixels with high enough intensity - same as in full pipeline
        spot_intensity = get_spot_intensity_jax(jnp.abs(spot_colors))
        config = nb.get_config()['omp']
        if nb.has_page('omp'):
            initial_intensity_thresh = nb.omp.initial_intensity_thresh
        else:
            initial_intensity_thresh = config['initial_intensity_thresh']
            if initial_intensity_thresh is None:
                initial_intensity_thresh = \
                    utils.round_any(nb.call_spots.median_abs_intensity * config['initial_intensity_thresh_auto_param'],
                                    config['initial_intensity_precision'])
            initial_intensity_thresh = \
                float(np.clip(initial_intensity_thresh, config['initial_intensity_thresh_min'],
                              config['initial_intensity_thresh_max']))

        keep = spot_intensity > initial_intensity_thresh
        bled_codes = nb.call_spots.bled_codes_ge
        n_genes = bled_codes.shape[0]
        bled_codes = jnp.asarray(bled_codes[np.ix_(np.arange(n_genes),
                                                   nb.basic_info.use_rounds, nb.basic_info.use_channels)])
        dp_norm_shift = nb.call_spots.dp_norm_shift * np.sqrt(n_use_rounds)

        dp_thresh = config['dp_thresh']
        alpha = config['alpha']
        beta = config['beta']
        max_genes = config['max_genes']
        weight_coef_fit = config['weight_coef_fit']

        all_coefs = np.zeros((spot_colors.shape[0], n_genes+nb.basic_info.n_channels))
        all_coefs[np.ix_(keep, np.arange(n_genes))], \
        all_coefs[np.ix_(keep, np.array(nb.basic_info.use_channels) + n_genes)] = \
            omp.get_all_coefs(spot_colors[keep], bled_codes, nb.call_spots.background_weight_shift, dp_norm_shift,
                              dp_thresh, alpha, beta, max_genes, weight_coef_fit)
        n_nonzero_pixels_thresh = np.min([im_size[0], 5])  # If 5 pixels non-zero, plot that gene
        plot_genes = np.where(np.sum(all_coefs != 0, axis=0) > n_nonzero_pixels_thresh)[0]
        coef_images = [all_coefs[:, g].reshape(im_diameter[0], im_diameter[1]) for g in plot_genes]
        n_plot = len(plot_genes)
        # at most n_max_rows rows
        if n_plot <= 16:
            n_max_rows = 4
        else:
            n_max_rows = int(np.ceil(np.sqrt(n_plot)))
        n_cols = int(np.ceil(n_plot / n_max_rows))
        subplot_row_columns = [int(np.ceil(n_plot / n_cols)), n_cols]
        fig_size = np.clip([n_cols+5, subplot_row_columns[0]+4], 3, 12)
        subplot_adjust = [0.05, 0.775, 0.05, 0.91]
        super().__init__(coef_images, None, subplot_row_columns, subplot_adjust=subplot_adjust, fig_size=fig_size,
                         cbar_pos=[0.9, 0.05, 0.03, 0.86], slider_pos=[0.85, 0.05, 0.01, 0.86])
        # set x, y coordinates to be those of the global coordinate system
        plot_extent = [im_yxz[:, 1].min()-0.5+nb.stitch.tile_origin[t, 1],
                       im_yxz[:, 1].max()+0.5+nb.stitch.tile_origin[t, 1],
                       im_yxz[:, 0].min()-0.5+nb.stitch.tile_origin[t, 0],
                       im_yxz[:, 0].max()+0.5+nb.stitch.tile_origin[t, 0]]
        for i in range(self.n_images):
            # Add cross-hair
            self.ax[i].axes.plot([spot_yxz_global[1], spot_yxz_global[1]], [plot_extent[2], plot_extent[3]],
                                 'k', linestyle=":", lw=1)
            self.ax[i].axes.plot([plot_extent[0], plot_extent[1]], [spot_yxz_global[0], spot_yxz_global[0]],
                                 'k', linestyle=":", lw=1)
            self.im[i].set_extent(plot_extent)
            self.ax[i].tick_params(labelbottom=False, labelleft=False)
            # Add title
            title_text = f'{plot_genes[i]}: {all_gene_names[plot_genes[i]]}'
            if plot_genes[i] >= n_genes:
                text_color = (0.7, 0.7, 0.7)  # If background, make grey
                title_text = all_gene_names[plot_genes[i]]
            elif plot_genes[i] == gene_no:
                text_color = 'g'
            else:
                text_color = 'w'  # TODO: maybe make color same as used in plot for each gene
            self.ax[i].set_title(title_text, color=text_color)
        plt.subplots_adjust(hspace=0.32)
        plt.suptitle(f'OMP gene coefficients for spot {spot_no} (match'
                     f' {np.round(spot_score, decimals=2)} to {gene_name})',
                     x=(subplot_adjust[0] + subplot_adjust[1]) / 2, size=13)
        self.change_norm()
        plt.show()
