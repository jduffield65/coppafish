from ..call_spots.spot_colors import ColorPlotBase
from ..call_spots.dot_product import view_score
from ..call_spots.background import view_background
from .track_fit import get_track_info
from ...spot_colors.base import get_spot_colors
from ...call_spots import omp_spot_score, get_spot_intensity
from ...setup import Notebook
from ...omp.coefs import get_all_coefs
from ...omp.base import get_initial_intensity_thresh
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple
plt.style.use('dark_background')


def get_coef_images(nb: Notebook, spot_no: int, method, im_size: List[int]) -> Tuple[np.ndarray, List[float],
                                                                                     List[float]]:
    """
    Gets image of $yxz$ dimension `(2*im_size[0]+1) x (2*im_size[1]+1) x (2*im_size[2]+1)` of the coefficients
    fitted by omp for each gene.

    Args:
        nb: *Notebook* containing experiment details. Must have run at least as far as `call_reference_spots`.
        spot_no: Spot of interest to get gene coefficient images for.
        method: `'anchor'` or `'omp'`.
            Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
        im_size: $yxz$ radius of image to get for each gene.

    Returns:
        `coef_images` - `float16 [n_genes x (2*im_size[0]+1) x (2*im_size[1]+1) x (2*im_size[2]+1)]`.
            Image for each gene, axis order is $gyxz$.
            `coef_images[g, 0, 0, 0]` refers to coefficient of gene g at `global_yxz = min_global_yxz`.
            `coef_images[g, -1, -1, -1]` refers to coefficient of gene g at `global_yxz = max_global_yxz`.
        `min_global_yxz` - `float [3]`. Min $yxz$ coordinates of image in global coordinates.
        `max_global_yxz` - `float [3]`. Max $yxz$ coordinates of image in global coordinates.
    """
    color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                        nb.basic_info.use_channels)]

    if method.lower() == 'omp':
        page_name = 'omp'
    else:
        page_name = 'ref_spots'
    t = nb.__getattribute__(page_name).tile[spot_no]
    spot_yxz = nb.__getattribute__(page_name).local_yxz[spot_no]

    # Subtlety here, may have y-axis flipped, but I think it is correct:
    # note im_yxz[1] refers to point at max_y, min_x+1, z. So when reshape and set plot_extent, should be correct.
    # I.e. im = np.zeros(49); im[1] = 1; im = im.reshape(7,7); plt.imshow(im, extent=[-0.5, 6.5, -0.5, 6.5])
    # will show the value 1 at max_y, min_x+1.
    im_yxz = np.array(np.meshgrid(np.arange(spot_yxz[0] - im_size[0], spot_yxz[0] + im_size[0] + 1)[::-1],
                                  np.arange(spot_yxz[1] - im_size[1], spot_yxz[1] + im_size[1] + 1),
                                  spot_yxz[2]),
                      dtype=np.int16).T.reshape(-1, 3)
    z = np.arange(-im_size[2], im_size[2]+1)
    im_yxz = np.vstack([im_yxz + [0, 0, val] for val in z])
    im_diameter_yx = [2 * im_size[0] + 1, 2 * im_size[1] + 1]
    spot_colors = get_spot_colors(im_yxz, t, nb.register.transform, nb.file_names, nb.basic_info) / color_norm

    # Only look at pixels with high enough intensity - same as in full pipeline
    spot_intensity = get_spot_intensity(np.abs(spot_colors))
    config = nb.get_config()['omp']
    if nb.has_page('omp'):
        initial_intensity_thresh = nb.omp.initial_intensity_thresh
    else:
        initial_intensity_thresh = get_initial_intensity_thresh(config, nb.call_spots)

    keep = spot_intensity > initial_intensity_thresh
    bled_codes = nb.call_spots.bled_codes_ge
    n_genes = bled_codes.shape[0]
    bled_codes = np.asarray(bled_codes[np.ix_(np.arange(n_genes),
                                              nb.basic_info.use_rounds, nb.basic_info.use_channels)])
    n_use_rounds = len(nb.basic_info.use_rounds)
    dp_norm_shift = nb.call_spots.dp_norm_shift * np.sqrt(n_use_rounds)

    dp_thresh = config['dp_thresh']
    if method.lower() == 'omp':
        alpha = config['alpha']
        beta = config['beta']
    else:
        config_call_spots = nb.get_config()['call_spots']
        alpha = config_call_spots['alpha']
        beta = config_call_spots['beta']
    max_genes = config['max_genes']
    weight_coef_fit = config['weight_coef_fit']

    all_coefs = np.zeros((spot_colors.shape[0], n_genes + nb.basic_info.n_channels))
    all_coefs[np.ix_(keep, np.arange(n_genes))], \
    all_coefs[np.ix_(keep, np.array(nb.basic_info.use_channels) + n_genes)] = \
        get_all_coefs(spot_colors[keep], bled_codes, nb.call_spots.background_weight_shift, dp_norm_shift,
                      dp_thresh, alpha, beta, max_genes, weight_coef_fit)

    n_genes = all_coefs.shape[1]
    nz = len(z)
    coef_images = np.zeros((n_genes, len(z), im_diameter_yx[0], im_diameter_yx[1]))
    for g in range(n_genes):
        ind = 0
        for z in range(nz):
            coef_images[g, z] = all_coefs[ind:ind+np.prod(im_diameter_yx), g].reshape(im_diameter_yx[0],
                                                                                      im_diameter_yx[1])
            ind += np.prod(im_diameter_yx)
    coef_images = np.moveaxis(coef_images, 1, -1)  # move z index to end
    min_global_yxz = im_yxz.min(axis=0)+nb.stitch.tile_origin[t]
    max_global_yxz = im_yxz.max(axis=0)+nb.stitch.tile_origin[t]
    return coef_images.astype(np.float16), min_global_yxz, max_global_yxz


class view_omp(ColorPlotBase):
    def __init__(self, nb: Notebook, spot_no: int, method: str = 'omp', im_size: int = 8):
        """
        Diagnostic to show omp coefficients of all genes in neighbourhood of spot.
        Only genes for which a significant number of pixels are non-zero will be plotted.

        !!! warning "Requires access to `nb.file_names.tile_dir`"

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
            im_size: Radius of image to be plotted for each gene.
        """
        coef_images, min_global_yxz, max_global_yxz = get_coef_images(nb, spot_no, method, [im_size, im_size, 0])

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
        spot_yxz_global = spot_yxz + nb.stitch.tile_origin[t]
        n_genes = nb.call_spots.bled_codes_ge.shape[0]

        n_nonzero_pixels_thresh = np.min([im_size, 5])  # If 5 pixels non-zero, plot that gene
        plot_genes = np.where(np.sum(coef_images != 0, axis=(1, 2, 3)) > n_nonzero_pixels_thresh)[0]
        coef_images = coef_images[plot_genes, :, :, 0]
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
        plot_extent = [min_global_yxz[1]-0.5, max_global_yxz[1]+0.5,
                       min_global_yxz[0]-0.5, max_global_yxz[0]+0.5]
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
                     f' {str(np.around(spot_score, 2))} to {gene_name})',
                     x=(subplot_adjust[0] + subplot_adjust[1]) / 2, size=13)
        self.change_norm()
        plt.show()


class view_omp_fit(ColorPlotBase):
    def __init__(self, nb: Notebook, spot_no: int, method: str = 'omp', dp_thresh: Optional[float] = None,
                 max_genes: Optional[int] = None):
        """
        Diagnostic to run omp on a single pixel and see which genes fitted at which iteration.
        Right-clicking on a particular bled code will cause *coppafish.plot.call_spots.dot_product.view_score*
        to run, indicating how the dot product calculation for that iteration was performed.

        Left-clicking on background image will cause coppafish.plot.call_spots.background.view_background to run,
        indicating how the dot product calculation for performed.

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
            spot_no: Spot of interest to be plotted.
            method: `'anchor'` or `'omp'`.
                Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
            dp_thresh: If None, will use value in omp section of config file.
            max_genes: If None, will use value in omp section of config file.
        """
        track_info, bled_codes, dp_thresh = get_track_info(nb, spot_no, method, dp_thresh, max_genes)
        # Add info so can call view_dot_product
        self.nb = nb
        self.track_info = track_info
        self.bled_codes = bled_codes
        self.dp_thresh = dp_thresh
        self.spot_no = spot_no
        self.fitting_method = method
        n_genes, n_use_rounds, n_use_channels = bled_codes.shape

        n_residual_images = track_info['residual'].shape[0]
        residual_images = [track_info['residual'][i].transpose() for i in range(n_residual_images)]
        background_image = np.zeros((n_use_rounds, n_use_channels))
        for c in range(n_use_channels):
            background_image += track_info['background_codes'][c] * track_info['background_coefs'][c]
        background_image = background_image.transpose()

        # allow for possibly adding background vector
        # TODO: Think may get index error if best gene ever was background_vector.
        bled_codes = np.append(bled_codes, track_info['background_codes'], axis=0)
        all_gene_names = list(nb.call_spots.gene_names) + [f'BG{i}' for i in nb.basic_info.use_channels]
        gene_images = [bled_codes[track_info['gene_added'][i]].transpose() *
                       track_info['coef'][i][track_info['gene_added'][i]] for i in range(2, n_residual_images)]
        all_images = residual_images + [background_image] + gene_images

        # Plot all images
        subplot_adjust = [0.06, 0.82, 0.075, 0.9]
        super().__init__(all_images, None, [2, n_residual_images], subplot_adjust=subplot_adjust, fig_size=(15, 7))

        # label axis
        self.ax[0].set_yticks(ticks=np.arange(self.im_data[0].shape[0]), labels=nb.basic_info.use_channels)
        self.ax[0].set_xticks(ticks=np.arange(self.im_data[0].shape[1]), labels=nb.basic_info.use_rounds)
        self.fig.supxlabel('Round', size=12, x=(subplot_adjust[0] + subplot_adjust[1]) / 2)
        self.fig.supylabel('Color Channel', size=12)
        plt.suptitle(f'Residual at each iteration of OMP for Spot {spot_no}. DP Threshold = {dp_thresh}',
                     x=(subplot_adjust[0] + subplot_adjust[1]) / 2)

        # Add titles for each subplot
        titles = ['Initial', 'Post Background']
        for g in track_info['gene_added'][2:]:
            titles = titles + ['Post ' + all_gene_names[g]]
        for i in range(n_residual_images):
            titles[i] = titles[i] + '\nRes = {:.2f}'.format(np.linalg.norm(residual_images[i]))
        titles = titles + ['Background']
        for i in range(2, n_residual_images):
            g = track_info['gene_added'][i]
            titles = titles + [f'{g}: {all_gene_names[g]}']
            titles[-1] = titles[-1] + '\nDP = {:.2f}'.format(track_info['dot_product'][i])

        # Make title red if dot product fell below dp_thresh or if best gene background
        is_fail_thresh = False
        for i in range(self.n_images):
            if np.isin(i, np.arange(2, n_residual_images)):
                # failed if bad dot product, gene added is background or gene added has previously been added
                is_fail_thresh = np.abs(track_info['dot_product'][i]) < dp_thresh or \
                                 track_info['gene_added'][i] >= n_genes or \
                                 np.isin(track_info['gene_added'][i], track_info['gene_added'][2:i])
                if is_fail_thresh:
                    text_color = 'r'
                else:
                    text_color = 'w'
            elif i == self.n_images - 1 and is_fail_thresh:
                text_color = 'r'
            else:
                text_color = 'w'
            self.ax[i].set_title(titles[i], size=8, color=text_color)

        # Add rectangles where added gene is intense
        for i in range(len(gene_images)):
            gene_coef = track_info['coef'][i+2][track_info['gene_added'][i+2]]
            intense_gene_cr = np.where(np.abs(gene_images[i] / gene_coef) > self.intense_gene_thresh)
            for j in range(len(intense_gene_cr[0])):
                for k in [i+1, i+1+n_residual_images]:
                    # can't add rectangle to multiple axes hence second for loop
                    rectangle = plt.Rectangle((intense_gene_cr[1][j]-0.5, intense_gene_cr[0][j]-0.5), 1, 1,
                                              fill=False, ec="g", linestyle=':', lw=2)
                    self.ax[k].add_patch(rectangle)

        self.change_norm()
        self.fig.canvas.mpl_connect('button_press_event', self.show_calc)
        self.track_info = track_info
        plt.show()

    def show_calc(self, event):
        x_click = event.x
        y_click = event.y
        if event.button.name == 'RIGHT':
            # If right click anywhere, it will show dot product calculation for the iteration clicked on.
            n_iters = len(self.track_info['gene_added']) - 2
            iter_x_coord = np.zeros(n_iters)
            for i in range(n_iters):
                iter_x_coord[i] = np.mean(self.ax[i+1].bbox.extents[:3:2])
            iter = np.argmin(np.abs(iter_x_coord - x_click))
            view_score(self.nb, self.spot_no, self.fitting_method, iter=iter,
                       omp_fit_info=[self.track_info, self.bled_codes, self.dp_thresh])
        else:
            # If click on background plot, it will show background calculation
            if y_click < self.ax[0].bbox.extents[1] and x_click < self.ax[1].bbox.extents[0]:
                view_background(self.nb, self.spot_no, self.fitting_method,
                                track_info=[self.track_info, None])
