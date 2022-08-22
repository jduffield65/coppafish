import warnings
from typing import Tuple, List, Optional
import matplotlib
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from ...setup import Notebook
from ...find_spots import spot_yxz
from ...stitch import get_shifts_to_search, compute_shift
plt.style.use('dark_background')


def interpolate_array(array: np.ndarray, invalid_value: float) -> np.ndarray:
    """
    Values in the `array` which are equal to `invalid_value` will be replaced by the value of the
    nearest valid `array` element.

    Args:
        array: n-dimensional array to interpolate.
        invalid_value: Value indicating where to interpolate `array`.

    Returns:
        interpolated array

    """
    ind = scipy.ndimage.distance_transform_edt(array == invalid_value, return_distances=False, return_indices=True)
    return array[tuple(ind)]


def get_plot_images_from_shifts(shifts: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, List]:
    """
    This converts a sequence of scores corresponding to particular `yx(z)` shifts to a 2(3) dimensional image
    which can then be plotted through `plt.imshow(image[:, :, 0], extent=extent[:4])` with y/x axis indicating
    the correct y/x shift for each score.

    Args:
        shifts: `int [n_shifts x 3 (or 2)]`.
            `shifts[i]` is the `yx(z)` shift which achieved `scores[i]`.
            YX shift is in units of YX pixels. Z shift is in units of z-pixels.
        scores: `float [n_shifts]`.
            `scores[i]` is the score corresponding to `shifts[i]`. It is approximately the number of neighbours
            between the two point clouds after the shift was applied.

    Returns:
        `image` - `float [(extent[2]-extent[3]) x (extent[1]-extent[0]) x (extent[5]-extent[4])]`.
            Imaging containing the scores at coordinates indicated by shifts.
        `extent` - `float [min_x_shift-0.5, max_x_shift, max_y_shift, min_y_shift-0.5, min_z_shift-0.5, max_z_shift]`.
            Indicates the shifts corresponding to the extremities of image.
            (Note the -0.5 added to min_shift is so value appears at centre of pixel).
    """
    min_yxz = np.min(shifts, axis=0)
    shifts_mod = shifts - min_yxz
    im_size = shifts_mod.max(axis=0) + 1
    image = np.zeros(im_size)
    image[tuple([shifts_mod[:, i] for i in range(im_size.size)])] = scores
    extent_ax_order = [1, 0]  # x is 1st index in im_show(extent=...)
    if im_size.size == 3:
        extent_ax_order = extent_ax_order + [2]
    extent = [[min_yxz[i] - 0.5, min_yxz[i] + im_size[i]] for i in extent_ax_order]
    extent = sum(extent, [])
    # 1st y extent needs to be max and 2nd min as second index indicates top row which corresponds to min_y
    # for our image.
    extent[2], extent[3] = extent[3], extent[2]
    return image, extent


def view_shifts(shifts_2d: np.ndarray, scores_2d: np.ndarray, shifts_3d: Optional[np.ndarray] = None,
                scores_3d: Optional[np.ndarray] = None, best_shift: Optional[np.ndarray] = None,
                score_thresh_2d: Optional[float] = None, best_shift_initial: Optional[np.ndarray] = None,
                score_thresh_3d: Optional[float] = None, thresh_shift: Optional[np.ndarray] = None,
                thresh_min_dist: Optional[int] = None,
                thresh_max_dist: Optional[int] = None, title: Optional[str] = None, show: bool = True):
    """
    Function to plot scores indicating number of neighbours between 2 point clouds corresponding to particular shifts
    applied to one of them. I.e. you can use this to view the output from
    `coppafish/stitch/shift/compute_shift` function.

    Args:
        shifts_2d: `int [n_shifts_2d x 2]`.
            `shifts_2d[i]` is the yx shift which achieved `scores_2d[i]` when considering just yx shift between
            point clouds.
            I.e. first step of finding optimal shift is collapsing 3D point cloud to just a few planes and then
            applying a yx shift to these planes.
        scores_2d: `float [n_shifts_2d]`.
            `scores_2d[i]` is the score corresponding to `shifts_2d[i]`. It is approximately the number of neighbours
            between the two point clouds after the shift was applied.
        shifts_3d: `int [n_shifts_3d x 3]`.
            `shifts_3d[i]` is the yxz shift which achieved `scores_3d[i]` when considering the yxz shift between
            point clouds. YX shift is in units of YX pixels. Z shift is in units of z-pixels.
            If None, only 2D image plotted.
        scores_3d: `float [n_shifts_3d]`.
            `scores_3d[i]` is the score corresponding to `shifts_3d[i]`. It is approximately the number of neighbours
            between the two point clouds after the shift was applied.
        best_shift: `int [y_shift, x_shift, z_shift]`.
            Best shift found by algorithm. YX shift is in units of YX pixels. Z shift is in units of z-pixels.
            Will be plotted as black cross on image if provided.
        score_thresh_2d: Threshold returned by `compute_shift` function for 2d calculation, if `score` is above this,
            it indicates an accepted 2D shift. If given, a red-white-blue colorbar will be used with white corresponding
            to `score_thresh_2d` in the 2D plot
        best_shift_initial: `int [y_shift, x_shift]`.
            Best yx shift found by in first search of algorithm. I.e. `score_thresh` computation based on this.
            Will show as green x if given.
        score_thresh_3d: Threshold returned by `compute_shift` function for 3d calculation. If given, a red-white-blue
            colorbar will be used with white corresponding to `score_thresh_3d` in the 3D plots.
        thresh_shift: `int [y_shift, x_shift, z_shift]`.
            yx shift corresponding to `score_thresh`. Will show as green + in both 2D and 3D plots if given.
        thresh_min_dist: `shift_thresh` is the shift with the max score in an annulus a distance between
            `thresh_min_dist` and `thresh_max_dist` away from `best_shift_initial`.
            Annulus will be shown in green if given.
        thresh_max_dist: `shift_thresh` is the shift with the max score in an annulus a distance between
            `thresh_min_dist` and `thresh_max_dist` away from `best_shift_initial`.
            Annulus will be shown in green if given.
        title: Title to show.
        show: If `True`, will call `plt.show()`, else will `return fig`.
    """
    image_2d, extent_2d = get_plot_images_from_shifts(np.rint(shifts_2d).astype(int), scores_2d)
    image_2d = interpolate_array(image_2d, 0)  # replace 0 with nearest neighbor value
    fig = plt.figure(figsize=(12, 8))
    if score_thresh_2d is None:
        score_thresh_2d = (image_2d.min() + image_2d.max()) / 2
        cmap_2d = 'virids'
    else:
        cmap_2d = 'bwr'
    v_max = np.max([image_2d.max(), 1.2 * score_thresh_2d])
    v_min = image_2d.min()
    if cmap_2d == 'bwr':
        cmap_extent = np.max([v_max - score_thresh_2d, score_thresh_2d - v_min])
        # Have equal range above and below score_thresh to not skew colormap
        v_min = score_thresh_2d - cmap_extent
        v_max = score_thresh_2d + cmap_extent
    cmap_norm = matplotlib.colors.TwoSlopeNorm(vmin=v_min, vcenter=score_thresh_2d, vmax=v_max)
    if shifts_3d is not None:
        images_3d, extent_3d = get_plot_images_from_shifts(np.rint(shifts_3d).astype(int), scores_3d)
        images_3d = interpolate_array(images_3d, 0)  # replace 0 with nearest neighbor value
        if score_thresh_3d is None:
            score_thresh_3d = (images_3d.min() + images_3d.max()) / 2
            cmap_3d = 'virids'
        else:
            cmap_3d = 'bwr'
        v_max_3d = np.max([images_3d.max(), 1.2 * score_thresh_3d])
        v_min_3d = images_3d.min()
        if cmap_3d == 'bwr':
            cmap_extent = np.max([v_max_3d - score_thresh_3d, score_thresh_3d - v_min_3d])
            # Have equal range above and below score_thresh to not skew colormap
            v_min_3d = score_thresh_3d - cmap_extent
            v_max_3d = score_thresh_3d + cmap_extent
        cmap_norm_3d = matplotlib.colors.TwoSlopeNorm(vmin=v_min_3d, vcenter=score_thresh_3d, vmax=v_max_3d)
        n_cols = images_3d.shape[2]
        if n_cols > 13:
            # If loads of z-planes, just show the 13 with the largest score
            n_cols = 13
            max_score_z = images_3d.max(axis=(0, 1))
            use_z = np.sort(np.argsort(max_score_z)[::-1][:n_cols])
        else:
            use_z = np.arange(n_cols)

        plot_3d_height = int(np.ceil(n_cols / 4))
        plot_2d_height = n_cols - plot_3d_height
        ax_2d = plt.subplot2grid(shape=(n_cols, n_cols), loc=(0, 0), colspan=n_cols, rowspan=plot_2d_height)
        ax_3d = [plt.subplot2grid(shape=(n_cols, n_cols), loc=(plot_2d_height + 1, i), rowspan=plot_3d_height) for i in
                 range(n_cols)]
        for i in range(n_cols):
            # share axes for 3D plots
            ax_3d[i].get_shared_y_axes().join(ax_3d[i], *ax_3d)
            ax_3d[i].get_shared_x_axes().join(ax_3d[i], *ax_3d)
            im_3d = ax_3d[i].imshow(images_3d[:, :, use_z[i]], extent=extent_3d[:4], aspect='auto', cmap=cmap_3d,
                                    norm=cmap_norm_3d)
            z_plane = int(np.rint(extent_3d[4] + use_z[i] + 0.5))
            ax_3d[i].set_title(f'Z = {z_plane}')
            if thresh_shift is not None and z_plane == thresh_shift[2]:
                # Indicate threshold shift on correct 3d plot
                ax_3d[i].plot(thresh_shift[1], thresh_shift[0], '+', color='lime', label='Thresh shift')
            if i > 0:
                ax_3d[i].tick_params(labelbottom=False, labelleft=False)
            if best_shift is not None:
                if z_plane == best_shift[2]:
                    ax_3d[i].plot(best_shift[1], best_shift[0], 'kx')

        fig.supxlabel('X')
        fig.supylabel('Y')
        ax_3d[0].invert_yaxis()
        cbar_gap = 0.05
        cbar_3d_height = plot_3d_height / (plot_3d_height + plot_2d_height)
        cbar_2d_height = plot_2d_height / (plot_3d_height + plot_2d_height)
        cbar_ax = fig.add_axes([0.9, 0.07, 0.03, cbar_3d_height - 2*cbar_gap])  # left, bottom, width, height
        fig.colorbar(im_3d, cax=cbar_ax)
        cbar_ax.set_ylim(np.clip(v_min_3d, 0, v_max_3d), v_max_3d)
    else:
        n_cols = 1
        ax_2d = plt.subplot2grid(shape=(n_cols, n_cols), loc=(0, 0))
        ax_2d.set_xlabel('X')
        ax_2d.set_ylabel('Y')

    im_2d = ax_2d.imshow(image_2d, extent=extent_2d, aspect='auto', cmap=cmap_2d, norm=cmap_norm)
    if best_shift is not None:
        ax_2d.plot(best_shift[1], best_shift[0], 'kx', label='Best shift')
    if best_shift_initial is not None and best_shift_initial != best_shift:
        ax_2d.plot(best_shift_initial[1], best_shift_initial[0], 'x', color='lime', label='Best shift initial')
    if thresh_shift is not None:
        ax_2d.plot(thresh_shift[1], thresh_shift[0], '+', color='lime', label='Thresh shift')
    if thresh_min_dist is not None and best_shift_initial is not None:
        ax_2d.add_patch(plt.Circle((best_shift_initial[1], best_shift_initial[0]), thresh_min_dist, color='lime',
                                   fill=False))
    if thresh_max_dist is not None and best_shift_initial is not None:
        ax_2d.add_patch(plt.Circle((best_shift_initial[1], best_shift_initial[0]), thresh_max_dist, color='lime',
                                   fill=False))
    ax_2d.invert_yaxis()
    if title is None:
        title = 'Approx number of neighbours found for all shifts'
    ax_2d.set_title(title)
    ax_2d.legend(facecolor='b')
    fig.subplots_adjust(left=0.07, right=0.85, bottom=0.07, top=0.95)
    if shifts_3d is None:
        cbar_ax = fig.add_axes([0.9, 0.07, 0.03, 0.9])  # left, bottom, width, height
    else:
        cbar_ax = fig.add_axes([0.9, 0.07 + cbar_3d_height, 0.03, cbar_2d_height - 2.5 * cbar_gap])
    fig.colorbar(im_2d, cax=cbar_ax)
    cbar_ax.set_ylim(np.clip(v_min, 0, v_max), v_max)
    if show:
        plt.show()
    else:
        return fig


def view_stitch_search(nb: Notebook, t: int, direction: Optional[str] = None):
    """
    Function to plot results of exhaustive search to find overlap between tile `t` and its neighbours.
    Useful for debugging the `stitch` section of the pipeline.
    White in the color plot refers to the value of `score_thresh` for this search.

    Args:
        nb: Notebook containing results of the experiment. Must contain `find_spots` page.
        t: Want to look at overlap between tile `t` and its north/east neighbour.
        direction: Direction of overlap interested in - either `'south'`/`'north'` or `'west'`/`'east'`.
            If `None`, then will look at both directions.
    """
    # NOTE that directions should actually be 'north' and 'east'
    if direction is None:
        directions = ['south', 'west']
    elif direction.lower() == 'south' or direction.lower() == 'west':
        directions = [direction.lower()]
    elif direction.lower() == 'north':
        directions = ['south']
    elif direction.lower() == 'east':
        directions = ['west']
    else:
        raise ValueError(f"direction must be either 'south' or 'west' but {direction} given.")
    direction_label = {'south': 'north', 'west': 'east'}  # label refers to actual direction

    config = nb.get_config()['stitch']
    # determine shifts to search over
    shifts = get_shifts_to_search(config, nb.basic_info)
    if not nb.basic_info.is_3d:
        config['nz_collapse'] = None
        config['shift_widen'][2] = 0  # so don't look for shifts in z direction
        config['shift_max_range'][2] = 0

    # find shifts between overlapping tiles
    c = nb.basic_info.ref_channel
    r = nb.basic_info.ref_round
    t_neighb = {'south': [], 'west': []}
    z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy

    # align to south neighbour followed by west neighbour
    t_neighb['south'] = np.where(np.sum(nb.basic_info.tilepos_yx == nb.basic_info.tilepos_yx[t, :] + [1, 0],
                                        axis=1) == 2)[0]
    t_neighb['west'] = np.where(np.sum(nb.basic_info.tilepos_yx == nb.basic_info.tilepos_yx[t, :] + [0, 1],
                                       axis=1) == 2)[0]
    fig = []
    for j in directions:
        if t_neighb[j] in nb.basic_info.use_tiles:
            print(f'Finding shift between tiles {t} and {t_neighb[j][0]} ({direction_label[j]} overlap)')
            shift, score, score_thresh, debug_info = \
                compute_shift(spot_yxz(nb.find_spots.spot_details, t, r, c),
                              spot_yxz(nb.find_spots.spot_details, t_neighb[j][0], r, c),
                              config['shift_score_thresh'],
                              config['shift_score_thresh_multiplier'],
                              config['shift_score_thresh_min_dist'],
                              config['shift_score_thresh_max_dist'],
                              config['neighb_dist_thresh'], shifts[j]['y'],
                              shifts[j]['x'], shifts[j]['z'],
                              config['shift_widen'], config['shift_max_range'],
                              z_scale, config['nz_collapse'],
                              config['shift_step'][2])
            title = f'Overlap between t={t} and neighbor in {direction_label[j]} (t={t_neighb[j][0]}). ' \
                    f'YXZ Shift = {shift}.'
            fig = fig + [view_shifts(debug_info['shifts_2d'], debug_info['scores_2d'], debug_info['shifts_3d'],
                                     debug_info['scores_3d'], shift, debug_info['min_score_2d'],
                                     debug_info['shift_2d_initial'],
                                     score_thresh, debug_info['shift_thresh'], config['shift_score_thresh_min_dist'],
                                     config['shift_score_thresh_max_dist'], title, False)]
    if len(fig) > 0:
        plt.show()
    else:
        warnings.warn(f"Tile {t} has no overlapping tiles in nb.basic_info.use_tiles.")
