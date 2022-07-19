import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.ndimage
from ..stitch import compute_shift
from ..find_spots import spot_yxz
from ..pipeline.stitch import get_shifts_to_search
from ..setup import Notebook
import warnings
from typing import Tuple, List, Optional


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
        - `image` - `float [(extent[2]-extent[3]) x (extent[1]-extent[0]) x (extent[5]-extent[4])]`.
            Imaging containing the scores at coordinates indicated by shifts.
        - `extent` - `float [min_x_shift-0.5, max_x_shift, max_y_shift, min_y_shift-0.5, min_z_shift-0.5, max_z_shift]`.
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
                score_thresh: Optional[float] = None, title: Optional[str] = None, show: bool = True):
    """
    Function to plot scores indicating number of neighbours between 2 point clouds corresponding to particular shifts
    applied to one of them. I.e. you can use this to view the output from iss/stitch/shift/compute_shift function.

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
        score_thresh: Threshold returned by `compute_shift` function, if `score` is above this,
            it indicates an accepted shift. If given, a red-white-blue colorbar will be used with white corresponding
            to `score_thresh`.
        title: Title to show.
        show: If `True`, will call `plt.show()`, else will `return fig`.
    """
    image_2d, extent_2d = get_plot_images_from_shifts(np.rint(shifts_2d).astype(int), scores_2d)
    image_2d = interpolate_array(image_2d, 0)  # replace 0 with nearest neighbor value
    fig = plt.figure()
    if score_thresh is None:
        score_thresh = (image_2d.min() + image_2d.max()) / 2
        cmap = 'virids'
    else:
        cmap = 'bwr'
    v_max = np.max([image_2d.max(), 1.2 * score_thresh])
    v_min = image_2d.min()
    if shifts_3d is not None:
        images_3d, extent_3d = get_plot_images_from_shifts(np.rint(shifts_3d).astype(int), scores_3d)
        images_3d = interpolate_array(images_3d, 0)  # replace 0 with nearest neighbor value
        if images_3d.max() > v_max:
            v_max = images_3d.max()
        if images_3d.min() < v_min:
            v_min = images_3d.min()
        cmap_norm = matplotlib.colors.TwoSlopeNorm(vmin=v_min, vcenter=score_thresh, vmax=v_max)
        n_cols = images_3d.shape[2]
        if n_cols > 13:
            # If loads of z-planes, just show the 13 with the largest score
            n_cols = 13
            max_score_z = images_3d.max(axis=(0, 1))
            max_score_z_thresh = max_score_z[np.argpartition(max_score_z, -n_cols)[-n_cols]]
            use_z = np.where(max_score_z >= max_score_z_thresh)[0]
        else:
            use_z = np.arange(n_cols)

        plot_2d_height = int(np.ceil(n_cols / 4))
        plot_3d_height = n_cols - plot_2d_height
        ax_2d = plt.subplot2grid(shape=(n_cols, n_cols), loc=(0, 0), colspan=n_cols, rowspan=plot_3d_height)
        ax_3d = [plt.subplot2grid(shape=(n_cols, n_cols), loc=(plot_3d_height + 1, i), rowspan=plot_2d_height) for i in
                 range(n_cols)]
        for i in range(n_cols):
            # share axes for 3D plots
            ax_3d[i].get_shared_y_axes().join(ax_3d[i], *ax_3d)
            ax_3d[i].get_shared_x_axes().join(ax_3d[i], *ax_3d)
            ax_3d[i].imshow(images_3d[:, :, use_z[i]], extent=extent_3d[:4], aspect='auto', cmap=cmap, norm=cmap_norm)
            z_plane = int(np.rint(extent_3d[4] + use_z[i] + 0.5))
            ax_3d[i].set_title(f'Z = {z_plane}')
            if i > 0:
                ax_3d[i].tick_params(labelbottom=False, labelleft=False)
            if best_shift is not None:
                if z_plane == best_shift[2]:
                    ax_3d[i].plot(best_shift[1], best_shift[0], 'kx')

        fig.supxlabel('X')
        fig.supylabel('Y')
    else:
        n_cols = 1
        ax_2d = plt.subplot2grid(shape=(n_cols, n_cols), loc=(0, 0))
        ax_2d.set_xlabel('X')
        ax_2d.set_ylabel('Y')
        cmap_norm = matplotlib.colors.TwoSlopeNorm(vmin=v_min, vcenter=score_thresh, vmax=v_max)

    im_2d = ax_2d.imshow(image_2d, extent=extent_2d, aspect='auto', cmap=cmap, norm=cmap_norm)
    if best_shift is not None:
        ax_2d.plot(best_shift[1], best_shift[0], 'kx')
    ax_2d.invert_yaxis()
    if title is None:
        title = 'Approx number of neighbours found for all shifts'
    ax_2d.set_title(title)
    fig.subplots_adjust(left=0.07, right=0.85, bottom=0.07, top=0.95)
    cbar_ax = fig.add_axes([0.9, 0.07, 0.03, 0.9])  # left, bottom, width, height
    fig.colorbar(im_2d, cax=cbar_ax)
    if show:
        plt.show()
    else:
        return fig


def view_stitch(nb: Notebook, t: int, direction: Optional[str] = None):
    """
    Function to plot results of exhaustive search to find overlap between tile `t` and its neighbours.
    Useful for debugging the stitch section of the pipeline.

    Args:
        nb: Notebook containing results of the experiment. Must contain `find_spots` page.
        t: Want to look at overlap between tile `t` and its south/west neighbour.
        direction: Direction of overlap interested in - either `'south'` or `'west'`.
            If `None`, then will look at both directions.
    """
    if direction is None:
        directions = ['south', 'west']
    elif direction.lower() == 'south' or direction.lower() == 'west':
        directions = [direction.lower()]
    else:
        raise ValueError(f"direction must be either 'south' or 'west' but {direction} given.")

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
            print(f'Finding shift between tiles {t} and {t_neighb[j][0]} ({j} overlap)')
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
            title = f'Overlap between t={t} and neighbor in {j} (t={t_neighb[j][0]}). YXZ Shift = {shift}.'
            fig = fig + [view_shifts(debug_info['shifts_2d'], debug_info['scores_2d'], debug_info['shifts_3d'],
                                     debug_info['scores_3d'], shift, score_thresh, title, False)]
    if len(fig) > 0:
        plt.show()
    else:
        warnings.warn(f"Tile {t} has no overlapping tiles in nb.basic_info.use_tiles.")
