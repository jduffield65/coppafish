import numpy as np
import matplotlib
import scipy.ndimage
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons
from scipy.spatial import KDTree
from ..stitch import compute_shift, get_shifts_to_search
from ..find_spots import spot_yxz
from ..call_spots import get_non_duplicate
from ..setup import Notebook
import warnings
from typing import Tuple, List, Optional

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


def view_stitch_search(nb: Notebook, t: int, direction: Optional[str] = None):
    """
    Function to plot results of exhaustive search to find overlap between tile `t` and its neighbours.
    Useful for debugging the `stitch` section of the pipeline.

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
                                     debug_info['scores_3d'], shift, score_thresh, title, False)]
    if len(fig) > 0:
        plt.show()
    else:
        warnings.warn(f"Tile {t} has no overlapping tiles in nb.basic_info.use_tiles.")


class view_point_clouds:
    def __init__(self, point_clouds: List, pc_labels: List, neighb_dist_thresh: float = 5, z_scale: float = 1,
                 super_title: Optional[str] = None):
        """
        Plots two point clouds. `point_clouds[0]` always plotted but you can change the second point cloud using
        radio buttons.

        Args:
            point_clouds: List of point clouds, each of which is yxz coordinates `float [n_points x 3]`.
                `point_clouds[0]` is always plotted.
                YX coordinates are in units of yx pixels. Z coordinates are in units of z pixels.
                Radio buttons used to select other point_cloud plotted.
            pc_labels: List of labels to appear in legend/radio-buttons for each point cloud.
                Must provide one for each `point_cloud`.
            neighb_dist_thresh: If distance between neighbours is less than this, a white line will connect them.
            z_scale: `pixel_size_z / pixel_size_y` i.e. used to convert z coordinates from z-pixels to yx pixels.
            super_title: Optional title for plot
        """
        n_point_clouds = len(point_clouds)
        if len(point_clouds) != len(pc_labels):
            raise ValueError(f'There are {n_point_clouds} point clouds but {len(pc_labels)} labels')
        self.fig, self.ax = plt.subplots(1, 1)
        subplots_adjust = [0.07, 0.775, 0.095, 0.89]
        self.fig.subplots_adjust(left=subplots_adjust[0], right=subplots_adjust[1], bottom=subplots_adjust[2],
                                 top=subplots_adjust[3])

        # Find neighbours between all point clouds and the first.
        self.neighb = [None]
        n_matches_str = 'Number Of Matches'
        for i in range(1, n_point_clouds):
            tree = KDTree(point_clouds[i] * [1, 1, z_scale])
            self.neighb = self.neighb + [tree.query(point_clouds[0] * [1, 1, z_scale],
                                                    distance_upper_bound=neighb_dist_thresh)[1]]
            self.neighb[i][self.neighb[i] == len(point_clouds[i])] = -1  # set inf distance neighb_ind to -1.
            n_matches_str = n_matches_str + f'\n- {pc_labels[i]}: {np.sum(self.neighb[i] >= 0)}'

        # Add text box to indicate number of matches between first point cloud and each of the others.
        n_matches_ax = self.fig.add_axes([0.8, subplots_adjust[3] - 0.6, 0.15, 0.2])
        plt.axis('off')
        n_matches_ax.text(0.05, 0.95, n_matches_str)

        #  round z to closest z-plane for plotting
        for i in range(n_point_clouds):
            point_clouds[i][:, 2] = np.rint(point_clouds[i][:, 2])

        # get yxz axis limits taking account all point clouds
        pc_min_lims = np.zeros((n_point_clouds, 3))
        pc_max_lims = np.zeros_like(pc_min_lims)
        for i in range(n_point_clouds):
            pc_min_lims[i] = np.min(point_clouds[i],axis=0)
            pc_max_lims[i] = np.max(point_clouds[i],axis=0)

        self.z_planes = np.arange(int(np.min(pc_min_lims[:, 2])), int(np.max(pc_max_lims[:, 2]) + 1))
        self.nz = len(self.z_planes)
        self.z_ind = 0
        self.z = self.z_planes[self.z_ind]
        self.z_thick = 0
        self.active_pc = [0, 1]
        self.in_z = [np.array([val[:, 2] >= self.z - self.z_thick, val[:, 2] <= self.z + self.z_thick]).all(axis=0)
                     for val in point_clouds]
        self.point_clouds = point_clouds
        self.pc_labels = np.array(pc_labels)
        self.pc_shapes = ['rx', 'bo']
        alpha = [1, 0.7]
        self.pc_plots = [self.ax.plot(point_clouds[self.active_pc[i]][self.in_z[i], 1],
                                      point_clouds[self.active_pc[i]][self.in_z[i], 0],
                                      self.pc_shapes[i], label=self.pc_labels[i], alpha=alpha[i])[0]
                         for i in range(2)]

        self.neighb_yx = None
        self.neighb_plot = None
        self.update_neighb_lines()

        self.ax.legend(loc='upper right')
        self.ax.set_ylabel('Y')
        self.ax.set_xlabel('X')
        self.ax.set_ylim(np.min(pc_min_lims[:, 0]), np.max(pc_max_lims[:, 0]))
        self.ax.set_xlim(np.min(pc_min_lims[:, 1]), np.max(pc_max_lims[:, 1]))

        if self.nz > 1:
            # If 3D, add text box to change number of z-planes collapsed onto single plane
            # and add scrolling to change z-plane
            self.ax.set_title(f'Z = {int(self.z)}', size=10)
            self.fig.canvas.mpl_connect('scroll_event', self.z_scroll)
            text_ax = self.fig.add_axes([0.8, 0.095, 0.15, 0.04])
        else:
            # For some reason in 2D, still need the text box otherwise buttons don't do work
            # But shift it off-screen and make small
            text_ax = self.fig.add_axes([40, 40, 0.00001, 0.00001])
        self.text_box = TextBox(text_ax, 'Z-Thick', self.z_thick, color='k', hovercolor=[0.2, 0.2, 0.2])
        self.text_box.cursor.set_color('r')
        # change text box title to be above not to the left of box
        label = text_ax.get_children()[0]  # label is a child of the TextBox axis
        if self.nz == 1:
            label.set_position([40, 40])  # shift label off-screen in 2D
        else:
            label.set_position([0.5, 2])  # [x,y] - change here to set the position
        # centering the text
        label.set_verticalalignment('top')
        label.set_horizontalalignment('center')
        self.text_box.on_submit(self.text_update)

        if n_point_clouds >= 3:
            # If 3 or more point clouds, add radio button to change the second point cloud shown.
            buttons_ax = self.fig.add_axes([0.8, subplots_adjust[3] - 0.2, 0.15, 0.2])
            self.buttons = RadioButtons(buttons_ax, self.pc_labels[1:], 0, activecolor='w')
            for i in range(n_point_clouds-1):
                self.buttons.circles[i].set_color('w')
                self.buttons.circles[i].set_color('w')
            self.buttons.set_active(0)
            self.buttons.on_clicked(self.button_update)
        if super_title is not None:
            plt.suptitle(super_title, x=(0.07 + 0.775) / 2)
        # plt.show()

    def z_scroll(self, event):
        # Scroll to change current z-plane
        if event.button == 'up':
            self.z_ind = (self.z_ind + 1) % self.nz
        else:
            self.z_ind = (self.z_ind - 1) % self.nz
        self.z = self.z_planes[self.z_ind]
        self.update()

    def text_update(self, text):
        # Use text box to change number of z-planes shown on single plane.
        # All points with z between self.z+/-self.z_thick will be shown.
        self.z_thick = int(np.rint(float(text)))
        self.update()

    def button_update(self, selected_label):
        # Use button to select the second point cloud to show.
        self.active_pc[1] = np.where(self.pc_labels == selected_label)[0][0]
        self.pc_plots[1].set_data(self.point_clouds[self.active_pc[1]][self.in_z[self.active_pc[1]], 1],
                                  self.point_clouds[self.active_pc[1]][self.in_z[self.active_pc[1]], 0])
        self.pc_plots[1].set_label(selected_label)
        self.update_neighb_lines()
        self.ax.legend(loc='upper right')
        self.ax.figure.canvas.draw()

    def update_neighb_lines(self):
        # This updates the white lines connecting neighbours between point clouds.
        for line in self.ax.get_lines():
            # Remove all current connecting lines - know these are the only symbols with '-' linestyle.
            if line.get_linestyle() == '-':
                line.remove()
        use_neighb = np.array([self.neighb[self.active_pc[1]] >= 0, self.in_z[0],
                               self.in_z[self.active_pc[1]][self.neighb[self.active_pc[1]]]]).all(axis=0)
        self.neighb_yx = [None, None]
        for i in range(2):
            self.neighb_yx[i] = \
                np.vstack([self.point_clouds[0][use_neighb, i],
                           self.point_clouds[self.active_pc[1]][self.neighb[self.active_pc[1]][use_neighb], i]])
        self.neighb_plot = self.ax.plot(self.neighb_yx[1], self.neighb_yx[0], 'w', alpha=0.5, linewidth=1)

    def update(self):
        # This updates plot to point clouds / connecting lines according to current params.
        self.in_z = [np.array([val[:, 2] >= self.z - self.z_thick, val[:, 2] <= self.z + self.z_thick]).all(axis=0)
                     for val in self.point_clouds]
        for i in range(2):
            self.pc_plots[i].set_data(self.point_clouds[self.active_pc[i]][self.in_z[self.active_pc[i]], 1],
                                      self.point_clouds[self.active_pc[i]][self.in_z[self.active_pc[i]], 0])
        self.update_neighb_lines()
        if self.nz > 1:
            self.ax.set_title(f"Z = {int(self.z)}", size=10)
        self.ax.figure.canvas.draw()


def view_stitch_overlap(nb: Notebook, t: int, direction: str = 'south'):
    """
    This plots point clouds of neighbouring tiles with:

    * No overlap
    * Initial guess at shift using `config['stitch']['expected_overlap']`
    * Overlap determined in stitch stage of the pipeline (using `nb.stitch.south_shifts` or `nb.stitch.west_shifts`)
    * Their final global coordinate system positions (using `nb.stitch.tile_origin`)

    Args:
        nb: *Notebook* containing at least `stitch` page.
        t: Want to look at overlap between tile `t` and its north or east neighbour.
        direction: Direction of overlap interested in - either `'south'`/`'north'` or `'west'`/`'east'`.
    """
    # NOTE that directions should actually be 'north' and 'east'
    if direction.lower() == 'south' or direction.lower() == 'west':
        direction = direction.lower()
    elif direction.lower() == 'north':
        direction = 'south'
    elif direction.lower() == 'east':
        direction = 'west'
    else:
        raise ValueError(f"direction must be either 'south' or 'west' but {direction} given.")

    direction_label = {'south': 'north', 'west': 'east'}  # label refers to actual direction
    if direction == 'south':
        t_neighb = np.where(np.sum(nb.basic_info.tilepos_yx == nb.basic_info.tilepos_yx[t, :] + [1, 0],
                                   axis=1) == 2)[0]
        if t_neighb not in nb.basic_info.use_tiles:
            warnings.warn(f"Tile {t} has no overlapping tiles in the south direction so changing to west.")
            direction = 'west'
        else:
            no_overlap_shift = np.array([-nb.basic_info.tile_sz, 0, 0])  # assuming no overlap between tiles
            found_shift = nb.stitch.south_shifts[np.where(nb.stitch.south_pairs[:,0] == t)[0]][0]
    if direction == 'west':
        t_neighb = np.where(np.sum(nb.basic_info.tilepos_yx == nb.basic_info.tilepos_yx[t, :] + [0, 1],
                                   axis=1) == 2)[0]
        if t_neighb not in nb.basic_info.use_tiles:
            raise ValueError(f"Tile {t} has no overlapping tiles in the west direction.")
        no_overlap_shift = np.array([0, -nb.basic_info.tile_sz, 0])  # assuming no overlap between tiles
        found_shift = nb.stitch.west_shifts[np.where(nb.stitch.west_pairs[:, 0] == t)[0]][0]

    config = nb.get_config()['stitch']
    t_neighb = t_neighb[0]
    r = nb.basic_info.ref_round
    c = nb.basic_info.ref_channel
    point_clouds = []
    # add global coordinates of neighbour tile as point cloud that is always present.
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t_neighb, r, c) +
                                   nb.stitch.tile_origin[t_neighb]]

    local_yxz_t = spot_yxz(nb.find_spots.spot_details, t, r, c)
    # Add point cloud for tile t assuming no overlap
    point_clouds = point_clouds + [local_yxz_t + nb.stitch.tile_origin[t_neighb] + no_overlap_shift]
    # Add point cloud assuming expected overlap
    initial_shift = (1-config['expected_overlap']) * no_overlap_shift
    point_clouds = point_clouds + [local_yxz_t + nb.stitch.tile_origin[t_neighb] + initial_shift]
    # Add point cloud for tile t with found shift
    point_clouds = point_clouds + [local_yxz_t + nb.stitch.tile_origin[t_neighb] + found_shift]
    # Add point cloud for tile t in global coordinate system
    point_clouds = point_clouds + [local_yxz_t + nb.stitch.tile_origin[t]]

    neighb_dist_thresh = config['neighb_dist_thresh']
    z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
    pc_labels = [f'Tile {t_neighb}', f'Tile {t} - No overlap',
                 f"Tile {t} - {int(config['expected_overlap']*100)}% overlap", f'Tile {t} - Shift', f'Tile {t} - Final']
    view_point_clouds(point_clouds, pc_labels, neighb_dist_thresh, z_scale,
                      f'Overlap between tile {t} and tile {t_neighb} in the {direction_label[direction]}')
    plt.show()


def view_stitch(nb: Notebook):
    """
    This plots all the reference spots found (`ref_round`/`ref_channel`) in the global coordinate system created
    in the `stitch` stage of the pipeline.

    It also indicates which of these spots are duplicates (detected on a tile which is not the tile whose centre
    they are closest to) to be removed in the `get_reference_spots` step of the pipeline.

    Args:
        nb: *Notebook* containing at least `stitch` page.
    """
    is_ref = np.all((nb.find_spots.spot_details[:, 1] == nb.basic_info.ref_round,
                     nb.find_spots.spot_details[:, 2] == nb.basic_info.ref_channel), axis=0)
    local_yxz = nb.find_spots.spot_details[is_ref, -3:]
    tile = nb.find_spots.spot_details[is_ref, 0]

    # find duplicate spots as those detected on a tile which is not tile centre they are closest to
    tile_origin = nb.stitch.tile_origin
    not_duplicate = get_non_duplicate(tile_origin, nb.basic_info.use_tiles, nb.basic_info.tile_centre,
                                      local_yxz, tile)

    global_yxz = local_yxz + nb.stitch.tile_origin[tile]
    global_yxz[:, 2] = np.rint(global_yxz[:, 2])  # make z coordinate an integer
    config = nb.get_config()['stitch']
    neighb_dist_thresh = config['neighb_dist_thresh']
    z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
    vpc = view_point_clouds([global_yxz[not_duplicate], global_yxz[np.invert(not_duplicate)]],
                            ['Not Duplicate', 'Duplicate'], neighb_dist_thresh, z_scale,
                            "Reference Spots in the Global Coordinate System")

    tile_sz = nb.basic_info.tile_sz
    for t in nb.basic_info.use_tiles:
        rect = matplotlib.patches.Rectangle((tile_origin[t, 1], tile_origin[t, 0]), tile_sz, tile_sz,
                                            linewidth=1, edgecolor='g', facecolor='none', linestyle=':')
        vpc.ax.add_patch(rect)
        vpc.ax.text(tile_origin[t, 1] + 20, tile_origin[t, 0] + 20, f"Tile {t}",
                    size=6, color='g', ha='left', weight='light')
    plt.show()
