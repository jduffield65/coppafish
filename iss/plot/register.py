import numpy as np
from ..stitch import compute_shift
# TODO: find_spots/base uses jax but not for functions of interest - what to do??
from ..find_spots import spot_yxz, get_isolated_points
from ..pcr import get_single_affine_transform
from ..no_jax.spot_colors import apply_transform
from .stitch import view_shifts
from ..setup import Notebook
from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons
from scipy.spatial import KDTree


def view_initial_shift(nb: Notebook, t: int, r: int, c: Optional[int] = None,
                       return_shift: bool = False) -> Optional[np.ndarray]:
    """
    Function to plot results of exhaustive search to find shift between `ref_round/ref_channel` and
    round `r`, channel `c` for tile `t`. This shift will then be used as the starting point when running point cloud
    registration to find affine transform.
    Useful for debugging the `register_initial` section of the pipeline.

    Args:
        nb: Notebook containing results of the experiment. Must contain `find_spots` page.
        t: tile interested in.
        r: Want to find the shift between the reference round and this round.
        c: Want to find the shift between the reference channel and this channel. If `None`, `config['shift_channel']`
            will be used, as it is in the pipeline.
        return_shift: If True, will return shift found and will not call plt.show() otherwise will return None.

    Returns:
        `best_shift` - `float [shift_y, shift_x, shift_z]`.
            Best shift found. `shift_z` is in units of z-pixels.
    """
    config = nb.get_config()['register_initial']
    if c is None:
        c = config['shift_channel']
        if c is None:
            c = nb.basic_info.ref_channel
    if not np.isin(c, nb.basic_info.use_channels):
        raise ValueError(f"c should be in nb.basic_info.use_channels, but value given is\n"
                         f"{c} which is not in use_channels = {nb.basic_info.use_channels}.")

    coords = ['y', 'x', 'z']
    shifts = [{}]
    for i in range(len(coords)):
        shifts[0][coords[i]] = np.arange(config['shift_min'][i],
                                         config['shift_max'][i] +
                                         config['shift_step'][i] / 2, config['shift_step'][i]).astype(int)
    if not nb.basic_info.is_3d:
        config['shift_widen'][2] = 0  # so don't look for shifts in z direction
        config['shift_max_range'][2] = 0
        shifts[0]['z'] = np.array([0], dtype=int)
    shifts = shifts * nb.basic_info.n_rounds  # get one set of shifts for each round
    c_ref = nb.basic_info.ref_channel
    r_ref = nb.basic_info.ref_round
    # to convert z coordinate units to xy pixels when calculating distance to nearest neighbours
    z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
    print(f'Finding shift between round {r_ref}, channel {c_ref} to round {r}, channel {c} for tile {t}')
    shift, shift_score, shift_score_thresh, debug_info = \
        compute_shift(spot_yxz(nb.find_spots.spot_details, t, r_ref, c_ref),
                      spot_yxz(nb.find_spots.spot_details, t, r, c),
                      config['shift_score_thresh'], config['shift_score_thresh_multiplier'],
                      config['shift_score_thresh_min_dist'], config['shift_score_thresh_max_dist'],
                      config['neighb_dist_thresh'], shifts[r]['y'], shifts[r]['x'], shifts[r]['z'],
                      config['shift_widen'], config['shift_max_range'], z_scale,
                      config['nz_collapse'], config['shift_step'][2])
    title = f'Shift between r={r_ref}, c={c_ref} and r={r}, c={c} for tile {t}. YXZ Shift = {shift}.'
    if return_shift:
        show = False
    else:
        show = True
    view_shifts(debug_info['shifts_2d'], debug_info['scores_2d'], debug_info['shifts_3d'],
                debug_info['scores_3d'], shift, shift_score_thresh, title, show)
    if return_shift:
        return shift


class view_point_clouds:
    def __init__(self, point_clouds: List, pc_labels: List, neighb_dist_thresh: float = 5, z_scale: float = 1,
                 super_title: Optional[str] = None):
        """
        Plots two point clouds. point_clouds[0] always plotted but you can change the second point cloud using
        radio buttons.

        Args:
            point_clouds: List of point clouds, each of which is yxz coordinates float [n_points x 3].
                point_clouds[0] is always plotted.
                YX coordinates are in units of yx pixels. Z coordinates are in units of z pixels.
                Radio buttons used to select other point_cloud plotted.
            pc_labels: List of labels to appear in legend/radio-buttons for each point cloud.
                Must provide one for each point_cloud.
            neighb_dist_thresh: If distance between neighbours is less than this, a white line will connect them.
            z_scale: pixel_size_z / pixel_size_y i.e. used to convert z coordinates from z-pixels to yx pixels.
            super_title: Optional title for plot
        """
        n_point_clouds = len(point_clouds)
        if len(point_clouds) != len(pc_labels):
            raise ValueError(f'There are {n_point_clouds} but {len(pc_labels)} labels')
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
        self.z_planes = np.arange(point_clouds[0][:, 2].min(), point_clouds[0][:, 2].max() + 1)
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
        self.ax.set_ylim(np.clip(point_clouds[0][:, 0].min() - 5, 0, np.inf), point_clouds[0][:, 0].max() + 5)
        self.ax.set_xlim(np.clip(point_clouds[0][:, 1].min() - 5, 0, np.inf), point_clouds[0][:, 1].max() + 5)

        if self.nz > 1:
            # If 3D, add text box to change number of z-planes collapsed onto single plane
            # and add scrolling to change z-plane
            self.ax.set_title(f'Z = {int(self.z)}', size=10)
            self.fig.canvas.mpl_connect('scroll_event', self.z_scroll)
            text_ax = self.fig.add_axes([0.8, 0.095, 0.15, 0.04])
            self.text_box = TextBox(text_ax, 'Z-Thick', self.z_thick, color='k', hovercolor=[0.2, 0.2, 0.2])
            self.text_box.cursor.set_color('r')
            # change text box title to be above not to the left of box
            label = text_ax.get_children()[0]  # label is a child of the TextBox axis
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
        plt.show()

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
        self.ax.set_title(f"Z = {int(self.z)}", size=10)
        self.ax.figure.canvas.draw()


def view_pcr(nb: Notebook, t: int, r: int, c: int):
    """
    Function to plot results of point cloud registration to find affine transform between `ref_round/ref_channel` and
    round `r`, channel `c` for tile `t`.
    Useful for debugging the `register` section of the pipeline.

    Args:
        nb: Notebook containing results of the experiment. Must contain `find_spots` page.
            If contains register_initial_debug and/or register pages, then transform from these will be used.
        t: tile interested in.
        r: Want to find the transform between the reference round and this round.
        c: Want to find the transform between the reference channel and this channel.
    """
    # TODO: if used regularisation i.e. if transform_outlier non-zero, add another transform which is the non-regularised
    #  i.e. transform_outlier
    config = nb.get_config()
    if nb.basic_info.is_3d:
        neighb_dist_thresh = config['register']['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['register']['neighb_dist_thresh_2d']
    z_scale = [1, 1, nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy]
    point_clouds = []
    # 1st point cloud is imaging one as does not change
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r, c)]
    r_ref = nb.basic_info.ref_round
    c_ref = nb.basic_info.ref_channel
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r_ref, c_ref)]
    for i in range(2):
        # only keep isolated spots, those whose second neighbour is far away
        isolated = get_isolated_points(point_clouds[i] * z_scale, 2 * neighb_dist_thresh)
        point_clouds[i] = point_clouds[i][isolated]
    z_scale = z_scale[2]
    # Add shifted reference point cloud
    if nb.has_page('register_initial_debug'):
        shift = nb.register_initial_debug.shift[t, r]
    else:
        shift = view_initial_shift(nb, t, r, return_shift=True)
    point_clouds = point_clouds + [point_clouds[1] + shift]

    # Add reference point cloud transformed by an affine transform
    if nb.has_page('register'):
        transform = nb.register.transform[t, r, c]
    else:
        transform = get_single_affine_transform(config['register'], point_clouds[1], point_clouds[0], z_scale, z_scale,
                                                shift, neighb_dist_thresh)[0]
    point_clouds = point_clouds + [apply_transform(point_clouds[1], transform, nb.basic_info.tile_centre, z_scale)]
    pc_labels = [f'Imaging: r{r}, c{c}', f'Reference: r{r_ref}, c{c_ref}', f'Reference: r{r_ref}, c{c_ref} - Shift',
                 f'Reference: r{r_ref}, c{c_ref} - Affine']
    view_point_clouds(point_clouds, pc_labels, neighb_dist_thresh, z_scale,
                      f'Transform of tile {t} to round {r}, channel {c}')
