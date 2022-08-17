import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons
from scipy.spatial import KDTree
from ...find_spots import spot_yxz
from ...call_spots import get_non_duplicate
from ...setup import Notebook
import warnings
import numpy_indexed
from typing import List, Optional
plt.style.use('dark_background')


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
        self.fig, self.ax = plt.subplots(1, 1, figsize=(15, 8))
        subplots_adjust = [0.07, 0.775, 0.095, 0.89]
        self.fig.subplots_adjust(left=subplots_adjust[0], right=subplots_adjust[1], bottom=subplots_adjust[2],
                                 top=subplots_adjust[3])

        # Find neighbours between all point clouds and the first.
        self.neighb = [None]
        n_matches_str = 'Number Of Matches'
        for i in range(1, n_point_clouds):
            tree = KDTree(point_clouds[i] * [1, 1, z_scale])
            dist, neighb = tree.query(point_clouds[0] * [1, 1, z_scale],
                                      distance_upper_bound=neighb_dist_thresh * 3)
            neighb[dist > neighb_dist_thresh] = -1  # set too large distance neighb_ind to -1.
            n_matches_str = n_matches_str + f'\n- {pc_labels[i]}: {np.sum(neighb >= 0)}'
            self.neighb = self.neighb + [neighb]

        # Add text box to indicate number of matches between first point cloud and each of the others.
        n_matches_ax = self.fig.add_axes([0.8, subplots_adjust[3] - 0.75, 0.15, 0.2])
        plt.axis('off')
        n_matches_ax.text(0.05, 0.95, n_matches_str)

        #  round z to closest z-plane for plotting
        for i in range(n_point_clouds):
            point_clouds[i][:, 2] = np.rint(point_clouds[i][:, 2])

        # get yxz axis limits taking account all point clouds
        pc_min_lims = np.zeros((n_point_clouds, 3))
        pc_max_lims = np.zeros_like(pc_min_lims)
        for i in range(n_point_clouds):
            pc_min_lims[i] = np.min(point_clouds[i], axis=0)
            pc_max_lims[i] = np.max(point_clouds[i], axis=0)

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
            plt.axis('off')
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
    they are closest to).
    These will be removed in the `get_reference_spots` step of the pipeline so we don't double count the same spot.

    Args:
        nb: *Notebook* containing at least `stitch` page.
    """
    is_ref = np.all((nb.find_spots.spot_details[:, 1] == nb.basic_info.ref_round,
                     nb.find_spots.spot_details[:, 2] == nb.basic_info.ref_channel), axis=0)
    local_yxz = nb.find_spots.spot_details[is_ref, -3:]
    tile = nb.find_spots.spot_details[is_ref, 0]
    local_yxz = local_yxz[np.isin(tile, nb.basic_info.use_tiles)]
    tile = tile[np.isin(tile, nb.basic_info.use_tiles)]

    # find duplicate spots as those detected on a tile which is not tile centre they are closest to
    tile_origin = nb.stitch.tile_origin
    not_duplicate = get_non_duplicate(tile_origin, nb.basic_info.use_tiles, nb.basic_info.tile_centre,
                                      local_yxz, tile)

    global_yxz = local_yxz + nb.stitch.tile_origin[tile]
    global_yxz[:, 2] = np.rint(global_yxz[:, 2])  # make z coordinate an integer
    config = nb.get_config()['stitch']
    neighb_dist_thresh = config['neighb_dist_thresh']
    z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy

    point_clouds = [global_yxz[not_duplicate], global_yxz[np.invert(not_duplicate)]]
    pc_labels = ['Not Duplicate', 'Duplicate']
    if nb.has_page('ref_spots'):
        # Add point cloud indicating those spots that were not saved in ref_spots page
        # because they were shifted outside the tile bounds on at least one round/channel
        # and thus spot_color could not be found.
        local_yxz = local_yxz[not_duplicate]  # Want to find those which were not duplicates but still removed
        tile = tile[not_duplicate]
        local_yxz_saved = nb.ref_spots.local_yxz
        tile_saved = nb.ref_spots.tile
        local_yxz_saved = local_yxz_saved[np.isin(tile_saved, nb.basic_info.use_tiles)]
        tile_saved = tile_saved[np.isin(tile_saved, nb.basic_info.use_tiles)]
        global_yxz_ns = np.zeros((0, 3)) # not saved in ref_spots spots
        for t in nb.basic_info.use_tiles:
            missing_ind = -100
            local_yxz_t = local_yxz[tile == t]
            removed_ind = np.where(numpy_indexed.indices(local_yxz_saved[tile_saved == t], local_yxz_t,
                                                         missing=missing_ind) == missing_ind)[0]
            global_yxz_ns = np.append(global_yxz_ns, local_yxz_t[removed_ind] + nb.stitch.tile_origin[t], axis=0)
        global_yxz_ns[:, 2] = np.rint(global_yxz_ns[:, 2])
        point_clouds += [global_yxz_ns]
        pc_labels += ["No Spot Color"]

    # Sometimes can be empty point cloud, so remove these
    use_pc = [len(pc) > 0 for pc in point_clouds]
    pc_labels = [pc_labels[i] for i in range(len(use_pc)) if use_pc[i]]
    point_clouds= [point_clouds[i] for i in range(len(use_pc)) if use_pc[i]]
    vpc = view_point_clouds(point_clouds, pc_labels, neighb_dist_thresh, z_scale,
                            "Reference Spots in the Global Coordinate System")

    tile_sz = nb.basic_info.tile_sz
    for t in nb.basic_info.use_tiles:
        rect = matplotlib.patches.Rectangle((tile_origin[t, 1], tile_origin[t, 0]), tile_sz, tile_sz,
                                            linewidth=1, edgecolor='w', facecolor='none', linestyle=':')
        vpc.ax.add_patch(rect)
        vpc.ax.text(tile_origin[t, 1] + 20, tile_origin[t, 0] + 20, f"Tile {t}",
                    size=6, color='w', ha='left', weight='light')
    plt.show()
