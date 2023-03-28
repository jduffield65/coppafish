import os
import numpy as np
import distinctipy
import matplotlib.pyplot as plt
import warnings
import napari
from skimage.filters import sobel
from ...setup import Notebook
from ..stitch.diagnostics import shift_info_plot
from typing import Optional
plt.style.use('dark_background')


def view_transform_distribution(nb: Notebook, transform_type='round', regularised=False):
    # TODO: Add pre and post transform regularisation and do it for the same colorbar
    # TODO: Remove the shift from here and put these in another function that shows the shifts for a particular round or
    #  channel on an xy tile positions before and after regularisation
    # TODO: Add another, more simple diagnostic that just mimcs this but for shifts
    if transform_type == 'round':
        if not regularised:
            transform = nb.register.round_transform[:, nb.basic_info.use_rounds]
    elif transform_type == 'channel':
        transform = nb.register.channel_transform[nb.basic_info.use_channels]

    scale = np.array([transform[:, :, 0, 0].T, transform[:, :, 1, 1].T, transform[:, :, 2, 2].T])
    shift = np.array([transform[:, :, 0, 3].T, transform[:, :, 1, 3].T, transform[:, :, 2, 3].T])

    # Plot shifts on left in order z y x
    plt.subplot(321)
    plt.imshow(shift[0])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('Z-Shift', fontsize=20)
    plt.colorbar()

    plt.subplot(323)
    plt.imshow(shift[1])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('Y-Shift', fontsize=20)
    plt.colorbar()

    plt.subplot(325)
    plt.imshow(shift[2])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('X-Shift', fontsize=20)
    plt.colorbar()

    # Plot scales on right in order z y x
    plt.subplot(322)
    plt.imshow(scale[0])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('Z-Scale', fontsize=20)
    plt.colorbar()

    plt.subplot(324)
    plt.imshow(scale[1])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('Y-Scale', fontsize=20)
    plt.colorbar()

    plt.subplot(326)
    plt.imshow(scale[2])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('X-Scale', fontsize=20)
    plt.colorbar()
    plt.show()


def view_scale_boxplots(nb, transform_type='round'):
    if transform_type == 'round':
        transform = nb.register.round_transform[:, nb.basic_info.use_rounds]
    elif transform_type == 'channel':
        transform = nb.register.channel_transform[nb.basic_info.use_channels]
    scale = np.array([transform[:, :, 0, 0].T, transform[:, :, 1, 1].T, transform[:, :, 2, 2].T])

    plt.subplot(311)
    plt.boxplot(scale[0].T)
    plt.title('Z-Scales', fontsize=20)
    plt.xlabel(transform_type, fontsize=20)
    plt.ylabel('scale', fontsize=20)
    plt.subplot(312)
    plt.boxplot(scale[1].T)
    plt.title('Y-Scales', fontsize=20)
    plt.xlabel(transform_type, fontsize=20)
    plt.ylabel('scale', fontsize=20)
    plt.subplot(313)
    plt.boxplot(scale[2].T)
    plt.title('X-Scales', fontsize=20)
    plt.xlabel(transform_type, fontsize=20)
    plt.ylabel('scale', fontsize=20)


def scale_box_plots(nb: Notebook):
    """
    Function to plot distribution of chromatic aberration scaling amongst tiles for each round and channel.
    Want very similar values for a given channel across all tiles and rounds for each dimension.
    Also expect $y$ and $x$ scaling to be very similar. $z$ scaling different due to unit conversion.

    Args:
        nb: *Notebook* containing the `register` and `register_debug` *NotebookPages*.
    """
    if nb.basic_info.is_3d:
        ndim = 3
        if np.ptp(nb.register.transform[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                               nb.basic_info.use_channels)][:, :, :, 2, 2]) < 1e-5:
            ndim = 2
            warnings.warn("Not showing z-scaling as all are the same")
    else:
        ndim = 2

    fig, ax = plt.subplots(ndim, figsize=(10, 6), sharex=True)
    ax[0].get_shared_y_axes().join(ax[0], ax[1])
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.15)
    y_titles = ["Scaling - Y", "Scaling - X", "Scaling - Z"]
    n_use_channels = len(nb.basic_info.use_channels)
    # different boxplot color for each channel
    # Must be distinct from black and white
    channel_colors = distinctipy.get_colors(n_use_channels, [(0, 0, 0), (1, 1, 1)])
    for i in range(ndim):
        box_data = [nb.register.transform[nb.basic_info.use_tiles, r, c, i, i] for c in nb.basic_info.use_channels
                    for r in nb.basic_info.use_rounds]
        bp = ax[i].boxplot(box_data, notch=0, sym='+', patch_artist=True)
        leg_markers = []
        c = -1
        for j in range(len(box_data)):
            if j % n_use_channels == 0:
                c += 1
                leg_markers = leg_markers + [bp['boxes'][j]]
            bp['boxes'][j].set_facecolor(channel_colors[c])
        ax[i].set_ylabel(y_titles[i])

        if i == ndim-1:
            tick_labels = np.tile(nb.basic_info.use_rounds, n_use_channels).tolist()
            leg_labels = nb.basic_info.use_channels
            ax[i].set_xticks(np.arange(len(tick_labels)))
            ax[i].set_xticklabels(tick_labels)
            ax[i].legend(leg_markers, leg_labels, title='Channel')
            ax[i].set_xlabel('Round')
    ax[0].set_title('Boxplots showing distribution of scalings due to\nchromatic aberration amongst tiles for each '
                    'round and channel')
    plt.show()


class view_affine_shift_info:
    def __init__(self, nb: Notebook, c: Optional[int] = None, outlier: bool = False):
        """
        For all affine transforms to imaging rounds/channels from the reference round computed in the `register` section
        of the pipeline, this plots the values of the shifts, `n_matches` (number of neighbours found) and
        `error` (average distance between neighbours).

        For each round and channel (channel is changed by scrolling with the mouse), there will be 3 plots:

        * y shift vs x shift for all tiles
        * z shift vs x shift for all tiles
        * `n_matches` vs `error` for all tiles

        In each case, the markers in the plots are numbers.
        These numbers indicate the tile the shift was found for.
        The number will be blue if `nb.register_debug.n_matches > nb.register_debug.n_matches_thresh` and red otherwise.

        Args:
            nb: Notebook containing at least the `register` page.
            c: If None, will give option to scroll with mouse to change channel. If specify c, will show just
                one channel with no scrolling.
            outlier: If `True`, will plot shifts from `nb.register_debug.transform_outlier` instead of
                `nb.register.transform`. In this case, only tiles for which
                `nb.register_debug.failed == True` are plotted for each round/channel.
        """
        self.outlier = outlier
        self.nb = nb
        if c is None:
            if self.outlier:
                # only show channels for which there is an outlier shift
                self.channels = np.sort(np.unique(np.where(nb.register_debug.failed)[2]))
                if len(self.channels) == 0:
                    raise ValueError(f"No outlier transforms were computed")
            else:
                self.channels = np.asarray(nb.basic_info.use_channels)
        else:
            self.channels = [c]
        self.n_channels = len(self.channels)
        self.c_ind = 0
        self.c = self.channels[self.c_ind]

        n_cols = len(nb.basic_info.use_rounds)
        if nb.basic_info.is_3d:
            n_rows = 3
        else:
            n_rows = 2
        self.fig, self.ax = plt.subplots(n_rows, n_cols, figsize=(15, 7))
        self.fig.subplots_adjust(hspace=0.4, bottom=0.08, left=0.06, right=0.97, top=0.9)
        self.shift_info = self.get_ax_lims(self.nb, self.channels, self.outlier)
        self.update()
        if self.n_channels > 1:
            self.fig.canvas.mpl_connect('scroll_event', self.z_scroll)
        plt.show()

    @staticmethod
    def get_ax_lims(nb: Notebook, channels, outlier: bool):
        # initialises shift_info with ax limits for each plot
        # Do this because want limits to remain the same as we change color channel.
        if nb.basic_info.is_3d:
            ndim = 3
            z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
        else:
            ndim = 2
        y_lim = np.zeros((ndim, 2))
        x_lim = np.zeros((ndim, 2))

        # same matches/error limits for all rounds as well as all channels
        n_matches = nb.register_debug.n_matches[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                                       channels)]
        y_lim[-1, :] = [np.clip(np.min(n_matches) - 100, 0, np.inf), np.max(n_matches) + 100]
        error = nb.register_debug.error[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                                       channels)]
        x_lim[-1, :] = [np.clip(np.min(error) - 0.1, 0, np.inf), np.max(error) + 0.1]

        if outlier:
            shifts = nb.register_debug.transform_outlier[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                                                   channels)][:, :, :, 3]
        else:
            shifts = nb.register.transform[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                                     channels)][:, :, :, 3]
        if ndim == 3:
            shifts[:, :, :, 2] = shifts[:, :, :, 2] / z_scale  # put z shift in units of z-pixels

        shift_info = {}
        for r in range(len(nb.basic_info.use_rounds)):
            name = f'Round {nb.basic_info.use_rounds[r]}'
            shift_info[name] = {}
            shift_info[name]['y_lim'] = y_lim.copy()
            shift_info[name]['x_lim'] = x_lim.copy()
            # 1st plot (Y shift vs X shift)
            shift_info[name]['y_lim'][0] = [np.min(shifts[:, r, :, 0]) - 3, np.max(shifts[:, r, :, 0]) + 3]
            shift_info[name]['x_lim'][0] = [np.min(shifts[:, r, :, 1]) - 3, np.max(shifts[:, r, :, 1]) + 3]
            if ndim == 3:
                # 2nd plot (Z shift vs X shift)
                shift_info[name]['y_lim'][1] = [np.min(shifts[:, r, :, 2]) - 1, np.max(shifts[:, r, :, 2]) + 1]
                shift_info[name]['x_lim'][1] = [np.min(shifts[:, r, :, 1]) - 3, np.max(shifts[:, r, :, 1]) + 3]
        return shift_info

    @staticmethod
    def get_shift_info(shift_info: dict, nb: Notebook, c: int, outlier: bool) -> dict:
        # Updates the shift_info dictionary to pass to shift_info_plot
        if nb.basic_info.is_3d:
            ndim = 3
            z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
        else:
            ndim = 2
        for r in nb.basic_info.use_rounds:
            name = f'Round {r}'
            shift_info[name]['tile'] = nb.basic_info.use_tiles
            if outlier:
                shift_info[name]['shift'] = nb.register_debug.transform_outlier[nb.basic_info.use_tiles, r, c, 3, :ndim]
            else:
                shift_info[name]['shift'] = nb.register.transform[nb.basic_info.use_tiles, r, c, 3, :ndim]
            if ndim == 3:
                # put z-shift in units of z-pixels
                shift_info[name]['shift'][:, 2] = shift_info[name]['shift'][:, 2] / z_scale
            shift_info[name]['n_matches'] = nb.register_debug.n_matches[nb.basic_info.use_tiles, r, c]
            shift_info[name]['n_matches_thresh'] = nb.register_debug.n_matches_thresh[nb.basic_info.use_tiles, r, c]
            if outlier:
                # Set matches to 0 if no outlier transform found so won't plot
                shift_info[name]['n_matches'][np.invert(nb.register_debug.failed[nb.basic_info.use_tiles, r, c])] = 0
            shift_info[name]['error'] = nb.register_debug.error[nb.basic_info.use_tiles, r, c]
        return shift_info

    def update(self):
        # Gets shift_info for current channel and updates plot figure.
        shift_info = self.get_shift_info(self.shift_info, self.nb, self.c, self.outlier)
        for ax in self.ax.flatten():
            ax.cla()
        if self.outlier:
            title_start = "Outlier "
        else:
            title_start = ""
        self.ax = shift_info_plot(shift_info, f"{title_start}Shifts found in register part of pipeline "
                                              f"from round {self.nb.basic_info.anchor_round}, channel "
                                              f"{self.nb.basic_info.anchor_channel} to channel "
                                              f"{self.c} for each round and tile",
                                  fig=self.fig, ax=self.ax, return_ax=True)
        self.ax[0, 0].figure.canvas.draw()

    def z_scroll(self, event):
        # Scroll to change channel shown in plots
        if event.button == 'up':
            self.c_ind = (self.c_ind + 1) % self.n_channels
        else:
            self.c_ind = (self.c_ind - 1) % self.n_channels
        self.c = self.channels[self.c_ind]
        self.update()


def view_overlay(nb, t, r, c, filter=True):
    """
    Function to overlay tile, round and channel with the anchor in napari and view the registration.
    This function is only long because we have to convert images to zyx and the transform to zyx * zyx
    Args:
        nb: Notebook
        t: common tile
        r: target image round
        c: target image channel
        filter: Boolean whether to filter
    """

    # initialise frequently used variables
    nbp_file, nbp_basic, transform = nb.file_names, nb.basic_info, nb.register.transform
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
    c_ref = nb.basic_info.anchor_channel

    # Initialise file directories
    output_dir = os.path.join(nb.file_names.output_dir, 'reg_images')
    anchor_file = 'anchor_t'+str(t)+'c'+str(c_ref)+'.npy'
    target_file = 'round'+str(r)+'_t'+str(t)+'c'+str(c)+'.npy'

    # Load in the files
    base_image = np.load(os.path.join(output_dir, anchor_file))
    target_image = np.load(os.path.join(output_dir, target_file))

    # Apply filter if needed
    if filter:
        base_image = sobel(base_image)
        target_image = sobel(target_image)

    affine_transform = change_basis(transform[t, r, c], np.array([25, 850, 850]), z_scale)

    viewer = napari.Viewer()
    viewer.add_image(base_image, blending='additive', colormap='red', affine=affine_transform)
    viewer.add_image(target_image, blending='additive', colormap='green')


# TODO: Change format of this to make it more similar to notebook outputs
def view_regression_scatter(shift, position, transform):
    """
    view 3 scatter plots for each data set shift vs positions
    Args:
        shift: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of shifts in zyx format
        position: num_tiles x num_rounds x z_sv x y_sv x x_sv x 3 array which of positions in zyx format
        transform: 3 x 4 affine transform obtained by previous robust regression
    """

    shift = shift.reshape((shift.shape[0] * shift.shape[1] * shift.shape[2], 3)).T
    position = position.reshape((position.shape[0] * position.shape[1] * position.shape[2], 3)).T

    z_range = np.arange(np.min(position[0]), np.max(position[0]))
    yx_range = np.arange(np.min(position[1]), np.max(position[1]))

    plt.subplot(1, 3, 1)
    plt.scatter(position[0], shift[0], alpha=1e2/shift.shape[1])
    plt.plot(z_range, (transform[0, 0] - 1) * z_range + transform[0,3])
    plt.title('Z-Shifts vs Z-Positions')

    plt.subplot(1, 3, 2)
    plt.scatter(position[1], shift[1], alpha=1e2 / shift.shape[1])
    plt.plot(yx_range, (transform[1, 1] - 1) * yx_range + transform[1, 3])
    plt.title('Y-Shifts vs Y-Positions')

    plt.subplot(1, 3, 3)
    plt.scatter(position[2], shift[2], alpha=1e2 / shift.shape[1])
    plt.plot(yx_range, (transform[2, 2] - 1) * yx_range + transform[2, 3])
    plt.title('X-Shifts vs X-Positions')

    plt.show()


def change_basis(A, new_origin, z_scale):
    """
    Takes in 4 x 3 yxz * yxz transform where z coord is in xy pixels and convert to 4 x 4 zyx * zyx where final
    Args:
        A: 4 x 3 yxz * yxz transform
        new_origin: new origin (zyx)
        z_scale: pixel_size_z/pixel_size_xy

    """
    # Transform saved as yxz * yxz but needs to be zyx * zyx. Convert this to something napari will understand. I think
    # this includes making the shift the final column as opposed to our convention of making the shift the final row
    affine_transform = np.vstack((A.T, np.array([0, 0, 0, 1])))

    row_shuffler = np.zeros((4, 4))
    row_shuffler[0, 1] = 1
    row_shuffler[1, 2] = 1
    row_shuffler[2, 0] = 1
    row_shuffler[3, 3] = 1

    # Now compute the affine transform, in the new basis
    affine_transform = np.linalg.inv(row_shuffler) @ affine_transform @ row_shuffler

    # z shift needs to be converted to z-pixels as opposed to yx
    affine_transform[0, 3] = affine_transform[0, 3] / z_scale

    # also add new origin conversion for shift
    affine_transform[:3, 3] += (affine_transform[:3, :3] - np.eye(3)) @ new_origin

    return affine_transform
