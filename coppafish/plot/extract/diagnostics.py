import matplotlib.pyplot as plt
import distinctipy
import numpy as np
from ...setup import Notebook
from ...call_spots import color_normalisation
from matplotlib.widgets import Button

plt.style.use('dark_background')


def thresh_box_plots(nb: Notebook):
    """
    Function to plot distribution of auto_threshold values amongst tiles for each round and channel.

    Args:
        nb: Notebook containing the extract NotebookPage.
    """
    box_data = [nb.extract.auto_thresh[:, r, c] for c in nb.basic_info.use_channels for r in nb.basic_info.use_rounds]
    if nb.basic_info.use_anchor:
        box_data = box_data + [nb.extract.auto_thresh[:, nb.basic_info.anchor_round, nb.basic_info.anchor_channel]]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.15)
    n_use_channels = len(nb.basic_info.use_channels)
    # different boxplot color for each channel (+ different color for anchor channel)
    # Must be distinct from black and white
    channel_colors = distinctipy.get_colors(n_use_channels + int(nb.basic_info.use_anchor), [(0, 0, 0), (1, 1, 1)])
    bp = ax1.boxplot(box_data, notch=0, sym='+', patch_artist=True)

    c = -1
    tick_labels = np.tile(nb.basic_info.use_rounds, n_use_channels).tolist()
    leg_labels = nb.basic_info.use_channels
    if nb.basic_info.use_anchor:
        tick_labels = tick_labels + ['Anchor']
        leg_labels = leg_labels + [f'Anchor ({nb.basic_info.anchor_channel})']
    ax1.set_xticklabels(tick_labels)
    if nb.basic_info.use_anchor:
        ticks = ax1.get_xticklabels()
        ticks[-1].set_rotation(90)

    leg_markers = []
    for i in range(len(box_data)):
        if i % n_use_channels == 0:
            c += 1
            leg_markers = leg_markers + [bp['boxes'][i]]
        bp['boxes'][i].set_facecolor(channel_colors[c])
    ax1.legend(leg_markers, leg_labels, title='Channel')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Auto Threshold')
    ax1.set_title('Boxplots showing distribution of Auto Threshold amongst tiles for each round and channel')
    plt.show()


def resample_histogram(hist_values_old: np.ndarray, hist_counts_old: np.ndarray,
                       hist_values_new: np.ndarray) -> np.ndarray:
    """
    Resamples hist counts according to hist_values_new.

    Args:
        hist_values_old: float [n_old_values]
        hist_counts_old: int [n_old_values]
        hist_values_new: float [n_new_values]

    Returns:
        int [n_new_values] - hist_counts_new.
    """
    hist_spacing = hist_values_new[1] - hist_values_new[0]  # spacing must be constant through hist_values_new
    hist_bins_new = np.append(hist_values_new, hist_values_new[-1] + hist_spacing)
    hist_bins_new = hist_bins_new - hist_spacing / 2
    hist_counts_new = np.histogram(hist_values_old, hist_bins_new, weights=hist_counts_old)
    return hist_counts_new[0]


class histogram_plots:
    def __init__(self, nb: Notebook):
        """
        Plots histograms showing distribution of intensity values combined from all tiles for each round and channel.
        There is also a Norm button which equalises color channels so all color channels should have most intensity
        values between -1 and 1.

        In the normalised histograms, a good channel will have a sharp peak near 0 accounting for non-spot pixels
        and a long tail from around 0.1 to just beyond 1 accounting for spot pixels.

        Args:
            nb: Notebook containing the extract NotebookPage.
        """
        self.use_rounds = nb.basic_info.use_rounds
        self.use_channels = nb.basic_info.use_channels
        self.hist_values = nb.extract.hist_values
        self.hist_values_norm = np.arange(-10, 10.004, 0.005)
        n_use_rounds = len(self.use_rounds)
        n_use_channels = len(self.use_channels)
        self.fig, self.ax1 = plt.subplots(n_use_channels, n_use_rounds, figsize=(10, 6), sharey=True, sharex=True)
        self.ax1 = self.ax1.flatten()

        # Compute color_norm_factor as it will be computed at call_spots step of pipeline
        config = nb.get_config()['call_spots']
        rc_ind = np.ix_(self.use_rounds, self.use_channels)
        hist_counts_use = np.moveaxis(np.moveaxis(nb.extract.hist_counts, 0, -1)[rc_ind], -1, 0)
        self.color_norm_factor = color_normalisation(self.hist_values, hist_counts_use,
                                                     config['color_norm_intensities'],
                                                     config['color_norm_probs'], config['bleed_matrix_method'])
        self.color_norm_factor = self.color_norm_factor.T.flatten()

        i = 0
        min_value = 3  # clip hist counts to this so don't get log(0) error
        self.plot_lines = []
        self.hist_counts_norm = []
        self.hist_counts = []
        for c in self.use_channels:
            for r in self.use_rounds:
                # Clip histogram to stop log(0) error.
                self.hist_counts = self.hist_counts + [np.clip(nb.extract.hist_counts[:, r, c], min_value, np.inf)]
                self.hist_counts_norm = self.hist_counts_norm + \
                                        [resample_histogram(self.hist_values / self.color_norm_factor[i],
                                                            self.hist_counts[i], self.hist_values_norm)]

                # Normalise histograms to give probabilities
                self.hist_counts[i] = self.hist_counts[i] / np.sum(nb.extract.hist_counts[:, r, c])
                self.hist_counts_norm[i] = self.hist_counts_norm[i] / np.sum(nb.extract.hist_counts[:, r, c])
                self.plot_lines = self.plot_lines + self.ax1[i].plot(self.hist_values, self.hist_counts[i])
                if r == nb.basic_info.use_rounds[0]:
                    self.ax1[i].set_ylabel(c)
                if c == nb.basic_info.use_channels[-1]:
                    self.ax1[i].set_xlabel(r)
                i += 1
        self.ax1[0].set_yscale('log')
        self.fig.supylabel('Channel')
        self.fig.supxlabel('Round')
        plt.suptitle('Histograms showing distribution of intensity values combined from '
                     'all tiles for each round and channel')

        self.norm = False
        self.xlims_norm = [-1, 1]
        self.xlims = [-300, 300]
        self.ax1[0].set_xlim(self.xlims[0], self.xlims[1])
        self.norm_button_ax = self.fig.add_axes([0.85, 0.02, 0.1, 0.05])
        self.norm_button = Button(self.norm_button_ax, 'Norm', hovercolor='0.275')
        self.norm_button.on_clicked(self.change_norm)

        plt.show()

    def change_norm(self, event=None):
        """
        Function triggered on press of normalisation button.
        Will either multiply or divide each image by the relevant `color_norm_factor`.
        """
        if self.norm:
            self.norm = False
        else:
            self.norm = True
        for i in range(len(self.plot_lines)):
            if self.norm:
                self.plot_lines[i].set_data(self.hist_values_norm, self.hist_counts_norm[i])
            else:
                self.plot_lines[i].set_data(self.hist_values, self.hist_counts[i])
        if self.norm:
            self.ax1[0].set_xlim(self.xlims_norm[0], self.xlims_norm[1])
        else:
            self.ax1[0].set_xlim(self.xlims[0], self.xlims[1])
        self.fig.canvas.draw()
