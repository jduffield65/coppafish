import os
import mplcursors
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
from ...setup import Notebook
from ...spot_colors.base import normalise_rc, remove_background
from ...call_spots.bleed_matrix import compute_bleed_matrix
from ..call_spots.spot_colors import view_spot
plt.style.use('dark_background')


def view_all_gene_scores(nb):
    """
    Plots the scores of all genes in a square grid. Useful for seeing the distribution of scores.
    Args:
        nb: Notebook object. Must have ref_spots page
    """
    grid_dim = int(np.ceil(np.sqrt(nb.call_spots.gene_names.shape[0])))
    fig, ax = plt.subplots(grid_dim, grid_dim, figsize=(10, 10))
    # Move subplots down to make room for the title and to the left to make room for the colourbar
    fig.subplots_adjust(top=0.9)
    fig.subplots_adjust(right=0.9)

    # We also want to plot the histograms in different colours, representing the number of spots for each gene
    for i in range(grid_dim ** 2):
        gene_i_scores = nb.ref_spots.score[nb.ref_spots.gene_no == i]
        n_spots = len(gene_i_scores)

        if n_spots < 50:
            n_bins = 10
        else:
            n_bins = 50
        # We want to choose the colour of the histogram based on the number of spots. We will use a log scale, with the
        # minimum number of spots being 1 and the maximum being 1000. Use a blue to red colourmap
        cmap = plt.get_cmap("coolwarm")
        norm = mpl.colors.Normalize(vmin=1, vmax=2000)

        # Plot the histogram of scores for each gene
        if i < nb.call_spots.gene_names.shape[0]:
            ax[i // grid_dim, i % grid_dim].hist(gene_i_scores, bins=n_bins, color=cmap(norm(n_spots)))
            ax[i // grid_dim, i % grid_dim].set_title(nb.call_spots.gene_names[i])
            ax[i // grid_dim, i % grid_dim].set_xlim(0, 1)
            ax[i // grid_dim, i % grid_dim].set_xticks([])
            ax[i // grid_dim, i % grid_dim].set_yticks([])

        # Next we want to delete the empty plots
        else:
            ax[i // grid_dim, i % grid_dim].axis("off")
            ax[i // grid_dim, i % grid_dim].set_xticks([])
            ax[i // grid_dim, i % grid_dim].set_yticks([])

    # Add overall title and colorbar
    fig.suptitle("Distribution of Scores for Each Gene", fontsize=16)
    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, label="Number of Spots")
    # Put label on the left of the colourbar
    cax.yaxis.set_label_position("left")
    plt.show()


def view_spot_brightness_hists(nb: Notebook):
    """
    Simple viewer of spot brightness histograms.
    Args:
        nb: Notebook
    """
    use_channels, n_rounds = nb.basic_info.use_channels, nb.basic_info.n_rounds
    spot_colours_no_bg = nb.ref_spots.colors[:, :, use_channels] - \
                         np.repeat(nb.ref_spots.background_strength[:, np.newaxis, :], n_rounds, axis=1)
    initial_bleed = nb.call_spots.initial_bleed_matrix[:, use_channels]
    _, spot_brightness = normalise_rc(spot_colours_no_bg, initial_bleed)
    n_rounds = len(spot_brightness)
    n_channels = len(spot_brightness[0])
    fig, axes = plt.subplots(nrows=n_rounds, ncols=n_channels, figsize=(10, 10))
    max_brightness = np.zeros(n_channels)
    min_brightness = np.zeros(n_channels)

    # convert to log scale and get max brightness for each channel (99th percentile) to set x-axis limits
    for r in range(n_rounds):
        for c in range(n_channels):
            spot_brightness[r][c] = np.array(spot_brightness[r][c])
            spot_brightness[r][c] = np.log(spot_brightness[r][c][spot_brightness[r][c] > 0])
    for c in range(n_channels):
        max_brightness[c] = np.percentile(np.concatenate([spot_brightness[r][c] for r in range(n_rounds)]), 99)
        min_brightness[c] = np.percentile(np.concatenate([spot_brightness[r][c] for r in range(n_rounds)]), 1)

    # We want to color each histogram by the number of spots in that histogram
    max_spots = np.max([len(spot_brightness[r][c]) for r in range(n_rounds) for c in range(n_channels)])

    for r in range(n_rounds):
        for c in range(n_channels):
            # plot histogram of spot_brightness[r][c] on axes[r, c]
            cmap = plt.get_cmap("coolwarm")
            norm = mpl.colors.Normalize(vmin=1, vmax=max_spots)
            axes[r, c].hist(spot_brightness[r][c], bins=np.linspace(min_brightness[c], max_brightness[c], 100),
                            color=cmap(norm(len(spot_brightness[r][c]))))
            # set x_ticks to 0, max_brightness for the final round only, otherwise []
            if r == n_rounds - 1:
                axes[r, c].set_xticks([0, np.round(max_brightness[c], 1)])
            else:
                axes[r, c].set_xticks([])
            # set y_ticks to []
            axes[r, c].set_yticks([])
            # would also like to add red vertical dotted line at the median spot brightness
            median = np.median(spot_brightness[r][c])
            axes[r, c].axvline(median, color='r', linestyle='dotted')
            # add label in top right corner of each plot with median spot brightness
            axes[r, c].text(0.95, 0.95, '{:.2f}'.format(median), color='r',
                            horizontalalignment='right', verticalalignment='top', transform=axes[r, c].transAxes,
                            fontsize=8)

    # Add labels to each row and column for round and channel
    for ax, col in zip(axes[0], use_channels):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], range(n_rounds)):
        ax.set_ylabel(row)

    # Add label for rows and label for columns
    fig.text(0.5, 0.04, 'Channels', ha='center')
    fig.text(0.04, 0.5, 'Rounds', va='center', rotation='vertical')

    # Add a colorbar on the right side of the figure
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # Now add a colorbar with the label 'Number of spots in each histogram'
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Number of spots in each histogram')

    # Add a title
    fig.suptitle("Spot brightness histograms")

    plt.show()

# We are now going to create a new class that will allow us to view the spots used to calculate the gene efficiency
# for a given gene. This will be useful for checking that the spots used are representative of the gene as a whole.
class GEViewer():
    def __init__(self, nb: Notebook):
        """
        Diagnostic to show the n_genes x n_rounds gene efficiency matrix as a heatmap.
        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
        """
        self.nb = nb
        self.n_genes = nb.call_spots.gene_efficiency.shape[0]
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        gene_efficiency = nb.call_spots.gene_efficiency
        self.ax.imshow(gene_efficiency, cmap='viridis', vmin=0, vmax=gene_efficiency.max(), aspect='auto')
        self.ax.set_xlabel('Round')
        self.ax.set_ylabel('Gene')
        self.ax.set_xticks(ticks=np.arange(gene_efficiency.shape[1]))
        self.ax.set_yticks([])

        # add colorbar
        self.ax.set_title('Gene Efficiency')
        cax = self.fig.add_axes([0.95, 0.1, 0.03, 0.8])
        cbar = self.fig.colorbar(self.ax.images[0], cax=cax)
        cbar.set_label('Gene Efficiency')

        # Adding gene names to y-axis would be too crowded. We will use mplcursors to show gene name of gene[r] when
        # hovering over row r of the heatmap. This means we need to only extract the y position of the mouse cursor.
        gene_names = nb.call_spots.gene_names
        mplcursors.cursor(self.ax, hover=True).connect("add", lambda sel: self.plot_gene_name(sel.target.index[0]))
        # 2. Allow genes to be selected by clicking on them
        mplcursors.cursor(self.ax, hover=False).connect("add", lambda sel: GESpotViewer(nb, sel.target.index[0]))
        # 3. We would like to add a white rectangle around the observed spot when we hover over it. We will
        # use mplcursors to do this. We need to add a rectangle to the plot when hovering over a gene.
        # We also want to turn off annotation when hovering over a gene so we will use the `hover=False` option.
        mplcursors.cursor(self.ax, hover=2).connect("add", lambda sel: self.add_rectangle(sel.target.index[0]))

        plt.show()

    def add_rectangle(self, index):
        # We need to remove any existing rectangles from the plot
        index = max(0, index)
        index = min(index, self.n_genes - 1)
        for rectangle in self.ax.patches:
            rectangle.remove()
        # We can then add a new rectangle to the plot
        self.ax.add_patch(
            Rectangle((-0.5, index - 0.5), self.nb.basic_info.n_rounds, 1, fill=False, edgecolor='white'))

    def plot_gene_name(self, index):
    # We need to remove any existing gene names from the plot
        index = max(0, index)
        index = min(index, self.n_genes - 1)
        for text in self.ax.texts:
            text.remove()
        # We can then add a new gene name to the top right of the plot in size 20 font
        self.ax.text(1.05, 1.05, self.nb.call_spots.gene_names[index], transform=self.ax.transAxes, size=20,
                        horizontalalignment='right', verticalalignment='top', color='white')


class GESpotViewer():
    def __init__(self, nb: Notebook, gene_index: int = 0):
        """
        Diagnostic to show the spots used to calculate the gene efficiency for a given gene.
        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
            iteration: iteration of call spots that we would like to view
            gene_index: Index of gene to be plotted.
        """
        self.nb = nb
        self.gene_index = gene_index
        self.n_genes = nb.call_spots.gene_efficiency.shape[0]
        self.mode = 'C'
        # Load spots
        self.load_spots(gene_index)
        # Now initialise the plot, adding fig and ax attributes to the class
        self.plot()

        plt.show()

    def load_spots(self, gene_index: int):
        nb = self.nb
        # First we need to find the spots used to calculate the gene efficiency for the given gene.
        self.gene_g_mask = nb.call_spots.use_ge * (nb.ref_spots.gene_no == gene_index)
        spots = nb.ref_spots.colors[self.gene_g_mask][:, :, nb.basic_info.use_channels]
        # remove background codes. To do this, repeat background_strenth along a new axis for rounds
        background_strength = nb.ref_spots.background_strength[self.gene_g_mask]
        background_strength = np.repeat(background_strength[:, np.newaxis, :], spots.shape[1], axis=1)
        # remove background from spots
        spots = spots - background_strength
        spots = spots.reshape((spots.shape[0], spots.shape[1] * spots.shape[2]))
        color_norm = np.repeat(nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                                      nb.basic_info.use_channels)].reshape((1, -1)),
                               spots.shape[0], axis=0)
        self.spots = spots / color_norm
        # order spots by nb.ref_spots.score in descending order
        self.spots = self.spots[np.argsort(nb.ref_spots.score[self.gene_g_mask])[::-1], :]
        # We need to find the expected spot profile for each round/channel
        self.spots_expected = nb.call_spots.bled_codes[self.gene_index, :, nb.basic_info.use_channels].T
        self.spots_expected = np.repeat(self.spots_expected.reshape((1, -1)), self.spots.shape[0], axis=0)

    def plot_ge_cmap(self):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 10))
        else:
            del self.fig, self.ax
            self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 10))
        # Now we can plot the spots. We want to create 2 subplots. One with the spots observed and one with the expected
        # spots. We will use the same color scale for both subplots.
        vmax = np.max([np.max(self.spots), np.max(self.spots_expected)]) / 2
        vmin = np.min([np.min(self.spots), np.min(self.spots_expected)]) / 5
        # We can then plot the spots observed and the spots expected.
        self.ax[0].imshow(self.spots, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
        self.ax[1].imshow(self.spots_expected, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', interpolation='none')
        # We can then add titles and axis labels to the subplots.
        self.ax[0].set_title('Observed Spot Colours')
        self.ax[1].set_title('Expected Spot Colours')
        self.ax[0].set_xlabel('Spot Colour (flattened)')
        self.ax[1].set_xlabel('Spot Colour (flattened)')
        self.ax[0].set_ylabel('Spot')
        self.ax[1].set_ylabel('Spot')

        # We would like to add red vertical lines to show the start of each round.
        for j in range(2):
            for i in range(self.nb.basic_info.n_rounds):
                self.ax[j].axvline(i * len(self.nb.basic_info.use_channels) - 0.5, color='r')

        # Set supertitle, colorbar and show plot
        self.fig.suptitle(
            'Gene Efficiency Calculation for Gene ' + self.nb.call_spots.gene_names[self.gene_index] + ' in Iteration '
            + str(0))

        # Add gene efficiency plot on top of prediction on the right
        ge = self.nb.call_spots.gene_efficiency[self.gene_index]
        ge_max = np.max(self.nb.call_spots.gene_efficiency)
        ge = ge * self.spots.shape[0] / ge_max
        ge = self.spots.shape[0] - ge
        ge = np.repeat(ge, self.nb.basic_info.n_rounds)
        # Now clip the gene efficiency to be between 0 and the number of spots
        ge = np.clip(ge, 0, self.spots.shape[0] - 1)
        # We can then plot the gene efficiency
        self.ax[1].plot(ge, color='w')
        # plot a white line at the gene efficiency of 1. This comes at
        self.ax[1].axhline(0, color='w', linestyle='--', label='Gene Efficiency = ' + str(np.round(ge_max, 2)))
        self.ax[1].axhline(self.spots.shape[0] - self.spots.shape[0] / ge_max, color='w', linestyle='--',
                           label='Gene Efficiency = 1')
        self.ax[1].axhline(self.spots.shape[0] - 1, color='w', linestyle='--', label='Gene Efficiency = 0')
        self.ax[1].legend(loc='upper right')
        # Add simple colorbar. Move this a little up to make space for the button.
        cax = self.fig.add_axes([0.925, 0.1, 0.03, 0.8])
        self.fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='viridis'),
                          cax=cax, label='Spot Colour')
        self.add_cmap_widgets()
        self.fig.canvas.draw_idle()

    def plot_ge_hist(self, percentile=95):
        """
        Plot n_rounds histograms of spot colours for the gene of interest. This will allow us to see how the spot
        intensity compares to the expected spot intensity.
        """
        nb = self.nb
        gene_code = nb.call_spots.gene_codes[self.gene_index]
        dye_names = nb.basic_info.dye_names
        # Swap all 2s and 3s in gene_code and swap 2 and 3 in dye names
        gene_code[gene_code == 2] = -1
        gene_code[gene_code == 3] = 2
        gene_code[gene_code == 10] = 3
        dye_names[2], dye_names[3] = dye_names[3], dye_names[2]
        n_channels = len(nb.basic_info.use_channels)
        colour_vector = nb.ref_spots.colors[:, :, nb.basic_info.use_channels][nb.ref_spots.isolated]
        colour_vector = colour_vector / nb.call_spots.color_norm_factor[:, nb.basic_info.use_channels]
        colour_vector = colour_vector.reshape((colour_vector.shape[0] * nb.basic_info.n_rounds, n_channels))
        dye = np.argmax(colour_vector, axis=1)

        dye_brightness = []
        for i in range(len(dye_names)):
            dye_brightness.append(colour_vector[dye == i, i])

        if not hasattr(self, 'fig'):
            # Create figure and axis
            self.fig, self.ax = plt.subplots(1, nb.basic_info.n_rounds, figsize=(20, 5))
        else:
            del self.fig, self.ax
            # Create figure and axis
            self.fig, self.ax = plt.subplots(1, nb.basic_info.n_rounds, figsize=(20, 5))

        fig, ax = self.fig, self.ax
        # Loop through each round and plot the histogram of spot colours
        for r in range(self.nb.basic_info.n_rounds):
            gene_g_spot_brightness = self.spots[:, r * n_channels + gene_code[r]]
            all_spot_brightness = dye_brightness[gene_code[r]]
            # Take max intensity as 90th percentile of rc_spot_intensity
            max_brightness = np.percentile(all_spot_brightness, percentile)
            # Now plot both spot intensities on the same histogram
            ax[r].hist(all_spot_brightness, bins=np.linspace(0, max_brightness, 20), alpha=0.5, density=True,
                       label='All genes')
            ax[r].hist(gene_g_spot_brightness, bins=np.linspace(0, max_brightness, 20), alpha=0.5, density=True,
                       label='Gene ' + nb.call_spots.gene_names[self.gene_index])

            # Add a box in the top right of the plot with the Gene Efficiency
            ge = nb.call_spots.gene_efficiency[self.gene_index, r]
            ax[r].set_yticks([])
            ax[r].set_xticks([])

            # We'd like to add two vertical lines at the median of both histograms. The first in blue (for
            # rc_spot_intensity) and the second in orange (for gene_g_spot_intensity).
            # First we need to calculate the median of both histograms
            rc_median = np.median(all_spot_brightness)
            gene_median = np.median(gene_g_spot_brightness)
            # Now plot the vertical lines
            ax[r].axvline(rc_median, color='blue', linestyle='--', label='Median = ' + str(np.round(rc_median, 2)))
            ax[r].axvline(gene_median, color='orange', linestyle='--',
                          label='Median = ' + str(np.round(gene_median, 2)))

            ax[r].set_title('Round ' + str(r) + ' Dye ' + dye_names[gene_code[r]])

            ax[r].set_xlabel('Spot Intensity \n Gene Efficiency = ' + str(np.round(ge, 2)))
            ax[r].set_ylabel('Frequency')
            # Add a legend for the median lines. We need to exclude the first two lines from the legend
            handles, labels = ax[r].get_legend_handles_labels()
            ax[r].legend(handles[2:], labels[2:], loc='upper right')

        # Adjust subplots to leave space on the right for the legend and buttons
        fig.subplots_adjust(right=0.9)

        # Add a single legend on the top right of the figure. Want to take only first 2 labels from ax 0
        legend_ax = fig.add_axes([0.925, 0.9, 0.05, 0.05])
        legend_ax.axis('off')
        legend_ax.legend(handles[:2], labels[:2], loc='upper right')

        fig.suptitle('Histogram of Spot Intensities for Gene ' + nb.call_spots.gene_names[self.gene_index] +
                     ' versus spots of the same dye in other genes for each round')
        self.add_hist_widgets()
        fig.canvas.draw_idle()

    def plot(self):
        # Get rid of any existing plots
        if hasattr(self, 'fig'):
            plt.close(self.fig)

        # Plot the correct plot
        if self.mode == 'C':
            self.plot_ge_cmap()
        elif self.mode == 'H':
            self.plot_ge_hist()

        self.add_switch_buttons()

    def button_cmap_clicked(self, event):
        self.mode = 'C'
        self.plot()

    def button_hist_clicked(self, event):
        self.mode = 'H'
        self.plot()

    def add_switch_buttons(self):
        # Add 2 buttons on the bottom right of the figure allowing the user to choose between viewing the
        # histogram or the colourmap. Make text black so that it is visible on the white background.
        ax_button = self.fig.add_axes([0.925, 0.05, 0.02, 0.03])
        self.button_hist = Button(ax_button, 'H', color='black')
        self.button_hist.on_clicked(self.button_hist_clicked)
        ax_button_cmap = self.fig.add_axes([0.95, 0.05, 0.02, 0.03])
        self.button_cmap = Button(ax_button_cmap, 'C', color='black')
        self.button_cmap.on_clicked(self.button_cmap_clicked)

    def add_cmap_widgets(self):
        # Initialise buttons and cursors
        # 1. We would like each row of the plot to be clickable, so that we can view the observed spot.
        spot_id = np.where(self.gene_g_mask)[0]
        mplcursors.cursor(self.ax[0], hover=False).connect(
            "add", lambda sel: view_spot(self.nb, spot_id[sel.target.index[0]]))
        # 2. We would like to add a white rectangle around the observed spot when we hover over it
        mplcursors.cursor(self.ax[0], hover=2).connect(
            "add", lambda sel: self.add_rectangle(sel.target.index[0]))

    def add_hist_widgets(self):
        # Add a slider on the right of the figure allowing the user to choose the percentile of the histogram
        # to use as the maximum intensity. This slider should be the same dimensions as the colorbar and should
        # be in the same position as the colorbar. We should slide vertically to change the percentile.
        self.ax_slider = self.fig.add_axes([0.94, 0.15, 0.02, 0.6])
        self.slider = Slider(self.ax_slider, 'Percentile', 80, 100, valinit=90, orientation='vertical')
        self.slider.on_changed(lambda val: self.update_hist(int(val)))

    def update_hist(self, percentile):
        nb = self.nb
        gene_code = nb.call_spots.gene_codes[self.gene_index]
        n_channels = len(nb.basic_info.use_channels)
        fig, ax = self.fig, self.ax
        dye_names = nb.basic_info.dye_names
        for r in range(self.nb.basic_info.n_rounds):
            matching_genes = [g for g in np.arange(nb.call_spots.gene_names.shape[0])
                              if nb.call_spots.gene_codes[g, r] == nb.call_spots.gene_codes[self.gene_index, r]]
            # Now look at all spots with a gene no in matching_genes
            use_spots = np.where(np.isin(self.nb.ref_spots.gene_no, matching_genes))[0]
            color_norm = nb.call_spots.color_norm_factor[r, nb.basic_info.use_channels[gene_code[r]]]
            rc_spot_intensity = self.nb.ref_spots.colors[use_spots][:, r,
                                nb.basic_info.use_channels[gene_code[r]]] / color_norm
            gene_g_spot_intensity = self.spots[:, r * n_channels + gene_code[r]]
            ax[r].clear()
            max_intensity = np.percentile(rc_spot_intensity, percentile)
            ax[r].hist(rc_spot_intensity, bins=np.linspace(0, max_intensity, 20), alpha=0.5, density=True,
                       label='All genes')
            ax[r].hist(gene_g_spot_intensity, bins=np.linspace(0, max_intensity, 20), alpha=0.5, density=True,
                       label='Gene ' + nb.call_spots.gene_names[self.gene_index])

            # We'd like to add two vertical lines at the median of both histograms. The first in blue (for
            # rc_spot_intensity) and the second in orange (for gene_g_spot_intensity).
            # First we need to calculate the median of both histograms
            rc_median = np.median(rc_spot_intensity)
            gene_median = np.median(gene_g_spot_intensity)
            # Now plot the vertical lines
            ax[r].axvline(rc_median, color='blue', linestyle='--', label='Median = ' + str(np.round(rc_median, 2)))
            ax[r].axvline(gene_median, color='orange', linestyle='--',
                          label='Median = ' + str(np.round(gene_median, 2)))

            # Add a box in the top right of the plot with the Gene Efficiency
            ge = nb.call_spots.gene_efficiency[self.gene_index, r]
            ax[r].set_yticks([])
            ax[r].set_xticks([])

            ax[r].set_title('Round ' + str(r) + ' Dye ' + dye_names[gene_code[r]])

            ax[r].set_xlabel('Spot Intensity \n Gene Efficiency = ' + str(np.round(ge, 2)))
            ax[r].set_ylabel('Frequency')

            # Add a legend for the median lines. We need to exclude the first two lines from the legend
            handles, labels = ax[r].get_legend_handles_labels()
            ax[r].legend(handles[2:], labels[2:], loc='upper right')

        self.fig.canvas.draw_idle()

    def add_rectangle(self, index):
        # We need to remove any existing rectangles from the plot
        index = max(0, index)
        index = min(index, self.spots.shape[0] - 1)
        for rectangle in self.ax[0].patches:
            rectangle.remove()
        # We can then add a new rectangle to the plot
        self.ax[0].add_patch(
            Rectangle((-0.5, index - 0.5), self.nb.basic_info.n_rounds * len(self.nb.basic_info.use_channels), 1,
                      fill=False, edgecolor='white'))


class BGNormViewer():
    """
    This function will plot all spots before and after background subtraction and order them by background noise.
    We will then plot the normalised spots too.
    Args:
        spot_colour_raw: [n_spots x n_rounds x n_channels_use] array of spots before background subtraction
        initial_bleed_matrix: [n_channels_use x n_dyes] array of bleed matrix
    """
    def __init__(self, nb):
        self.nb = nb
        spot_colour_raw = nb.ref_spots.colors.copy()[nb.ref_spots.isolated][:, :, nb.basic_info.use_channels]
        spot_colours_subtracted, background_noise = remove_background(spot_colour_raw.copy())
        norm_factor = nb.call_spots.color_norm_factor.copy()
        # Norm factor is now a 5 x 7 array so will get an error when we try to divide by it. Let's take the median
        # across the rounds dimension
        # norm_factor = np.median(norm_factor, axis=0)
        spot_colours_normed = spot_colours_subtracted / norm_factor[None, :, nb.basic_info.use_channels]
        n_spots, n_rounds, n_channels_use = spot_colour_raw.shape
        spot_colour_raw = spot_colour_raw.swapaxes(1, 2)
        spot_colours_subtracted = spot_colours_subtracted.swapaxes(1, 2)
        spot_colours_normed = spot_colours_normed.swapaxes(1, 2)
        spot_colour_raw = np.reshape(spot_colour_raw, (n_spots, n_rounds * n_channels_use))
        spot_colours_subtracted = np.reshape(spot_colours_subtracted, (n_spots, n_rounds * n_channels_use))
        spot_colours_normed = np.reshape(spot_colours_normed, (n_spots, n_rounds * n_channels_use))
        background_noise = np.sum(background_noise, axis=1)

        # Now we'd like to order the spots by background noise in descending order
        # We'll do this by sorting the background noise and then reordering the spots
        spot_colour_raw = spot_colour_raw[np.argsort(background_noise)[::-1]]
        spot_colours_subtracted = spot_colours_subtracted[np.argsort(background_noise)[::-1]]
        spot_colours_normed = spot_colours_normed[np.argsort(background_noise)[::-1]]

        # We're going to make a little viewer to show spots before and after background subtraction
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        max_intensity = np.max(spot_colour_raw)
        min_intensity = np.min(spot_colour_raw)
        ax[0].imshow(spot_colour_raw, aspect='auto', vmin=min_intensity / 10, vmax=max_intensity / 10,
                     interpolation='none')
        ax[0].set_title('Before background subtraction')
        ax[0].set_xticks([])

        ax[1].imshow(spot_colours_subtracted, aspect='auto', vmin=min_intensity / 10, vmax=max_intensity / 10,
                     interpolation='none')
        ax[1].set_title('After background subtraction')
        ax[1].set_xticks([])

        max_intensity = np.max(spot_colours_normed)
        min_intensity = np.min(spot_colours_normed)
        ax[2].imshow(spot_colours_normed, aspect='auto', vmin=min_intensity / 10, vmax=max_intensity / 10,
                     interpolation='none')
        ax[2].set_title('After background subtraction and normalisation')
        ax[2].set_xticks([])

        # Now add vertical dashed red lines to separate channels
        for i in range(n_rounds - 1):
            for j in range(3):
                ax[j].axvline((i + 1) * n_channels_use - 0.5, color='r', linestyle='--')

        # For each ax, add the channels to the x axis. Since the x axis has n_rounds channels, followed by n_rounds
        # channels, etc, we only need to add the channels once. We want to add the channel name at the bottom of
        # each set of channels, so we'll add the channel name at the position of the middle channel in each set
        for i in range(n_channels_use):
            y_pos = int(n_spots * (1 + 0.05))
            for j in range(3):
                ax[j].text(i * n_rounds + n_rounds // 2, y_pos, nb.basic_info.use_channels[i], color='w', fontsize=10)

        # Add a title
        fig.suptitle('Background subtraction and normalisation', fontsize=16)

        plt.show()

    # add slider to allow us to vary value of interp between 0 and 1 and update plot
    # def add_hist_widgets(self):
    #     Add a slider on the right of the figure allowing the user to choose the percentile of the histogram
    #     to use as the maximum intensity. This slider should be the same dimensions as the colorbar and should
    #     be in the same position as the colorbar. We should slide vertically to change the percentile.
        # self.ax_slider = self.fig.add_axes([0.94, 0.15, 0.02, 0.6])
        # self.slider = Slider(self.ax_slider, 'Interpolation Coefficient', 0, 1, valinit=0, orientation='vertical')
        # self.slider.on_changed(lambda val: self.update_hist(int(val)))
    #
    # TODO: Add 2 buttons, one for separating normalisation by channel and one for separating by round and channel


class ViewBleedCalc:
    def __init__(self, nb: Notebook):
        self.nb = nb
        self.dye_names = nb.basic_info.dye_names.copy()
        # swap dyes 2 and 3
        dye_2, dye_3 = self.dye_names[2].copy(), self.dye_names[3].copy()
        self.dye_names[2], self.dye_names[3] = dye_3, dye_2
        color_norm = nb.call_spots.color_norm_factor[:, nb.basic_info.use_channels]
        # We're going to remove background from spots, so need to expand the background strength variable from
        # n_spots x n_channels to n_spots x n_rounds x n_channels by repeating the values for each round
        background_strength = np.repeat(nb.ref_spots.background_strength[nb.ref_spots.isolated, np.newaxis, :],
                                             len(nb.basic_info.use_rounds), axis=1)
        self.isolated_spots = nb.ref_spots.colors[nb.ref_spots.isolated][:, :, nb.basic_info.use_channels] - \
                              background_strength
        self.isolated_spots = self.isolated_spots / color_norm
        # Get current working directory and load default bleed matrix
        self.default_bleed = np.load(os.path.join(os.getcwd(), 'coppafish/setup/default_bleed.npy')).copy()
        # swap columns 2 and 3 to match the order of the channels in the notebook
        self.default_bleed[:, [2, 3]] = self.default_bleed[:, [3, 2]]
        # Normalise each column of default_bleed to have L2 norm of 1
        self.dye_template = self.default_bleed / np.linalg.norm(self.default_bleed, axis=0)
        # Now we are going to loop through all isolated spots, convert these to n_rounds colour vectors and then
        # assign each colour vector to a dye
        self.colour_vectors = self.isolated_spots.reshape((self.isolated_spots.shape[0] * self.nb.basic_info.n_rounds,
                                                           len(self.nb.basic_info.use_channels)))
        self.all_dye_score = np.zeros((self.colour_vectors.shape[0], self.dye_template.shape[1]))
        self.dye_score = np.zeros(self.colour_vectors.shape[0])
        self.dye_assignment = np.zeros(self.colour_vectors.shape[0], dtype=int)
        for i in range(self.colour_vectors.shape[0]):
            # Assign each vector to dye which maximises dot product with vector
            self.all_dye_score[i] = self.colour_vectors[i] @ self.dye_template
            self.dye_assignment[i] = np.argmax(self.all_dye_score[i])
            self.dye_score[i] = np.max(self.all_dye_score[i])

        # Now we have assigned each colour vector to a dye, we can view all colour vectors assigned to each dye
        max_intensity = np.percentile(self.colour_vectors, 99.5)
        max_score = np.percentile(self.dye_score, 99.5)
        fig, ax = plt.subplots(2, self.dye_template.shape[1], figsize=(10, 10))
        for i in range(self.dye_template.shape[1]):
            # Plot the colour vectors assigned to each dye
            dye_vectors = self.colour_vectors[self.dye_assignment == i]
            # Order these vectors by dye score in descending order
            scores = self.dye_score[self.dye_assignment == i]
            # Now use these scores to order the vectors
            dye_vectors = dye_vectors[np.argsort(scores)[::-1]]
            ax[0, i].imshow(dye_vectors, vmin=0, vmax=max_intensity/2, aspect='auto', interpolation='none')
            ax[0, i].set_title(self.nb.basic_info.dye_names[i])
            ax[0, i].set_yticks([])
            ax[0, i].set_xticks([])

            # Add a horizontal red line at a dye score of 1. Take the first index where the dye score is less than 1.
            score_1_index = np.where(np.sort(scores)[::-1] < 1)[0][0]
            ax[0, i].axhline(score_1_index, color='r', linestyle='--', label='Dye Score = 1')

            # Plot a histogram of the dye scores
            ax[1, i].hist(self.dye_score[self.dye_assignment == i], bins=np.linspace(0, max_score / 2, 200))
            ax[1, i].set_title(self.dye_names[i])
            ax[1, i].set_xlabel('Dye Score')
            ax[1, i].set_ylabel('Frequency')
            ax[1, i].set_yticks([])
            mean = np.mean(self.dye_score[self.dye_assignment == i])
            ax[1, i].axvline(mean, color='r')
            # Add label in top right corner of each plot with median dye score
            ax[1, i].text(0.95, 0.95, 'Mean = ' + '{:.2f}'.format(mean), color='r',
                          horizontalalignment='right', verticalalignment='top', transform=ax[1, i].transAxes,
                          fontsize=8)

        # Add a single colour bar for all plots on the right. Label this as spot intensity. Make the range 0 to vmax/2
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_ax.set_ylabel('Spot Intensity (normalised)')
        fig.colorbar(ax[0, 0].get_images()[0], cax=cbar_ax, ticks=np.linspace(0, max_intensity / 2, dtype=int))

        # Add single legend for all red dotted lines on top row
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.85, 0.95))

        # Add super title
        fig.suptitle('Bleed matrix calculation', fontsize=16)

        plt.show()
