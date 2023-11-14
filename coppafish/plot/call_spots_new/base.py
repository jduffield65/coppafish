import mplcursors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
from sklearn import linear_model
import matplotlib as mpl
try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources

from ...setup import Notebook
from ...spot_colors.base import normalise_rc
from ..call_spots.spot_colors import view_codes


plt.style.use('dark_background')


class ViewAllGeneScores():
    """
    Module to view all gene scores in a grid of n_genes histograms.
    """
    def __init__(self, nb):
        """
        Load in notebook and spots.
        Args:
            nb: Notebook object. Must have ref_spots page
        """
        self.nb = nb
        self.load_values()
        self.plot()

    def load_values(self, mode: str = 'score'):
        """
        Load in values to be plotted.
        Args:
            mode: 'score', 'prob', 'score_diff' or 'intensity'
        """
        self.mode = mode
        if mode == 'score':
            values = self.nb.ref_spots.score
        elif mode == 'prob':
            values = np.max(self.nb.ref_spots.gene_probs, axis=1)
        elif mode == 'score_diff':
            values = self.nb.ref_spots.score_diff
        elif mode == 'intensity':
            values = self.nb.ref_spots.intensity
        else:
            raise ValueError("mode must be 'score', 'prob', 'score_diff' or 'intensity'")

        gene_values = np.zeros((self.nb.call_spots.gene_names.shape[0], 0)).tolist()
        gene_prob_assignments = np.argmax(self.nb.ref_spots.gene_probs, axis=1)
        for i in range(len(gene_values)):
            if mode != 'prob':
                gene_values[i] = values[self.nb.ref_spots.gene_no == i]
            else:
                gene_values[i] = values[gene_prob_assignments == i]

        self.gene_values = gene_values
        self.n_spots = np.array([len(gene_values[i]) for i in range(len(gene_values))])

    def plot(self):
        """
        Plot self.gene_values in a grid of histograms. Side length of grid is sqrt(n_genes).
        """
        # Delete any existing plots
        grid_dim = int(np.ceil(np.sqrt(len(self.nb.call_spots.gene_names))))
        if hasattr(self, 'fig'):
            plt.close(self.fig)

        self.fig, self.ax = plt.subplots(grid_dim, grid_dim, figsize=(10, 10))

        # Move subplots down to make room for the title and to the left to make room for the colourbar
        self.fig.subplots_adjust(top=0.9)
        self.fig.subplots_adjust(right=0.9)

        # Now loop through each subplot. If there are more subplots than genes, we want to delete the empty ones
        # We also want to plot the histogram of scores for each gene, and colour the histogram based on the number of
        # spots for that gene
        for i in range(grid_dim ** 2):
            # Choose the colour of the histogram based on the number of spots. We will use a log scale, with the
            # minimum number of spots being 1 and the maximum being 1000. Use a blue to red colourmap
            cmap = plt.get_cmap("coolwarm")
            norm = mpl.colors.Normalize(vmin=1, vmax=np.percentile(self.n_spots, 99))

            # Plot the histogram of scores for each gene
            r, c = i // grid_dim, i % grid_dim
            if i < len(self.nb.call_spots.gene_names):
                if self.n_spots[i] < 50:
                    n_bins = 10
                else:
                    n_bins = 50
                self.ax[r, c].hist(self.gene_values[i], bins=n_bins, color=cmap(norm(self.n_spots[i])))
                self.ax[r, c].set_title(self.nb.call_spots.gene_names[i])
                self.ax[r, c].set_xlim(0, 1)
                self.ax[r, c].set_xticks([])
                self.ax[r, c].set_yticks([])

            # Next we want to delete the empty plots
            else:
                self.ax[r, c].axis("off")
                self.ax[r, c].set_xticks([])
                self.ax[r, c].set_yticks([])

        # Add overall title, colourbar and buttons
        self.fig.suptitle("Distribution of Scores for Each Gene", fontsize=16)
        cax = self.fig.add_axes([0.95, 0.1, 0.03, 0.6])
        mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, label="Number of Spots")
        # Put label on the left of the colourbar
        cax.yaxis.set_label_position("left")
        self.add_buttons()
        self.fig.canvas.draw_idle()

    def add_buttons(self):
        """
        Add buttons to the plot. Buttons are:
            - 'score': plot the distribution of scores
            - 'prob': plot the distribution of probabilities
            - 'score_diff': plot the distribution of score differences
            - 'intensity': plot the distribution of intensities
        """
        # coords for colourbar are [0.95, 0.1, 0.03, 0.6], we want to put the buttons just above the colourbar

        # create axes for the buttons
        ax_intensity = self.fig.add_axes([0.95, 0.75, 0.03, 0.03])
        ax_score_diff = self.fig.add_axes([0.95, 0.8, 0.03, 0.03])
        ax_prob = self.fig.add_axes([0.95, 0.85, 0.03, 0.03])
        ax_score = self.fig.add_axes([0.95, 0.9, 0.03, 0.03])

        # create the buttons, make them black. Hovering over will make them white
        self.button_intensity = Button(ax_intensity, 'intensity', color='black', hovercolor='white')
        self.button_score_diff = Button(ax_score_diff, 'score_diff', color='black', hovercolor='white')
        self.button_prob = Button(ax_prob, 'prob', color='black', hovercolor='white')
        self.button_score = Button(ax_score, 'score', color='black', hovercolor='white')

        # connect the buttons to the update function. We need to ensure the event passed to update is a string
        self.button_intensity.on_clicked(lambda event: self.update('intensity'))
        self.button_score_diff.on_clicked(lambda event: self.update('score_diff'))
        self.button_prob.on_clicked(lambda event: self.update('prob'))
        self.button_score.on_clicked(lambda event: self.update('score'))

    def update(self, event):
        """
            Update the plot when a button is clicked.
        """
        self.mode = event
        self.load_values(mode=self.mode)
        self.plot()


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
        mplcursors.cursor(self.ax, hover=True).connect("add", lambda sel: self.plot_gene_name(sel.index[0]))
        # 2. Allow genes to be selected by clicking on them
        mplcursors.cursor(self.ax, hover=False).connect("add", lambda sel: GESpotViewer(nb, sel.index[0]))
        # 3. We would like to add a white rectangle around the observed spot when we hover over it. We will
        # use mplcursors to do this. We need to add a rectangle to the plot when hovering over a gene.
        # We also want to turn off annotation when hovering over a gene so we will use the `hover=False` option.
        mplcursors.cursor(self.ax, hover=2).connect("add", lambda sel: self.add_rectangle(sel.index[0]))

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
    def __init__(self, nb: Notebook, gene_index: int = 0, use_ge=True):
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
        self.use_ge = use_ge
        # Load spots
        self.load_spots(gene_index)
        # Now initialise the plot, adding fig and ax attributes to the class
        self.plot()

        plt.show()

    def load_spots(self, gene_index: int, use_ge=True):
        nb = self.nb
        n_channels = len(nb.basic_info.use_channels)
        # First we need to find the spots used to calculate the gene efficiency for the given gene.
        initial_assignment = np.argmax(nb.ref_spots.gene_probs, axis=1)
        if use_ge:
            self.gene_g_mask = nb.call_spots.use_ge * (initial_assignment == gene_index)
        else:
            self.gene_g_mask = nb.ref_spots.gene_no == gene_index
        # self.gene_g_mask = nb.ref_spots.gene_no == gene_index
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
        # order spots by nb.ref_spots.score
        # self.spots = self.spots[np.argsort(nb.ref_spots.score[self.gene_g_mask]), :]
        # We need to find the expected spot profile for each round/channel
        self.spots_expected = nb.call_spots.bled_codes[self.gene_index, :, nb.basic_info.use_channels].T
        self.spots_expected = np.repeat(self.spots_expected.reshape((1, -1)), self.spots.shape[0], axis=0)
        # Now for each spot we would like to store a dye_efficiency vector. This is the least squares solution to
        # the equation spots[i, r, :] = dye_efficiency * spots_expected[0, r, :].
        auxilliary_spots = self.spots.reshape((self.spots.shape[0], self.spots.shape[1] // n_channels, n_channels))
        auxilliary_spots_expected = self.spots_expected.reshape((self.spots.shape[0], self.spots.shape[1] // n_channels,
                                                                 n_channels))[0, :, :]
        self.dye_efficiency = np.zeros((auxilliary_spots.shape[0], auxilliary_spots.shape[1]))

        for s in range(auxilliary_spots.shape[0]):
            for r in range(auxilliary_spots.shape[1]):
                a = auxilliary_spots[s, r]
                b = auxilliary_spots_expected[r]
                self.dye_efficiency[s, r] = np.dot(a, b) / np.dot(b, b)

        self.dye_efficiency_norm = self.dye_efficiency / np.linalg.norm(self.dye_efficiency, axis=1)[:, np.newaxis]
        # Estimate parameters of VMF distribution for dye efficiency norm
        r_bar = np.linalg.norm(np.mean(self.dye_efficiency_norm, axis=0))
        self.mu_hat = np.mean(self.dye_efficiency_norm, axis=0) / r_bar
        self.kappa_hat = r_bar * (nb.basic_info.n_rounds - r_bar ** 2) / (1 - r_bar ** 2)
        self.likelihood = np.sum(np.repeat(self.mu_hat[np.newaxis, :],
                                           self.spots.shape[0], axis=0) * self.dye_efficiency_norm, axis=1)

        # order spots by likelihood of dye efficiency
        self.spots = self.spots[np.argsort(self.likelihood), :]
        self.spot_id = np.where(self.gene_g_mask)[0][np.argsort(self.likelihood)]

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
        self.ax[0].set_ylim(0, self.spots.shape[0])
        self.ax[1].set_ylim(0, self.spots.shape[0])
        self.ax[0].set_xlim(-.5, self.spots.shape[1] - .5)
        self.ax[1].set_xlim(-.5, self.spots.shape[1] - .5)
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
            'Gene Efficiency Calculation for Gene ' + self.nb.call_spots.gene_names[self.gene_index] +
            '. \n Estimated concentration parameter = ' + str(np.round(self.kappa_hat, 2)))

        # Add gene efficiency plot on top of prediction on the right
        ge = self.nb.call_spots.gene_efficiency[self.gene_index]
        ge_max = np.max(self.nb.call_spots.gene_efficiency)
        ge = ge * self.spots.shape[0] / ge_max
        ge = np.repeat(ge, self.nb.basic_info.n_rounds)
        # Now clip the gene efficiency to be between 0 and the number of spots
        ge = np.clip(ge, 0, self.spots.shape[0] - 1)
        # We can then plot the gene efficiency
        self.ax[1].plot(ge, color='w')
        # plot a white line at the gene efficiency of 1.
        self.ax[1].axhline(0, color='w', linestyle='--', label='Gene Efficiency = ' + str(np.round(ge_max, 2)))
        self.ax[1].axhline(self.spots.shape[0] / ge_max, color='w', linestyle='--', label='Gene Efficiency = 1')
        self.ax[1].axhline(self.spots.shape[0] - 1, color='w', linestyle='--', label='Gene Efficiency = 0')
        self.ax[1].legend(loc='upper right')

        # For each round, we'd like to plot the dye efficiencies associated with the spots in this gene and round.
        # First, for each round, sort dye_efficiencies in ascending order, then clip between 0 and ge_max.
        for r in range(self.nb.basic_info.n_rounds):
            dye_efficiency = self.dye_efficiency[:, r]
            dye_efficiency = np.sort(dye_efficiency)
            dye_efficiency = np.clip(dye_efficiency, 0, ge_max)
            # Now we can plot the dye efficiencies. We want to plot them on the left subplot, and we want each round's
            # dye efficiencies to be in between the red lines corresponding to that round.
            # In order to make the dye_efficiencies appear in the correct place, we need to scale and shift them like
            # we did with gene efficiency.
            dye_efficiency = dye_efficiency * self.spots.shape[0] / ge_max
            x_start, x_end = (r * len(self.nb.basic_info.use_channels) - 0.5,
                              (r + 1) * len(self.nb.basic_info.use_channels) - 0.5)
            self.ax[0].plot(np.linspace(x_start, x_end, len(dye_efficiency)), dye_efficiency, color='w')

        # Add simple colorbar. Move this a little up to make space for the button.
        cax = self.fig.add_axes([0.925, 0.1, 0.03, 0.7])
        self.fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='viridis'),
                          cax=cax, label='Spot Colour')
        self.add_cmap_widgets()
        self.fig.canvas.draw_idle()

    def plot_ge_hist(self, percentile=95):
        """
        Plot n_rounds histograms of all gene_efficiencies for each spot in each round.
        """
        nb = self.nb
        gene_code = nb.call_spots.gene_codes[self.gene_index]

        if not hasattr(self, 'fig'):
            # Create figure and axis
            self.fig, self.ax = plt.subplots(1, nb.basic_info.n_rounds, figsize=(20, 5))
        else:
            del self.fig, self.ax
            # Create figure and axis
            self.fig, self.ax = plt.subplots(1, nb.basic_info.n_rounds, figsize=(20, 5))

        fig, ax = self.fig, self.ax
        # Loop through each round and plot the histogram of dye efficiencies
        for r in range(self.nb.basic_info.n_rounds):
            # Take max efficiency as percentile of rc_spot_intensity
            max_efficiency = np.percentile(self.dye_efficiency_norm, percentile)
            # Now plot both spot intensities on the same histogram
            ax[r].hist(self.dye_efficiency_norm[:, r], bins=np.linspace(0, max_efficiency, 20), density=True)

            # Add a box in the top right of the plot with the Gene Efficiency
            ge = nb.call_spots.gene_efficiency[self.gene_index, r]
            ax[r].set_yticks([])
            ax[r].set_xticks([])
            ax[r].set_xlabel('Relative Dye Efficiency \n Gene Efficiency = ' + str(np.round(ge, 2)))
            ax[r].set_ylabel('Frequency')

        # Adjust subplots to leave space on the right for the legend and buttons
        fig.subplots_adjust(right=0.9)

        # Add a single legend on the top right of the figure. Want to take only first 2 labels from ax 0
        legend_ax = fig.add_axes([0.925, 0.9, 0.05, 0.05])
        legend_ax.axis('off')
        fig.suptitle('Histogram of relative dye efficiencies for Gene ' + nb.call_spots.gene_names[self.gene_index])
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

    def button_all_clicked(self, event):
        self.use_ge = not self.use_ge
        self.load_spots(self.gene_index, self.use_ge)
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
        # Add button to show all spots in the gene. Put this in the top right of the figure
        ax_button_show_all = self.fig.add_axes([0.925, 0.825, 0.05, 0.05])
        self.button_all = Button(ax_button_show_all, 'ALL', color='black')
        self.button_all.on_clicked(self.button_all_clicked)

    def add_cmap_widgets(self):
        # Initialise buttons and cursors
        # 1. We would like each row of the plot to be clickable, so that we can view the observed spot.
        mplcursors.cursor(self.ax[0], hover=False).connect(
            "add", lambda sel: view_codes(self.nb, self.spot_id[sel.index[0]]))
        # 2. We would like to add a white rectangle around the observed spot when we hover over it
        mplcursors.cursor(self.ax[0], hover=2).connect(
            "add", lambda sel: self.add_rectangle(sel.index[0]))

    def add_hist_widgets(self):
        # Add a slider on the right of the figure allowing the user to choose the percentile of the histogram
        # to use as the maximum intensity. This slider should be the same dimensions as the colorbar and should
        # be in the same position as the colorbar. We should slide vertically to change the percentile.
        self.ax_slider = self.fig.add_axes([0.94, 0.15, 0.02, 0.6])
        self.slider = Slider(self.ax_slider, 'Percentile', 80, 100, valinit=90, orientation='vertical')
        self.slider.on_changed(lambda val: self.update_hist(int(val)))

    def update_hist(self, percentile):
        nb = self.nb
        fig, ax = self.fig, self.ax
        # Loop through each round and plot the histogram of dye efficiencies
        for r in range(self.nb.basic_info.n_rounds):
            # Take max efficiency as percentile of rc_spot_intensity
            max_efficiency = np.percentile(self.dye_efficiency_norm, percentile)
            # Now plot both spot intensities on the same histogram
            ax[r].cla()
            ax[r].hist(self.dye_efficiency_norm[:, r], bins=np.linspace(0, max_efficiency, 20), density=True)

            # Add a box in the top right of the plot with the Gene Efficiency
            ge = nb.call_spots.gene_efficiency[self.gene_index, r]
            ax[r].set_yticks([])
            ax[r].set_xticks([])
            ax[r].set_xlabel('Spot Intensity \n Gene Efficiency = ' + str(np.round(ge, 2)))
            ax[r].set_ylabel('Frequency')

        # Adjust subplots to leave space on the right for the legend and buttons
        fig.subplots_adjust(right=0.9)
        fig.canvas.draw_idle()

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
    """
    def __init__(self, nb):
        self.nb = nb
        isolated = nb.ref_spots.isolated
        n_spots = np.sum(isolated)
        n_rounds, n_channels_use = len(nb.basic_info.use_rounds), len(nb.basic_info.use_channels)
        norm_factor = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds, nb.basic_info.use_channels)]
        background_noise = np.repeat(nb.ref_spots.background_strength[isolated][:, np.newaxis, :],
                                     nb.basic_info.n_rounds, axis=1)

        spot_colour_raw = nb.ref_spots.colors.copy()[isolated][:, :, nb.basic_info.use_channels]
        spot_colour_no_bg = spot_colour_raw - background_noise
        spot_colour_normed_no_bg = spot_colour_no_bg / norm_factor[None, :, :]
        # Now we'd like to order the spots by background noise in descending order
        background_noise = np.sum(abs(background_noise), axis=(1, 2))
        spot_colour_raw = spot_colour_raw[np.argsort(background_noise)[::-1]]
        spot_colour_no_bg = spot_colour_no_bg[np.argsort(background_noise)[::-1]]
        spot_colour_normed_no_bg = spot_colour_normed_no_bg[np.argsort(background_noise)[::-1]]
        # Finally, we need to reshape the spots to be n_spots x n_rounds * n_channels. Since we want the channels to be
        # in consecutive blocks of size n_rounds, we can reshape by first switching the round and channel axes.
        spot_colour_raw = spot_colour_raw.transpose((0, 2, 1))
        spot_colour_raw = spot_colour_raw.reshape((n_spots, n_rounds * n_channels_use))
        spot_colour_no_bg = spot_colour_no_bg.transpose((0, 2, 1))
        spot_colour_no_bg = spot_colour_no_bg.reshape((n_spots, n_rounds * n_channels_use))
        spot_colour_normed_no_bg = spot_colour_normed_no_bg.transpose((0, 2, 1))
        spot_colour_normed_no_bg = spot_colour_normed_no_bg.reshape((n_spots, n_rounds * n_channels_use))

        # We're going to make a little viewer to show spots before and after background subtraction
        colour_scaling_factor = 10
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        max_intensity = np.max(spot_colour_raw)
        min_intensity = np.min(spot_colour_raw)
        ax[0].imshow(spot_colour_raw, aspect='auto', vmin=min_intensity / colour_scaling_factor,
                     vmax=max_intensity / colour_scaling_factor, interpolation='none')
        ax[0].set_title('Before background subtraction and normalisation')
        ax[0].set_xticks([])

        max_intensity = np.max(spot_colour_no_bg)
        min_intensity = np.min(spot_colour_no_bg)
        ax[1].imshow(spot_colour_no_bg, aspect='auto', vmin=min_intensity / colour_scaling_factor,
                     vmax=max_intensity / colour_scaling_factor, interpolation='none')
        ax[1].set_title('After bg removal, before normalisation')
        ax[1].set_xticks([])

        max_intensity = np.max(spot_colour_normed_no_bg)
        min_intensity = np.min(spot_colour_normed_no_bg)
        ax[2].imshow(spot_colour_normed_no_bg, aspect='auto', vmin=min_intensity / colour_scaling_factor,
                     vmax=max_intensity / colour_scaling_factor, interpolation='none')
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
        if nb.file_names.initial_bleed_matrix is not None:
            self.default_bleed = np.load(nb.file_names.initial_bleed_matrix)
        else:
            # Get current working directory and load the default bleed matrix
            self.default_bleed \
                = np.load(importlib_resources.files('coppafish.setup').joinpath('default_bleed.npy')).copy()
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


class GeneScoreScatter():
    def __init__(self, nb: Notebook, gene_no: int = 0):
        """
        Function to view a scatter plot of the gene scores for each gene in each spot. The x-axis is the second-highest
        gene score for each spot and the y-axis is the highest gene score for each spot. This is useful for identifying
        spots where the gene scores are very similar and therefore the spot is likely to be a false positive.
        Args:
            nb: Notebook (containing page ref spots)
            gene_no: Gene number to plot
        """
        self.nb = nb
        self.gene_no = gene_no
        self.n_genes = len(self.nb.call_spots.gene_names)

        # Plot the scatter plot
        self.plot_scatter()

    def plot_scatter(self, event=None):
        # Plot the scatter plot
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()
        gene_g_mask = self.nb.ref_spots.gene_no == self.gene_no
        self.score = self.nb.ref_spots.score[gene_g_mask]
        self.second_score = self.score - self.nb.ref_spots.score_diff[gene_g_mask]
        self.ax.scatter(x=self.second_score, y=self.score, s=1)
        # Add a line at y=x
        self.ax.plot([0, 1], [0, 1], color='r')
        # Set x and y limits to be 0 to 1
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Second highest gene score')
        self.ax.set_ylabel('Highest gene score')
        self.ax.set_title('Gene score scatter plot for gene ' + self.nb.call_spots.gene_names[self.gene_no])
        # Shift the image a bit to the left to make room for buttons
        self.fig.subplots_adjust(right=0.8)
        self.add_gene_slider()
        self.fig.canvas.draw()

    def add_gene_slider(self):
        # Add a slider to select the gene. As we are moving the slider, we want to show the name of the gene
        # corresponding to the slider value. This will be displayed in the text box below the slider.
        # We do not want to update the scatter plot as we move the slider, as this will be slow. Instead, we will
        # add a button to update the scatter plot once we have selected the gene.
        # First, create the slider. Make it vertical and put it on the right of the plot.
        self.gene_slider_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
        self.gene_slider = Slider(self.gene_slider_ax, 'Gene', 0, self.n_genes - 1, valinit=self.gene_no,
                                    valstep=1, orientation="vertical")
        self.gene_slider.on_changed(self.update_gene)
        self.gene_text_ax = self.fig.add_axes([0.85, 0.05, 0.05, 0.05])
        # remove x and y ticks from button axes and gene text axes
        self.gene_text_ax.set_xticks([])
        self.gene_text_ax.set_yticks([])
        self.gene_text = self.gene_text_ax.text(0.5, 0.5, self.nb.call_spots.gene_names[self.gene_no],
                                                horizontalalignment='center', verticalalignment='center')
        self.plot_button_ax = self.fig.add_axes([0.85, 0.9, 0.05, 0.05])
        self.plot_button_ax.set_xticks([])
        self.plot_button_ax.set_yticks([])
        # Now create a clickable button to update the scatter plot, make the button black
        self.plot_button = Button(self.plot_button_ax, 'Plot', color='k')
        # Now link the click event to the function to update the scatter plot
        self.plot_button.on_clicked(self.plot_scatter)

    def update_gene(self, val):
        # Update the gene number and gene name
        self.gene_no = int(val)
        self.gene_text.set_text(self.nb.call_spots.gene_names[self.gene_no])
        # Update the scatter plot
        self.fig.canvas.draw()


class GEScatter():
    def __init__(self, nb: Notebook, gene_no: int = 0):
        """
        Function to view a scatter plot of the gene scores for each gene in each spot. The x-axis is the second-highest
        gene score for each spot and the y-axis is the highest gene score for each spot. This is useful for identifying
        spots where the gene scores are very similar and therefore the spot is likely to be a false positive.
        Args:
            nb: Notebook (containing page ref spots)
            gene_no: Gene number to plot
        """
        self.nb = nb
        self.gene_no = gene_no
        self.n_genes = len(self.nb.call_spots.gene_names)
        self.gene_codes = self.nb.call_spots.gene_codes.copy()
        # Transpose dye 2 and dye 3
        self.gene_codes[self.gene_codes == 2] = 9
        self.gene_codes[self.gene_codes == 3] = 2
        self.gene_codes[self.gene_codes == 9] = 3

        # Plot the scatter plot
        self.plot_scatter()

    def plot_scatter(self, event=None):
        # Plot the scatter plots
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(nrows=7, ncols=7, figsize=(20, 5))
        else:
            for ax in self.ax.flatten():
                ax.clear()
        gene_g_mask = (self.nb.ref_spots.gene_no == self.gene_no) * (self.nb.call_spots.use_ge)
        spot_colours = self.nb.ref_spots.colors[gene_g_mask][:, :, self.nb.basic_info.use_channels]
        # Remove background from spot colours
        background = np.repeat(self.nb.ref_spots.background_strength[gene_g_mask, np.newaxis, :], 7, axis=1)
        spot_colours = spot_colours - background
        spot_colours = spot_colours / self.nb.call_spots.color_norm_factor[:, self.nb.basic_info.use_channels]
        gene_g_code = self.gene_codes[self.gene_no]
        spot_round_brightness = np.zeros((spot_colours.shape[0], 7))
        # Populate spot_round_brightness with the brightness of each spot in each round (this is the intensity of the
        # spot in round r in the max channel of the dye corresponding to gene_g_code[r])
        for r in range(7):
            spot_round_brightness[:, r] = spot_colours[:, r, gene_g_code[r]]

        # Now plot the scatter plots of spot_round_brightness vs spot_round_brightness[:, r] for each r
        for row in range(7):
            for col in range(7):
                self.ax[row][col].scatter(x=spot_round_brightness[:, col], y=spot_round_brightness[:, row], s=1)
                self.ax[row][col].set_ylim(np.percentile(spot_round_brightness, 1),
                                    np.percentile(spot_round_brightness, 99))
                self.ax[row][col].set_xlim(np.percentile(spot_round_brightness, 1),
                                    np.percentile(spot_round_brightness, 99))
                # plot the line of best fit
                x = spot_round_brightness[:, col]
                y = spot_round_brightness[:, row]
                # Calculate the line of best fit. Use robust linear regression to avoid outliers
                huber = linear_model._huber.HuberRegressor()
                huber.fit(x[:, np.newaxis], y)
                m = huber.coef_[0]
                c = huber.intercept_
                r_squared = np.corrcoef(x, y)
                self.ax[row][col].plot(x, m * x + c, color='red', label='R$^2$ = ' + str(round(r_squared[0, 1], 2)) + '\n' +
                                       'Slope = ' + str(round(m, 2)))
                self.ax[row][col].set_xticks([])
                self.ax[row][col].set_yticks([])

        # Add row and column labels
        ge = self.nb.call_spots.gene_efficiency[self.gene_no]
        for i in range(7):
            self.ax[6][i].set_xlabel('Round ' + str(i))
            self.ax[i][0].set_ylabel('Round ' + str(i) + '\n' + 'GE = ' + str(round(ge[i], 2)))

        # Add a title
        self.fig.suptitle('Correlation between rounds for gene ' + self.nb.call_spots.gene_names[self.gene_no])
        self.fig.subplots_adjust(right=0.8)
        self.add_gene_slider()
        self.fig.canvas.draw()

    def add_gene_slider(self):
        # Add a slider to select the gene. As we are moving the slider, we want to show the name of the gene
        # corresponding to the slider value. This will be displayed in the text box below the slider.
        # We do not want to update the scatter plot as we move the slider, as this will be slow. Instead, we will
        # add a button to update the scatter plot once we have selected the gene.
        # First, create the slider. Make it vertical and put it on the right of the plot.
        self.gene_slider_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
        self.gene_slider = Slider(self.gene_slider_ax, 'Gene', 0, self.n_genes - 1, valinit=self.gene_no,
                                    valstep=1, orientation="vertical")
        self.gene_slider.on_changed(self.update_gene)
        self.gene_text_ax = self.fig.add_axes([0.85, 0.05, 0.05, 0.05])
        # remove x and y ticks from button axes and gene text axes
        self.gene_text_ax.set_xticks([])
        self.gene_text_ax.set_yticks([])
        self.gene_text = self.gene_text_ax.text(0.5, 0.5, self.nb.call_spots.gene_names[self.gene_no],
                                                horizontalalignment='center', verticalalignment='center')
        self.plot_button_ax = self.fig.add_axes([0.85, 0.9, 0.05, 0.05])
        self.plot_button_ax.set_xticks([])
        self.plot_button_ax.set_yticks([])
        # Now create a clickable button to update the scatter plot, make the button black
        self.plot_button = Button(self.plot_button_ax, 'Plot', color='k')
        # Now link the click event to the function to update the scatter plot
        self.plot_button.on_clicked(self.plot_scatter)

    def update_gene(self, val):
        # Update the gene number and gene name
        self.gene_no = int(val)
        self.gene_text.set_text(self.nb.call_spots.gene_names[self.gene_no])


class GeneProbs():
    def __init__(self, nb: Notebook, gene_no: int = 0):
        self.nb = nb
        self.gene_no = gene_no
        self.n_genes = len(self.nb.call_spots.gene_names)
        self.plot_gene_probs()

    def load_gene_probs(self):
        gene_g_mask = self.nb.ref_spots.gene_no == self.gene_no
        gene_g_ids = np.where(gene_g_mask)[0]
        # Get the spot probabilities for the selected gene
        spot_probs = self.nb.ref_spots.gene_probs[gene_g_mask]
        # We will order the spots by the gene they are assigned to with the highest probability
        self.prob_matrix = np.zeros((0, self.n_genes))
        self.spot_ids = np.zeros(0, dtype=int)
        num_assignments = np.zeros(self.n_genes)
        for i in range(self.n_genes):
            num_assignments[i] = np.sum(np.argmax(spot_probs, axis=1) == i)
        # Loop through genes in order of number of assignments
        for i in np.argsort(num_assignments)[::-1]:
            # get spots where this gene has the highest probability
            gene_i_mask = np.argmax(spot_probs, axis=1) == i
            # add these spots to the matrix
            prob = spot_probs[gene_i_mask, :]
            self.prob_matrix = np.concatenate((self.prob_matrix, prob[np.argsort(prob[:, i])[::-1]]), axis=0)
            spot_ids = gene_g_ids[gene_i_mask]
            spot_ids = spot_ids[np.argsort(prob[:, i])[::-1]]
            self.spot_ids = np.concatenate((self.spot_ids, spot_ids))
        # update num assignments
        self.num_assignments = num_assignments

    def plot_gene_probs(self, event=None):
        # Plot the prob image
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            self.ax.clear()
        # Load the gene probabilities
        self.load_gene_probs()
        # Plot the matrix. Make sure that the image does not go further than 0.9 right to the edge of the plot
        self.ax.imshow(self.prob_matrix, cmap='viridis', aspect='auto', interpolation='none')
        # Add white horizontal lines to separate the genes. Do this by looping through num_assignments in descending
        # order and adding a line at the end of each gene
        num_assignments = self.num_assignments[np.argsort(self.num_assignments)[::-1]]
        for i in range(self.n_genes):
            self.ax.axhline(np.sum(num_assignments[:i + 1]) - 0.5, color='w', linewidth=1, linestyle='--',
                            alpha=max(0.1, 1 - i / 10))

        self.ax.set_xlabel('Gene')
        self.ax.set_ylabel('Spot')
        self.fig.subplots_adjust(right=0.9)
        self.ax.set_xticks(np.arange(self.n_genes), self.nb.call_spots.gene_names, rotation=90)
        self.ax.set_title('Spot probabilities for gene ' + self.nb.call_spots.gene_names[self.gene_no])
        self.add_gene_slider()
        self.add_widgets()
        self.fig.canvas.draw()

    def add_gene_slider(self):
        # Add a slider to select the gene. As we are moving the slider, we want to show the name of the gene
        # corresponding to the slider value. This will be displayed in the text box below the slider.
        # We do not want to update the scatter plot as we move the slider, as this will be slow. Instead, we will
        # add a button to update the scatter plot once we have selected the gene.
        # First, create the slider. Make it vertical and put it on the right of the plot.
        self.gene_slider_ax = self.fig.add_axes([0.90, 0.15, 0.05, 0.7])
        self.gene_slider = Slider(self.gene_slider_ax, 'Gene', 0, self.n_genes - 1, valinit=self.gene_no,
                                  valstep=1, orientation="vertical")
        self.gene_slider.on_changed(self.update_gene)
        self.gene_text_ax = self.fig.add_axes([0.90, 0.05, 0.05, 0.05])
        # remove x and y ticks from button axes and gene text axes
        self.gene_text_ax.set_xticks([])
        self.gene_text_ax.set_yticks([])
        self.gene_text = self.gene_text_ax.text(0.5, 0.5, self.nb.call_spots.gene_names[self.gene_no],
                                                horizontalalignment='center', verticalalignment='center')
        self.plot_button_ax = self.fig.add_axes([0.90, 0.9, 0.05, 0.05])
        self.plot_button_ax.set_xticks([])
        self.plot_button_ax.set_yticks([])
        # Now create a clickable button to update the scatter plot, make the button black
        self.plot_button = Button(self.plot_button_ax, 'Plot', color='k')
        # Now link the click event to the function to update the scatter plot
        self.plot_button.on_clicked(self.plot_gene_probs)

    def add_score_plot(self, spot_no: int):
        """
        Want to add plot of score for spot for each gene.
        Args:
            spot_no: spot number to plot
        """
        # First, we need to get the scores for this spot
        scores = np.zeros(self.n_genes)
        bled_codes = self.nb.call_spots.bled_codes_ge[:, :, self.nb.basic_info.use_channels]
        spot = self.nb.ref_spots.colors[spot_no][:, self.nb.basic_info.use_channels]
        background = self.nb.ref_spots.background_strength[spot_no][np.newaxis, :]
        background = np.repeat(background, self.nb.basic_info.n_rounds, axis=0)
        spot = spot - background
        spot = spot / self.nb.call_spots.color_norm_factor[:, self.nb.basic_info.use_channels]
        spot = spot / np.linalg.norm(spot)
        for i in range(self.n_genes):
            scores[i] = np.sum(spot * bled_codes[i])

        # Now we can plot the scores for each gene. We want to plot this in a new window, so we need to create a new
        # figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.bar(np.arange(self.n_genes), scores)
        ax.set_xticks(np.arange(self.n_genes), self.nb.call_spots.gene_names, rotation=90, fontsize=8)
        ax.set_ylabel('Score')
        ax.set_xlabel('Gene')
        ax.set_title('Scores for spot ' + str(spot_no))
        fig.tight_layout()
        fig.canvas.draw()

    def update_gene(self, val):
        # Update the gene number and gene name
        self.gene_no = int(val)
        self.gene_text.set_text(self.nb.call_spots.gene_names[self.gene_no])
        # Update the scatter plot
        self.fig.canvas.draw()

    def add_rectangle(self, index):
        # We need to remove any existing rectangles from the plot
        index = max(0, index)
        index = min(index, self.prob_matrix.shape[0] - 1)
        for rectangle in self.ax.patches:
            rectangle.remove()
        # We can then add a new rectangle to the plot
        self.ax.add_patch(
            Rectangle((-0.5, index - 0.5), self.n_genes, 1, fill=False, edgecolor='white'))

    def add_widgets(self):
        # Initialise buttons and cursors
        # 1. We would like each row of the plot to be clickable, so that we can view the observed spot.
        mplcursors.cursor(self.ax, hover=False).connect(
            "add", lambda sel: self.row_clicked(sel))
        # 2. We would like to add a white rectangle around the observed spot when we hover over it
        mplcursors.cursor(self.ax, hover=2).connect(
            "add", lambda sel: self.add_rectangle(sel.index[0]))

    def row_clicked(self, event):
        # When a row is clicked, we want to view the observed spot
        view_codes(self.nb, self.spot_ids[event.index[0]])
        self.add_score_plot(self.spot_ids[event.index[0]])
