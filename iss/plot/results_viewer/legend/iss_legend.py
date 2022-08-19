import napari
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import numpy as np
from typing import Optional

# MPL background
plt.style.use('dark_background')

# Subplot settings
left = 0.05
right = 0.95
bottom = 0.05
top = 0.95
hspace = 0.05
n_labels_per_column = 26
n_gene_label_letters = 6   # max number of letters in gene label legend


def cell_type_ax(ax, cells):

    ax.set_title('Celltype legend')

    for c in cells.index:

        x = 0.05 + ((c // 25) * 2)
        y = 25 - (c - (25 * (c // 25)))

        rect = patches.Rectangle((x, y), 0.12, 0.7, facecolor=cells.loc[c, 'color'], edgecolor='w', linewidth=.5)
        ax.add_patch(rect)

        ax.text(x + .15, y - .02, s=cells.loc[c, 'className'])

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim((-.05, 6))
    ax.set_ylim((0, 26))

    return ax


def gene_ax(ax: plt.Axes, gene_legend_info: pd.DataFrame, genes: np.ndarray):

    ax.set_title('Gene legend')

    added_ind = 0
    n_columns = int(np.ceil(genes.size/n_labels_per_column))
    n_genes_per_column = int(np.ceil(genes.size/n_columns))
    for g in gene_legend_info.index:
        if np.isin(gene_legend_info['GeneNames'][g], genes):
            gene_color = (gene_legend_info.loc[g, 'ColorR'], gene_legend_info.loc[g, 'ColorG'],
                          gene_legend_info.loc[g, 'ColorB'])

            x = 0.05 + (added_ind // n_genes_per_column)/n_columns * 2.5
            y = n_genes_per_column - (added_ind - (n_genes_per_column * (added_ind // n_genes_per_column)))

            m = gene_legend_info.loc[g, 'mpl_symbol']

            ax.scatter(x=x, y=y, marker=m, facecolor=gene_color, s=50)
            ax.text(x=x + .05, y=y - .3, s=gene_legend_info.loc[g, 'GeneNames'][:n_gene_label_letters], c=gene_color)
            added_ind += 1

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim((-.05, x+0.5))  # last x value is the maximum

    return ax


def add_legend(gene_legend_info: Optional[pd.DataFrame],
               genes: Optional[np.ndarray] = None,
               cell_legend_info: Optional[pd.DataFrame] = None) -> [FigureCanvas, plt.Axes, int]:
    """
    This returns a legend which displays the genes and/or cell types present.

    Args:
        gene_legend_info: `[n_legend_genes x 6]` pandas data frame containing the following information for each gene
            - GeneNames - str, name of gene with first letter capital
            - ColorR - float, Rgb color for plotting
            - ColorG - float, rGb color for plotting
            - ColorB - float, rgB color for plotting
            - napari_symbol - str, symbol used to plot in napari
            - mpl_symbol - str, equivalent of napari symbol in matplotlib.
        genes: str [n_genes]
            Genes in current experiment.
        cell_legend_info: [n_legend_cell_types x 3] pandas data frame containing the following information for each
            cell type
            - className
            - IdentifiedType
            - color

    Returns:
        - mpl_widget - figure of legend.
        - ax - axes containing info about gene symbols in `ax.collections` and gene labels in `ax.texts`.
        - n_gene_label_letters - max number of letters in each gene label in the legend.
    """
    if genes is None and gene_legend_info is not None:
        genes = np.asarray(gene_legend_info['GeneNames'])  # show all genes in legend if not given
    if genes is not None:
        # make x dimension of figure equal to number of columns in legend
        fig_size = [int(np.ceil(genes.size/n_labels_per_column)), 4]
    else:
        fig_size = [5, 4]
    mpl_widget = FigureCanvas(Figure(figsize=fig_size))
    if cell_legend_info is not None and gene_legend_info is not None:

        ax = mpl_widget.figure.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1],
                                                           'hspace': hspace,
                                                           'top': top,
                                                           'bottom': bottom,
                                                           'left': left,
                                                           'right': right})
        ax[0] = cell_type_ax(ax[0], cells=cell_legend_info)
        ax[1] = gene_ax(ax[1], gene_legend_info, genes)

    elif cell_legend_info is not None and gene_legend_info is None:

        ax = mpl_widget.figure.subplots(gridspec_kw={'top': top,
                                                     'bottom': bottom,
                                                     'left': left,
                                                     'right': right})

        cell_type_ax(ax, cells=cell_legend_info)

    elif cell_legend_info is None and gene_legend_info is not None:

        ax = mpl_widget.figure.subplots(gridspec_kw={'top': top,
                                                     'bottom': bottom,
                                                     'left': left,
                                                     'right': right})

        gene_ax(ax, gene_legend_info, genes)

    else:
        print('Both gene_legend_info and cell_legend_info are None so no legend added.')
        return None

    return mpl_widget, ax, n_gene_label_letters


if __name__ == "__main__":

    # Load files
    legend_folder = os.path.dirname(os.path.realpath(__file__))
    genes = pd.read_csv(os.path.join(legend_folder, 'gene_color.csv'))
    cells = pd.read_csv(os.path.join(legend_folder, 'cell_color.csv'))

    viewer = napari.Viewer()
    fig, ax, _ = add_legend(gene_legend_info=genes, cell_legend_info=cells)
    viewer.window.add_dock_widget(fig, area='right', name='Genes')
    napari.run()
