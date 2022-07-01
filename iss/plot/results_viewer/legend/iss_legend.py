import napari
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

# MPL background
plt.style.use('dark_background')

# Subplot settings
left = 0.05
right = 0.95
bottom = 0.05
top = 0.95
hspace = 0.05


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


def gene_ax(ax, genes):

    ax.set_title('Gene legend')

    for g in genes.index:

        gene_color = (genes.loc[g, 'ColorR'], genes.loc[g, 'ColorG'], genes.loc[g, 'ColorB'])

        x = 0.05 + (g // 26) * .5
        y = 26 - (g - (26 * (g // 26)))

        m = genes.loc[g, 'mpl_symbol']

        ax.scatter(x=x, y=y, marker=m, facecolor=gene_color, s=50)
        ax.text(x=x + .05, y=y - .3, s=genes.loc[g, 'GeneNames'][:6], c=gene_color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim((-.05, 2.6))

    return ax


def add_legend(viewer, genes, cells, celltype=True, gene=True):

    mpl_widget = FigureCanvas(Figure(figsize=(4, 12)))

    if celltype and gene:

        ax = mpl_widget.figure.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1],
                                                           'hspace': hspace,
                                                           'top': top,
                                                           'bottom': bottom,
                                                           'left': left,
                                                           'right': right})
        ax[0] = cell_type_ax(ax[0], cells=cells)
        ax[1] = gene_ax(ax[1], genes=genes)

    elif celltype and not gene:

        ax = mpl_widget.figure.subplots(gridspec_kw={'top': top,
                                                     'bottom': bottom,
                                                     'left': left,
                                                     'right': right})

        cell_type_ax(ax, cells=cells)

    elif not celltype and gene:

        ax = mpl_widget.figure.subplots(gridspec_kw={'top': top,
                                                     'bottom': bottom,
                                                     'left': left,
                                                     'right': right})

        gene_ax(ax, genes=genes)

    else:
        print('You need to select either celltype or gene')

    viewer.window.add_dock_widget(mpl_widget)


if __name__ == "__main__":

    # Load files
    legend_folder = os.path.dirname(os.path.realpath(__file__))
    genes = pd.read_csv(os.path.join(legend_folder, 'gene_color.csv'))
    cells = pd.read_csv(os.path.join(legend_folder, 'cell_color.csv'))

    viewer = napari.Viewer()
    add_legend(viewer, celltype=True, gene=False, genes=genes, cells=cells)
    napari.run()
