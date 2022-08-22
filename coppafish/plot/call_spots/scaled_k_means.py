import numpy as np
from matplotlib import pyplot as plt
from ...setup import Notebook
from .score_calc import background_fitting
from ...call_spots.bleed_matrix import get_bleed_matrix

plt.style.use('dark_background')


def view_scaled_k_means(nb: Notebook, r: int = 0, check: bool = False):
    """
    Plot to show how `scaled_k_means` was used to compute the bleed matrix.
    There will be upto 3 columns, each with 2 plots.

    The vector for dye $d$ in the `bleed_matrix` is computed from all the spot round vectors
    whose dot product to the dye $d$ vector was the highest.
    The boxplots in the first row show these dot product values for each dye.

    The second row then shows the bleed matrix at each stage of the computation.

    The first column shows the initial bleed matrix. The second column shows the bleed matrix after running
    `scaled_k_means` once with a score threshold of 0. The third column shows the final `bleed_matrix` after running
    `scaled_k_means` a second time with `score_thresh` for dye $d$ set to the median of the scores assigned to
    dye $d$ in the first run. Third column only present if `config['call_spots']['bleed_matrix_anneal']==True`.

    Args:
        nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
        r: Round of bleed matrix to view. Only relevant if `config['call_spots']['bleed_matrix_method'] = 'separate'`.
        check: If True, will raise error if `bleed_matrix` computed here is different to that saved in notebook
    """
    # Fit background to spot_colors as is done in call_reference_spots before bleed_matrix calc
    spot_colors = background_fitting(nb, 'ref')[1]

    # Get bleed matrix and plotting info
    rcd_ind = np.ix_(nb.basic_info.use_rounds, nb.basic_info.use_channels, nb.basic_info.use_dyes)
    initial_bleed_matrix = nb.call_spots.initial_bleed_matrix[rcd_ind]
    config = nb.get_config()['call_spots']
    if config['bleed_matrix_method'].lower() == 'separate':
        r_ind = int(np.where(np.asarray(nb.basic_info.use_rounds) == r)[0])
        title_start = f"Bleed matrix for round {r} "
    else:
        r_ind = 0
        title_start = "Bleed matrix "
    debug_info = get_bleed_matrix(spot_colors[nb.ref_spots.isolated], initial_bleed_matrix,
                                  config['bleed_matrix_method'], config['bleed_matrix_score_thresh'],
                                  config['bleed_matrix_min_cluster_size'], config['bleed_matrix_n_iter'],
                                  config['bleed_matrix_anneal'], r_ind)[1]
    if check:
        if np.abs(nb.call_spots.bleed_matrix[rcd_ind][r_ind]-debug_info['bleed_matrix'][-1]).max() > 1e-3:
            raise ValueError("Bleed Matrix saved to Notebook is different from that computed here "
                             "with get_bleed_matrix.\nMake sure that the background and bleed_matrix"
                             "parameters in the config file have not changed.")

    # Make it so all bleed matrices have same L2 norm as final one
    bm_norm = np.linalg.norm(debug_info['bleed_matrix'][-1])
    bleed_matrix = [bm * bm_norm / np.linalg.norm(bm) for bm in debug_info['bleed_matrix']]
    vmax = np.max(bleed_matrix)

    # Set up plot
    n_plots, n_use_channels, n_use_dyes = debug_info['bleed_matrix'].shape
    fig, ax = plt.subplots(2, n_plots, figsize=(11, 7), sharex=True)
    ax[0, 0].get_shared_y_axes().join(ax[0, 0], ax[0, 1])
    subplot_adjust = [0.075, 0.92, 0.08, 0.88]
    fig.subplots_adjust(left=subplot_adjust[0], right=subplot_adjust[1], bottom=subplot_adjust[2],
                        top=subplot_adjust[3])
    titles = ['Initial Bleed Matrix', 'Bleed Matrix after Scaled K Means 1', 'Bleed Matrix after Scaled K Means 2']
    if not config['bleed_matrix_anneal']:
        titles[1] = 'Bleed Matrix after Scaled K Means'
    for i in range(n_plots):
        box_data = [debug_info['cluster_score'][i][debug_info['cluster_ind'][i] == d] for d in range(n_use_dyes)]
        bp = ax[0, i].boxplot(box_data, notch=0, sym='+', patch_artist=True)
        for d in range(n_use_dyes):
            ax[0, i].text(d + 1, np.percentile(box_data[d], 25), "{:.1e}".format(len(box_data[d])),
                          horizontalalignment='center', color=bp['medians'][d].get_color(), size=5, clip_on=True)
            im = ax[1, i].imshow(bleed_matrix[i], extent=[0.5, n_use_dyes + 0.5, -0.5, n_use_channels - 0.5],
                                 aspect='auto', vmin=0, vmax=vmax)
        if nb.basic_info.dye_names is None:
            ax[1, i].set_xticks(ticks=np.arange(1, n_use_dyes + 1), labels=nb.basic_info.use_dyes)
        else:
            subplot_adjust[2] = 0.15
            fig.subplots_adjust(bottom=subplot_adjust[2])
            ax[1, i].set_xticks(ticks=np.arange(1, n_use_dyes + 1),
                                labels=np.asarray(nb.basic_info.dye_names)[nb.basic_info.use_dyes], rotation=45)
        ax[0, i].set_title(titles[i], fontsize=10)
        if i == 0:
            ax[0, i].set_ylabel('Dot Product to Best Dye Vector')
            ax[1, i].set_ylabel('Channel')
    fig.supxlabel('Dye', size=12)
    mid_point = (subplot_adjust[2] + subplot_adjust[3]) / 2
    cbar_ax = fig.add_axes([subplot_adjust[1] + 0.01, subplot_adjust[2],
                            0.005, mid_point - subplot_adjust[2] - 0.04])  # left, bottom, width, height
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle(f'{title_start}at {n_plots} different stages\n Box plots showing the dot product of spot round '
                 f'vectors with the dye vector they best matched to in the bleed matrix', size=11)
    ax[1, 1].set_title('Bleed Matrix where each dye column was computed from all vectors assigned to that dye '
                       'in the boxplot above',
                       size=11)
    plt.show()
