import numpy as np
from .shift import view_register_search
from ...find_spots import spot_yxz, get_isolated_points
from ...register import get_single_affine_transform
from ...spot_colors.base import apply_transform
from ..stitch import view_point_clouds
from ...setup import Notebook
import distinctipy
from typing import Optional, List
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def view_icp(nb: Notebook, t: int, r: int, c: int):
    """
    Function to plot results of iterative closest point to find affine transform between
    `ref_round/ref_channel` and round `r`, channel `c` for tile `t`.
    Useful for debugging the `register` section of the pipeline.

    Args:
        nb: Notebook containing results of the experiment. Must contain `find_spots` page.
            If contains `register_initial` and/or `register` pages, then transform from these will be used.
        t: tile interested in.
        r: Want to find the transform between the reference round and this round.
        c: Want to find the transform between the reference channel and this channel.
    """
    config = nb.get_config()['register']
    if nb.basic_info.is_3d:
        neighb_dist_thresh = config['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['neighb_dist_thresh_2d']
    z_scale = [1, 1, nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy]
    point_clouds = []
    # 1st point cloud is imaging one as does not change
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r, c)]
    # only keep isolated spots, those whose second neighbour is far away
    # Do this only for imaging point cloud as that is what is done in pipeline/register
    isolated = get_isolated_points(point_clouds[0] * z_scale, 2 * neighb_dist_thresh)
    point_clouds[0] = point_clouds[0][isolated]

    # 2nd is untransformed reference point cloud
    r_ref = nb.basic_info.ref_round
    c_ref = nb.basic_info.ref_channel
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r_ref, c_ref)]
    z_scale = z_scale[2]

    # Add shifted reference point cloud
    if nb.has_page('register_initial'):
        shift = nb.register_initial.shift[t, r]
    else:
        shift = view_register_search(nb, t, r, return_shift=True)
    point_clouds = point_clouds + [point_clouds[1] + shift]

    # Add reference point cloud transformed by an affine transform
    transform_outlier = None
    if nb.has_page('register'):
        transform = nb.register.transform[t, r, c]
        if nb.has_page('register_debug'):
            # If particular tile/round/channel was found by regularised least squares
            transform_outlier = nb.register_debug.transform_outlier[t, r, c]
            if np.abs(transform_outlier).max() == 0:
                transform_outlier = None
    else:
        start_transform = np.eye(4, 3)  # no scaling just shift to start off icp
        start_transform[3] = shift * [1, 1, z_scale]
        transform = get_single_affine_transform(point_clouds[1], point_clouds[0], z_scale, z_scale,
                                                start_transform, neighb_dist_thresh, nb.basic_info.tile_centre,
                                                config['n_iter'])[0]

    if not nb.basic_info.is_3d:
        # use numpy not jax.numpy as reading in tiff is done in numpy.
        tile_sz = np.array([nb.basic_info.tile_sz, nb.basic_info.tile_sz, 1], dtype=np.int16)
    else:
        tile_sz = np.array([nb.basic_info.tile_sz, nb.basic_info.tile_sz, nb.basic_info.nz], dtype=np.int16)

    if transform_outlier is not None:
        point_clouds = point_clouds + [apply_transform(point_clouds[1], transform_outlier, nb.basic_info.tile_centre,
                                                       z_scale, tile_sz)[0]]

    point_clouds = point_clouds + [apply_transform(point_clouds[1], transform, nb.basic_info.tile_centre, z_scale,
                                                   tile_sz)[0]]
    pc_labels = [f'Imaging: r{r}, c{c}', f'Reference: r{r_ref}, c{c_ref}', f'Reference: r{r_ref}, c{c_ref} - Shift',
                 f'Reference: r{r_ref}, c{c_ref} - Affine']
    if transform_outlier is not None:
        pc_labels = pc_labels + [f'Reference: r{r_ref}, c{c_ref} - Regularized']
    view_point_clouds(point_clouds, pc_labels, neighb_dist_thresh, z_scale,
                      f'Transform of tile {t} to round {r}, channel {c}')
    plt.show()


def view_icp_reg(nb: Notebook, t: int, r: int, c: int, reg_constant: Optional[List] = None,
                 reg_factor: Optional[List] = None, reg_transform: Optional[np.ndarray] = None,
                 start_transform: Optional[np.ndarray] = None, plot_residual: bool = False):
    """
    Function to plot how regularisation changes the affine transform found through iterative closest point between
    `ref_round/ref_channel` and round $r$, channel $c$ for tile $t$.

    Useful for finding suitable values for `config['register']['regularize_constant']`
    and `config['register']['regularize_factor']`.


    Args:
        nb: Notebook containing results of the experiment. Must contain `find_spots` page.
            Must contain `register_initial`/`register_debug` pages if `start_transform`/`reg_transform` not specified.
        t: tile interested in.
        r: Want to find the transform between the reference round and this round.
        c: Want to find the transform between the reference channel and this channel.
        reg_constant: `int [n_reg]`
            Constant used when doing regularized least squares.
            Will be a point cloud produced for affine transformed produced with each of these parameters.
            Value will be indicated by $\lambda$ in legend/buttons.
            If `None`, will use `config['register']['regularize_constant']`.
        reg_factor: `float [n_reg]`
            Factor to boost rotation/scaling term in regularized least squares.
            Must be same length as `reg_constant`.
            Value will be indicated by $\mu$ in legend/buttons.
            If `None`, will use `config['register']['regularize_factor']`.

            The regularized term in the loss function for finding the transform is:

            $0.5\lambda (\mu D_{scale}^2 + D_{shift}^2)$

            Where:

            * $D_{scale}^2$ is the squared distance between `transform[:3, :]` and `reg_transform[:3, :]`.
            I.e. the squared difference of the scaling/rotation part of the transform from the target.
            * $D_{shift}^2$ is the squared distance between `transform[3]` and `reg_transform[3]`.
            I.e. the squared difference of the shift part of the transform from the target.
            * $\lambda$ is `reg_constant` - the larger it is, the smaller $D_{scale}^2$ and $D_{shift}^2$.
            * $\mu$ is `reg_factor` - the larger it is, the smaller $D_{scale}^2$.
        reg_transform: Transform to regularize towards i.e. the expected transform.
            If not specified, will use average transformation based on `nb.register_debug.av_scaling[c]`
            and `nb.register_debug.av_shifts[t, r]` with no rotation.
            This is the same as what is used in `coppafish.register.icp`.
        start_transform: Initial transform to use as starting point for ICP.
            If `None`, will use `nb.register_initial.shift[t, r]` for the non-regularized case
            and `reg_transform` for the regularized case to match method used in `coppafish.register.icp`.
        plot_residual: If `True`, will run plot_reg_residual as well.
    """
    config = nb.get_config()['register']
    n_iter = config['n_iter']
    if nb.basic_info.is_3d:
        neighb_dist_thresh = config['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['neighb_dist_thresh_2d']
    z_scale = [1, 1, nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy]
    if not nb.basic_info.is_3d:
        # use numpy not jax.numpy as reading in tiff is done in numpy.
        tile_sz = np.array([nb.basic_info.tile_sz, nb.basic_info.tile_sz, 1], dtype=np.int16)
    else:
        tile_sz = np.array([nb.basic_info.tile_sz, nb.basic_info.tile_sz, nb.basic_info.nz], dtype=np.int16)
    point_clouds = []
    # 1st point cloud is imaging one as does not change
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r, c)]
    # only keep isolated spots, those whose second neighbour is far away
    # Do this only for imaging point cloud as that is what is done in pipeline/register
    isolated = get_isolated_points(point_clouds[0] * z_scale, 2 * neighb_dist_thresh)
    point_clouds[0] = point_clouds[0][isolated]

    # 2nd is untransformed reference point cloud
    r_ref = nb.basic_info.ref_round
    c_ref = nb.basic_info.ref_channel
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r_ref, c_ref)]
    z_scale = z_scale[2]

    # 3rd is ref point cloud transformed according to affine transform with no regularisation
    if start_transform is None:
        # no scaling just shift to start off icp as used in pipeline if no start_transform given
        shift = nb.register_initial.shift[t, r]
        start_transform = np.eye(4, 3)
        start_transform[3] = shift * [1, 1, z_scale]
    transform_no_reg = get_single_affine_transform(point_clouds[1], point_clouds[0], z_scale, z_scale,
                                                   start_transform, neighb_dist_thresh, nb.basic_info.tile_centre,
                                                   n_iter)[0]
    point_clouds = point_clouds + [apply_transform(point_clouds[1], transform_no_reg, nb.basic_info.tile_centre, z_scale,
                                                   tile_sz)[0]]

    # 4th is ref point cloud transformed according to target reg transform
    if reg_transform is None:
        # If reg transform not specified, use av scaling for c and av shift for t, r.
        # Also set start_transform to reg_transform if not specified
        # This is same as what is done in `coppafish.register.base.icp` for t/r/c where regularisation required.
        reg_transform = np.eye(4, 3) * nb.register_debug.av_scaling[c]
        reg_transform[3] = nb.register_debug.av_shifts[t, r]
        start_transform = reg_transform.copy()
    point_clouds = point_clouds + [apply_transform(point_clouds[1], reg_transform, nb.basic_info.tile_centre, z_scale,
                                                   tile_sz)[0]]
    pc_labels = [f'Imaging: r{r}, c{c}', f'Reference: r{r_ref}, c{c_ref}',
                 r'$\lambda=0$', r'$\lambda=\infty$']

    # Now add ref point cloud transformed according to reg transform found with all reg params given.
    # If None given, use default params.
    if reg_constant is None:
        reg_constant = [config['regularize_constant']]
    if reg_factor is None:
        reg_factor = [config['regularize_factor']]

    n_reg = len(reg_constant)
    if n_reg != len(reg_factor):
        raise ValueError(f"reg_constant and reg_factor need to be same size but they "
                         f"have {n_reg} and {len(reg_factor)} values respectively")
    transforms = np.zeros((n_reg, 4, 3))
    for i in range(n_reg):
        # Deviation in scale/rotation is much less than permitted deviation in shift so boost scale reg constant.
        reg_constant_scale = np.sqrt(0.5 * reg_constant[i] * reg_factor[i])
        reg_constant_shift = np.sqrt(0.5 * reg_constant[i])
        transforms[i] = \
            get_single_affine_transform(point_clouds[1], point_clouds[0], z_scale, z_scale, start_transform,
                                        neighb_dist_thresh, nb.basic_info.tile_centre, n_iter,
                                        reg_constant_scale, reg_constant_shift, reg_transform)[0]
        point_clouds += [apply_transform(point_clouds[1], transforms[i], nb.basic_info.tile_centre,
                                         z_scale, tile_sz)[0]]
        if reg_constant[i] > 1000:
            pc_labels += [r'$\lambda={:.0E},\mu={:.0E}$'.format(int(reg_constant[i]), int(reg_factor[i]))]
        else:
            pc_labels += [r'$\lambda={:.0f},\mu={:.0E}$'.format(int(reg_constant[i]), int(reg_factor[i]))]

    vpc = view_point_clouds(point_clouds, pc_labels, neighb_dist_thresh, z_scale,
                            f'Regularized transform of tile {t} to round {r}, channel {c}')
    n_matches_reg_target = np.sum(vpc.neighb[3] >= 0)

    if plot_residual and n_reg > 1:
        plot_reg_residual(reg_transform, transforms, reg_constant, reg_factor, transform_no_reg, n_matches_reg_target)
    else:
        plt.show()


def plot_reg_residual(reg_transform: np.ndarray, transforms_plot: List[np.ndarray],
                      reg_constant: List, reg_factor: List, transform_no_reg: Optional[np.ndarray] = None,
                      n_matches: Optional[int] = None):
    """
    This shows how changing the regularization parameters affect how close the affine transform
    is to that which it was being regularized towards. E.g. it should show that
    the larger `reg_constant`, the smaller the difference (y-axis values in the plots).

    There will be up to 4 plots, in each, the different colors refer to the different
    `reg_constant`/`reg_factor` combinations and the smaller the y-axis value,
    the closer the transform is to `reg_transform`. The different axis variables in the plot are
    explained in the `reg_factor` variable description below.

    Args:
        reg_transform: `float [4 x 3]`
            Transform which was regularized towards i.e. the expected transform.
        transforms_plot: `[n_reg]`.
            `transforms_plot[i]` is the `[4 x 3]` transform found with regularization parameters
            `reg_constant[i]` and `reg_factor[i]`.
        reg_constant: `int [n_reg]`
            Constant used when doing regularized least squares.
            Value will be indicated by $\lambda$ on x-axis.
        reg_factor: `float [n_reg]`
            Factor to boost rotation/scaling term in regularized least squares.
            Must be same length as `reg_constant`.
            Value will be indicated by $\mu$ on x-axis.

            The regularized term in the loss function for finding the transform is:

            $0.5\lambda (\mu D_{scale}^2 + D_{shift}^2)$

            Where:

            * $D_{scale}^2$ is the squared distance between `transform[:3, :]` and `reg_transform[:3, :]`.
            I.e. the squared difference of the scaling/rotation part of the transform from the target.
            * $D_{shift}^2$ is the squared distance between `transform[3]` and `reg_transform[3]`.
            I.e. the squared difference of the shift part of the transform from the target.
            * $\lambda$ is `reg_constant` - the larger it is, the smaller $D_{scale}^2$ and $D_{shift}^2$.
            * $\mu$ is `reg_factor` - the larger it is, the smaller $D_{scale}^2$.
        transform_no_reg: `float [4 x 3]`.
            Transform found with no regularization. If given, will show white line
            labelled by $\lambda=0$ with y-axis value indicating value with no regularization.
        n_matches: Number of nearest neighbours found for `reg_transform`.
            If given, will show white line labelled by $n_{matches}$ with x-axis value
            equal to this on the plots where $\lambda$ is the x-axis variable.
            This is because we expect the regularization should have more of an effect if `reg_constant > n_matches`,
            i.e. the y-axis variable should go towards zero at x-axis values beyond this line.
    """
    n_reg = len(reg_constant)
    col_info = {}
    col_ind = 0
    if len(np.unique(reg_constant)) > 1:
        col_info[col_ind] = {}
        if len(np.unique(reg_factor)) == 1:
            col_info[col_ind]['title'] = r"Varying $\lambda$ ($\mu = {:.0E}$)".format(int(reg_factor[0]))
        else:
            col_info[col_ind]['title'] = f"Varying $\lambda$"
        col_info[col_ind]['x_label'] = r"$\lambda$"
        col_info[col_ind]['x'] = reg_constant
        col_info[col_ind]['x_lims'] = [int(0.9 * np.min(reg_constant)), int(1.1 * np.max(reg_constant))]
        if n_matches is not None:
            col_info[col_ind]['x_lims'] = [int(0.9 * np.min(list(reg_constant) + [n_matches])),
                                           int(1.1 * np.max(list(reg_constant) + [n_matches]))]
        col_info[col_ind]['log_x'] = np.ptp(col_info[col_ind]['x_lims']) > 100
        col_info[col_ind]['n_matches'] = n_matches
        col_ind += 1
    if len(np.unique(reg_factor)) > 1:
        col_info[col_ind] = {}
        if len(np.unique(reg_constant)) == 1:
            col_info[col_ind]['title'] = rf"Varying $\mu$ ($\lambda = {int(reg_constant[0])}$)"
        else:
            col_info[col_ind]['title'] = f"Varying $\mu$"
        col_info[col_ind]['x_label'] = r"$\mu$"
        col_info[col_ind]['x'] = reg_factor
        col_info[col_ind]['x_lims'] = [int(0.9 * np.min(reg_factor)), int(1.1 * np.max(reg_factor))]
        col_info[col_ind]['log_x'] = np.ptp(col_info[col_ind]['x_lims']) > 100
        col_info[col_ind]['n_matches'] = None

    if n_reg != len(reg_factor):
        raise ValueError(f"reg_constant and reg_factor need to be same size but they "
                         f"have {n_reg} and {len(reg_factor)} values respectively")
    if n_reg != len(transforms_plot):
        raise ValueError(f"reg_constant and transforms_plot need to be same size but they "
                         f"have {n_reg} and {len(transforms_plot)} values respectively")
    if len(col_info) == 0:
        raise ValueError("Not enough data to plot")

    y0 = [np.linalg.norm(transforms_plot[i][:3, :]-reg_transform[:3, :]) for i in range(n_reg)]
    y1 = [np.linalg.norm(transforms_plot[i][3] - reg_transform[3]) for i in range(n_reg)]
    if transform_no_reg is not None:
        y0_no_reg = np.linalg.norm(transform_no_reg[:3, :]-reg_transform[:3,:])
        y1_no_reg = np.linalg.norm(transform_no_reg[3] - reg_transform[3])
        y0_lim = [np.min(y0 + [y0_no_reg]) - 0.0002, np.max(y0 + [y0_no_reg]) + 0.0002]
        y1_lim = [np.min(y1 + [y1_no_reg]) - 0.2, np.max(y1 + [y1_no_reg]) + 0.2]
    else:
        y0_lim = [np.min(y0) - 0.0002, np.max(y0) + 0.0002]
        y1_lim = [np.min(y1) - 0.2, np.max(y1) + 0.2]
    y0_lim = np.clip(y0_lim, 0, np.inf)
    y1_lim = np.clip(y1_lim, 0, np.inf)
    colors = distinctipy.get_colors(n_reg, [(0, 0, 0), (1, 1, 1)])
    fig, ax = plt.subplots(2, len(col_info), figsize=(15, 7))
    if len(col_info) == 1:
        ax = ax[:, np.newaxis]
    for i in range(len(col_info)):
        if transform_no_reg is not None:
            ax[0, i].plot(col_info[i]['x_lims'], [y0_no_reg, y0_no_reg], 'w:')
            ax[0, i].text(np.mean(col_info[i]['x_lims']), y0_no_reg, r"$\lambda = 0$",
                          va='bottom', ha="center", color='w')
            ax[1, i].plot(col_info[i]['x_lims'], [y1_no_reg, y1_no_reg], 'w:')
            ax[1, i].text(np.mean(col_info[i]['x_lims']), y1_no_reg, r"$\lambda = 0$",
                          va='bottom', ha="center", color='w')
        if col_info[i]['n_matches'] is not None:
            ax[0, i].plot([n_matches, n_matches], y0_lim, 'w:')
            ax[0, i].text(n_matches, np.percentile(y0_lim, 10), r"$n_{matches}$",
                          va='bottom', ha="right", color='w', rotation=90)
            ax[1, i].plot([n_matches, n_matches], y1_lim, 'w:')
            ax[1, i].text(n_matches, np.percentile(y1_lim, 10), r"$n_{matches}$",
                          va='bottom', ha="right", color='w', rotation=90)
        ax[0, i].scatter(col_info[i]['x'], y0, color=colors)
        ax[1, i].scatter(col_info[i]['x'], y1, color=colors)
        ax[0, i].get_shared_x_axes().join(ax[0, i], ax[1, i])
        ax[1, i].set_xlabel(col_info[i]['x_label'])
        ax[0, i].set_title(col_info[i]['title'])
        if col_info[i]['log_x']:
            ax[1, i].set_xscale('log')
        ax[1, i].set_xlim(col_info[i]['x_lims'])
        if i == 1:
            ax[0, 0].get_shared_y_axes().join(ax[0, 0], ax[0, 1])
            ax[1, 0].get_shared_y_axes().join(ax[1, 0], ax[1, 1])
        ax[0, 0].set_ylim(y0_lim)
        ax[0, 0].set_ylabel("$D_{scale}$")
        ax[1, 0].set_ylim(y1_lim)
        ax[1, 0].set_ylabel("$D_{shift}$")
    fig.suptitle("How varying regularization parameters affects how similar transform found is to target transform")
    plt.show()
