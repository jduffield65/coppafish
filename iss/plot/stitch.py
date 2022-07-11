import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.ndimage


def interpolate_array(array, invalid_value):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Args:
        array:
        invalid_value:

    Returns:

    """
    ind = scipy.ndimage.distance_transform_edt(array == invalid_value, return_distances=False, return_indices=True)
    return array[tuple(ind)]


def get_plot_images_from_shifts(shifts, scores):
    min_yxz = np.min(shifts, axis=0)
    shifts_mod = shifts - min_yxz
    im_size = shifts_mod.max(axis=0) + 1
    image = np.zeros(im_size)
    image[tuple([shifts_mod[:,i] for i in range(im_size.size)])] = scores
    extent_ax_order = [1, 0]  # x is 1st index in im_show(extent=...)
    if im_size.size == 3:
        extent_ax_order = extent_ax_order + [2]
    extent = [[min_yxz[i] - 0.5, min_yxz[i] + im_size[i] - 0.5] for i in extent_ax_order]
    extent = sum(extent, [])
    # 1st y extent needs to be max and 2nd min as second index indicates top row which corresponds to min_y
    # for our image.
    extent[2], extent[3] = extent[3], extent[2]
    return image, extent


def view_shifts(best_shift, best_score, score_thresh, shifts_2d, scores_2d, shifts_3d, scores_3d):
    image_2d, extent_2d = get_plot_images_from_shifts(np.rint(shifts_2d).astype(int), scores_2d)
    image_2d = interpolate_array(image_2d, 0)  # replace 0 with nearest neighbor value
    fig = plt.figure()
    v_max = np.max([image_2d.max(), 1.2 * score_thresh])
    v_min = image_2d.min()
    if shifts_3d is not None:
        images_3d, extent_3d = get_plot_images_from_shifts(np.rint(shifts_3d).astype(int), scores_3d)
        images_3d = interpolate_array(images_3d, 0)  # replace 0 with nearest neighbor value
        if images_3d.max() > v_max:
            v_max = images_3d.max()
        if images_3d.min() < v_min:
            v_min = images_3d.min()
        cmap_norm = matplotlib.colors.TwoSlopeNorm(vmin=v_min, vcenter=score_thresh, vmax=v_max)
        n_cols = images_3d.shape[2]
        if n_cols > 13:
            # If loads of z-planes, just show the 13 with the largest score
            n_cols = 13
            max_score_z = images_3d.max(axis=(0, 1))
            max_score_z_thresh = max_score_z[np.argpartition(max_score_z, -n_cols)[-n_cols]]
            use_z = np.where(max_score_z >= max_score_z_thresh)[0]
        else:
            use_z = np.arange(n_cols)

        plot_2d_height = int(np.ceil(n_cols / 4))
        plot_3d_height = n_cols - plot_2d_height
        ax_2d = plt.subplot2grid(shape=(n_cols, n_cols), loc=(0, 0), colspan=n_cols, rowspan=plot_3d_height)
        ax_3d = [plt.subplot2grid(shape=(n_cols, n_cols), loc=(plot_3d_height + 1, i), rowspan=plot_2d_height) for i in range(n_cols)]
        for i in range(n_cols):
            # share axes for 3D plots
            ax_3d[i].get_shared_y_axes().join(ax_3d[i], *ax_3d)
            ax_3d[i].get_shared_x_axes().join(ax_3d[i], *ax_3d)
            ax_3d[i].imshow(images_3d[:, :, use_z[i]], extent=extent_3d[:4], aspect='auto', cmap='bwr', norm=cmap_norm)
            z_plane = int(np.rint(extent_3d[4] + use_z[i] + 0.5))
            ax_3d[i].set_title(f'Z = {z_plane}')
            if i > 0:
                ax_3d[i].tick_params(labelbottom=False, labelleft=False)
            if z_plane == best_shift[2]:
                ax_3d[i].plot(best_shift[1], best_shift[0], 'kx')

        fig.supxlabel('X')
        fig.supylabel('Y')
    else:
        n_cols = 1
        ax_2d = plt.subplot2grid(shape=(n_cols, n_cols), loc=(0, 0))
        ax_2d.set_xlabel('X')
        ax_2d.set_ylabel('Y')
        cmap_norm = matplotlib.colors.TwoSlopeNorm(vmin=v_min, vcenter=score_thresh, vmax=v_max)

    im_2d = ax_2d.imshow(image_2d, extent=extent_2d, aspect='auto', cmap='bwr', norm=cmap_norm)
    ax_2d.plot(best_shift[1], best_shift[0], 'kx')
    ax_2d.invert_yaxis()
    ax_2d.set_title('Approx number of neighbours found for all shifts')
    fig.subplots_adjust(left=0.07, right=0.85, bottom=0.07, top=0.95)
    cbar_ax = fig.add_axes([0.9, 0.07, 0.03, 0.9])  # left, bottom, width, height
    fig.colorbar(im_2d, cax=cbar_ax)
    plt.show()
