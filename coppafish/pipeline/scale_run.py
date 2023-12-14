import numpy as np
import warnings

from ..setup.notebook import NotebookPage
from .. import utils, extract, scale


def compute_scale(config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage) -> NotebookPage:
    """
    This calculates the scale factors for the filtered sequencing images and anchor images to convert the float types 
    that are produced after filtering to integers that take the full range of `np.uint16` which the filtered images 
    will be saved as when 'filter' is run.

    Args:
        config (dict): dictionary obtained from 'scale' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.

    Returns:
        `NotebookPage[scale]` page containing image scale factors which are needed by 'extract' in the pipeline and may 
            be useful for debugging purposes.

    Notes:
        - See 'scale' sections of `notebook_comments.json` file for description of the variables that are stored.
    """
    # Check scaling won't cause clipping when saving as uint16
    scale_norm_max = np.iinfo(np.uint16).max - nbp_basic.tile_pixel_value_shift
    if config["scale_norm"] >= scale_norm_max:
        raise ValueError(
            f"\nconfig['extract']['scale_norm'] = {config['scale_norm']} but it must be below " f"{scale_norm_max}"
        )

    nbp = NotebookPage("scale")
    
    if config["r1"] is None:
        config["r1"] = extract.base.get_pixel_length(config["r1_auto_microns"], nbp_basic.pixel_size_xy)
    if config["r2"] is None:
        config["r2"] = config["r1"] * 2
    filter_kernel = utils.morphology.hanning_diff(config["r1"], config["r2"])

    if config["r_smooth"] is not None:
        if len(config["r_smooth"]) == 2:
            if nbp_basic.is_3d:
                warnings.warn(
                    f"Running 3D pipeline but only 2D smoothing requested with r_smooth" f" = {config['r_smooth']}."
                )
        elif len(config["r_smooth"]) == 3:
            if not nbp_basic.is_3d:
                raise ValueError("Running 2D pipeline but 3D smoothing requested.")
        else:
            raise ValueError(
                f"r_smooth provided was {config['r_smooth']}.\n"
                f"But it needs to be a 2 radii for 2D smoothing or 3 radii for 3D smoothing.\n"
                f"I.e. it is the wrong shape."
            )
        if config["r_smooth"][0] > config["r2"]:
            raise ValueError(
                f"Smoothing radius, {config['r_smooth'][0]}, is larger than the outer radius of the\n"
                f"hanning filter, {config['r2']}, making the filtering step redundant."
            )

        # smooth_kernel = utils.strel.fspecial(*tuple(config['r_smooth']))
        smooth_kernel = np.ones(tuple(np.array(config["r_smooth"], dtype=int) * 2 - 1))
        smooth_kernel = smooth_kernel / np.sum(smooth_kernel)

        if np.max(config["r_smooth"]) == 1:
            warnings.warn("Max radius of smooth filter was 1, so not using.")
            config["r_smooth"] = None
    

    # check to see if scales have already been computed
    config["scale"], config["scale_anchor"] = scale.base.get_scale_from_txt(
        nbp_file.scale, config["scale"], config["scale_anchor"]
    )

    if config["scale"] is None and len(nbp_file.round) > 0:
        # ensure scale_norm value is reasonable so max pixel value in tiff file is significant factor of max of uint16
        scale_norm_min = (np.iinfo("uint16").max - nbp_basic.tile_pixel_value_shift) / 5
        scale_norm_max = np.iinfo("uint16").max - nbp_basic.tile_pixel_value_shift
        if not scale_norm_min <= config["scale_norm"] <= scale_norm_max:
            raise utils.errors.OutOfBoundsError("scale_norm", config["scale_norm"], scale_norm_min, scale_norm_max)
        # If using smoothing, apply this as well before scale
        if config["r_smooth"] is None:
            smooth_kernel_2d = None
        else:
            if smooth_kernel.ndim == 3:
                # take central plane of smooth filter if 3D as scale is found from a single z-plane.
                smooth_kernel_2d = smooth_kernel[:, :, config["r_smooth"][2] - 1]
            else:
                smooth_kernel_2d = smooth_kernel.copy()
            # smoothing is averaging so to average in 2D, need to re normalise filter
            smooth_kernel_2d = smooth_kernel_2d / np.sum(smooth_kernel_2d)
            if np.max(config["r_smooth"][:2]) <= 1:
                smooth_kernel_2d = None  # If dimensions of 2D kernel are [1, 1] is equivalent to no smoothing
        print("Computing scale...")
        nbp.scale_tile, nbp.scale_channel, nbp.scale_z, config["scale"] = scale.base.get_scale(
            nbp_file,
            nbp_basic,
            0,
            nbp_basic.use_tiles,
            nbp_basic.use_channels,
            nbp_basic.use_z,
            config["scale_norm"],
            filter_kernel,
            smooth_kernel_2d,
        )
    else:
        nbp.scale_tile = None
        nbp.scale_channel = None
        nbp.scale_z = None
        smooth_kernel_2d = None
    nbp.scale = config["scale"]
    
    if nbp_basic.use_anchor == False:
        nbp.scale_anchor_tile = None
        nbp.scale_anchor_z = None
        nbp.scale_anchor = None
    elif config["scale_anchor"] is None:
        print("Computing scale_anchor...")
        nbp.scale_anchor_tile, _, nbp.scale_anchor_z, config["scale_anchor"] = scale.base.get_scale(
            nbp_file,
            nbp_basic,
            nbp_basic.anchor_round,
            nbp_basic.use_tiles,
            [nbp_basic.anchor_channel],
            nbp_basic.use_z,
            config["scale_norm"],
            filter_kernel,
            smooth_kernel_2d,
        )

    # Save scale values to disk in case need to re-run
    scale.base.save_scale(nbp_file.scale, nbp.scale, config["scale_anchor"])
    
    nbp.r_smooth = config["r_smooth"]
    nbp.r1 = config["r1"]
    nbp.r2 = config["r2"]
    nbp.scale_anchor = config["scale_anchor"]
    
    return nbp
