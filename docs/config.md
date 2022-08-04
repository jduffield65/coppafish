# Default Config Settings
## file_names
The *file_names* section specifies the files that will be used throughout the pipeline. Variables in this section can be changed at any point in the pipeline, and the notebook created using it can still be loaded in.

* **notebook_name**: *str*.

	Name of notebook file in output directory will be *notebook_name*.npz 

	Default: `notebook`

* **input_dir**: *str*.

	Directory where the raw .nd2 files or .npy stacks are 

	Default: `MUST BE SPECIFIED`

* **output_dir**: *str*.

	Directory where notebook is saved 

	Default: `MUST BE SPECIFIED`

* **tile_dir**: *str*.

	Directory where tile .npy files saved 

	Default: `MUST BE SPECIFIED`

* **round**: *maybe_list_str*.

	Names of .nd2 files for the imaging rounds. Leave empty if only using anchor. 

	Default: `None`

* **anchor**: *maybe_str*.

	Name of the file for the anchor round. Leave empty if not using anchor. 

	Default: `None`

* **raw_extension**: *str*.

	.nd2 or .npy indicating the data type of the raw data. 

	Default: `.nd2`

* **raw_metadata**: *maybe_str*.

	If .npy raw_extension, this is the name of the .json file in *input_dir* which contains the metadata required extracted from the initial .nd2 files. I.e. it contains the output of *iss/utils/nd2/save_metadata*: 

	 - `xy_pos` - `List [n_tiles x 2]`. xy position of tiles in pixels. 

	 - `pixel_microns` - `float`. xy pixel size in microns. 

	 - `pixel_microns_z` - `float`. z pixel size in microns. 

	 - `sizes` - dict with fov (`t`), channels (`c`), y, x, z-planes (`z`) dimensions. 

	Default: `None`

* **dye_camera_laser**: *maybe_file*.

	csv file giving the approximate raw intensity for each dye with each camera/laser combination. If not set, the file *iss/setup/dye_camera_laser_raw_intensity.csv* file will be used. 

	Default: `None`

* **code_book**: *str*.

	Text file which contains the codes indicating which dye to expect on each round for each gene. 

	Default: `MUST BE SPECIFIED`

* **scale**: *str*.

	Text file saved in *tile_dir* containing `extract['scale']` and `extract['scale_anchor']` values used to create the tile .npy files in the *tile_dir*. If the second value is 0, it means `extract['scale_anchor']` has not been calculated yet. 

	 If the extract step of the pipeline is re-run with `extract['scale']` or `extract['scale_anchor']` different to values saved here, an error will be raised. 

	Default: `scale`

* **psf**: *str*.

	npy file in output directory indicating average spot shape. If deconvolution required and file does not exist, will be computed automatically in extract step. (this is psf before tapering and scaled to fill uint16 range). 

	Default: `psf`

* **omp_spot_shape**: *str*.

	npy file in *output_dir* indicating average shape in omp coefficient image. It only indicates the sign of the coefficient i.e. only contains -1, 0, 1. If file does not exist, it is computed from the coefficient images of all genes of the central tile. 

	Default: `omp_spot_shape`

* **omp_spot_info**: *str*.

	npy file in *output_dir* containing information about spots found in omp step. After each tile is completed, information will be saved to this file. If file does not exist, it will be saved after first tile of OMP step. 

	Default: `omp_spot_info`

* **omp_spot_coef**: *str*.

	npz file in *output_dir* containing gene coefficients for all spots found in omp step. After each tile is completed, information will be saved to this file. If file does not exist, it will be saved after first tile of OMP step. 

	Default: `omp_spot_coef`

* **big_dapi_image**: *str*.

	npz file in *output_dir* where stitched DAPI image is saved. If it does not exist, it will be saved if `basic_info['dapi_channel']` is not `None`. 

	Default: `dapi_image`

* **big_anchor_image**: *str*.

	npz file in *output_dir* where stitched image of `ref_round`/`ref_channel` is saved. If it does not exist, it will be saved. 

	Default: `anchor_image`

* **pciseq**: *list_str*.

	csv files in *output_dir* where plotting information for pciSeq will be saved. First file is name where *omp* method output will be saved. Second file is name where *ref_spots* method output will be saved. If files don't exist, they will be created when the function *iss/export_to_pciseq* is run. 

	Default: `pciseq_omp, pciseq_anchor`

## basic_info
The *basic_info* section indicates information required throughout the pipeline.

* **is_3d**: *bool*.

	Whether to use the 3d pipeline. 

	Default: `MUST BE SPECIFIED`

* **anchor_channel**: *maybe_int*.

	Channel in anchor round used as reference and to build coordinate system on. Usually channel with most spots. Leave blank if anchor not used. 

	Default: `None`

* **dapi_channel**: *maybe_int*.

	Channel in anchor round that contains DAPI images. Leave blank if no DAPI. 

	Default: `None`

* **ref_round**: *maybe_int*.

	Round to align all imaging rounds to. Will be set to `anchor_round` if `anchor_channel` and `file_names['anchor']` specified. 

	Default: `None`

* **ref_channel**: *maybe_int*.

	Channel in `ref_round` used as reference and to build coordinate system on. Usually channel with most spots. Will be set to `anchor_channel` if `anchor_channel` and `file_names['anchor']` specified. 

	Default: `None`

* **use_channels**: *maybe_list_int*.

	Channels in imaging rounds to use throughout pipeline. Leave blank to use all. 

	Default: `None`

* **use_rounds**: *maybe_list_int*.

	Imaging rounds to use throughout pipeline. Leave blank to use all. 

	Default: `None`

* **use_z**: *maybe_list_int*.

	z planes used to make tile .npy files. Leave blank to use all. If 2 values provided, all z-planes between and including the values given will be used. 

	Default: `None`

* **use_tiles**: *maybe_list_int*.

	Tiles used throughout pipeline. Leave blank to use all. For an experiment where the tiles are arranged in a 4 x 3 (ny x nx) grid, tile indices are indicated as below: 

	 | 2  | 1  | 0  | 

	 | 5  | 4  | 3  | 

	 | 8  | 7  | 6  | 

	 | 11 | 10 | 9  | 

	Default: `None`

* **use_dyes**: *maybe_list_int*.

	Dyes to use when when assigning spots to genes. Leave blank to use all. 

	Default: `None`

* **dye_names**: *maybe_list_str*.

	Name of dyes used in correct order. So for gene with code `360...`, gene appears with `dye_names[3]` in round 0, `dye_names[6]` in round 1, `dye_names[0]` in round 2 etc. If left blank, then assumes each channel corresponds to a different dye i.e. code 0 in code_book = channel 0. For quad_cam data, this needs to be specified. 

	Default: `None`

* **channel_camera**: *maybe_list_int*.

	`channel_camera[i]` is the wavelength in nm of the camera used for channel `i`. Only need to be provided if `dye_names` provided to help estimate dye intensity in each channel. 

	Default: `None`

* **channel_laser**: *maybe_list_int*.

	`channel_laser[i]` is the wavelengths in nm of the camera/laser used for channel `i`. Only need to be provided if `dye_names` provided to help estimate dye intensity in each channel. 

	Default: `None`

* **tile_pixel_value_shift**: *int*.

	This is added onto every tile (except DAPI) when it is saved and removed from every tile when loaded. Required so we can have negative pixel values when save to .npy as uint16. 

	Default: `15000`

* **ignore_first_z_plane**: *bool*.

	Previously had cases where first z plane in .nd2 file was in wrong place and caused focus stacking to be weird or identify lots of spots on first plane. Hence it is safest to not load first plane and this is done if `ignore_first_z_plane = True`. 

	Default: `True`

## extract
The *extract* section contains parameters which specify how to filter the raw microscope images to produce the .npy files saved to `file_names['tile_dir']`.

* **wait_time**: *int*.

	Time to wait in seconds for raw data to come in before crashing. Assumes first round is already in the `file_names['input_dir']` Want this to be large so can run pipeline while collecting data. 

	Default: `21600`

* **r1**: *maybe_int*.

	Filtering is done with a 2D difference of hanning filter with inner radius `r1` within which it is positive and outer radius `r2` so annulus between `r1` and `r2` is negative. Should be approx radius of spot. Typical = 3. 

	 For `r1 = 3` and `r2 = 6`, a `2048 x 2048 x 50` image took 4.1s. For `2 <= r1 <= 5` and `r2` double this, the time taken seemed to be constant. 

	 Leave blank to auto detect using `r1_auto_microns micron`. 

	Default: `None`

* **r2**: *maybe_int*.

	Filtering is done with a 2D difference of hanning filter with inner radius `r1` within which it is positive and outer radius `r2` so annulus between `r1` and `r2` is negative. Should be approx radius of spot. Typical = 6. Leave blank to set to twice `r1`. 

	Default: `None`

* **r_dapi**: *maybe_int*.

	Filtering for DAPI images is a tophat with r_dapi radius. Should be approx radius of object of interest. Typical = 48. Leave blank to auto detect using `r_dapi_auto_microns`. 

	Default: `None`

* **r1_auto_microns**: *number*.

	If `r1` not specified, will convert to units of pixels from this micron value. 

	Default: `0.5`

* **r_dapi_auto_microns**: *maybe_number*.

	If `r_dapi` not specified. Will convert to units of pixels from this micron value. Typical = 8.0. If both this and `r_dapi` left blank, DAPI image will not be filtered and no .npy file saved. Instead DAPI will be loaded directly from raw data and then stitched. 

	Default: `None`

* **scale**: *maybe_number*.

	Each filtered image is multiplied by scale. This is because the image is saved as uint16 so to gain information from the decimal points, should multiply image so max pixel number is in the 10,000s (less than 65,536). Leave empty to auto-detect using `scale_norm`. 

	Default: `None`

* **scale_norm**: *maybe_int*.

	If `scale` not given, `scale = scale_norm/max(scale_image)`. Where `scale_image` is the `n_channels x n_y x n_x x n_z` image belonging to the central tile (saved as `nb.extract_debug.scale_tile`) of round 0 after filtering and smoothing. 

	 Must be less than `np.iinfo(np.uint16).max - config['basic_info']['tile_pixel_value_shift']` which is typically $65535 - 15000 = 50535$. 

	Default: `35000`

* **scale_anchor**: *maybe_number*.

	Analogous to `scale` but have different normalisation for anchor round/anchor channel as not used in final spot_colors. Leave empty to auto-detect using `scale_norm`. 

	Default: `None`

* **auto_thresh_multiplier**: *number*.

	`nb.extract.auto_thresh[t,r,c]` is default threshold to find spots on tile t, round r, channel c. Value is set to `auto_thresh_multiplier * median(abs(image))` where `image` is the image produced for tile t, round r, channel c in the extract step of the pipeline and saved to `file_names['tile_dir']`. 

	Default: `10`

* **deconvolve**: *bool*.

	For 3D pipeline, whether to perform wiener deconvolution before hanning filtering. 

	Default: `False`

* **psf_detect_radius_xy**: *int*.

	Need to detect spots to determine point spread function (psf) used in the wiener deconvolution. Only relevant if `deconvolve == True`. To detect spot, pixel needs to be above dilation with this radius in xy plane. 

	Default: `2`

* **psf_detect_radius_z**: *int*.

	Need to detect spots to determine point spread function (psf) used in the wiener deconvolution. Only relevant if `deconvolve == True`. To detect spot, pixel needs to be above dilation with this radius in z direction. 

	Default: `2`

* **psf_intensity_thresh**: *maybe_number*.

	Spots contribute to `psf` if they are above this intensity. If not given, will be computed the same as `auto_thresh` i.e. `median(image) + auto_thresh_multiplier*median(abs(image-median(image)))`. Note that for raw data, `median(image)` is not zero hence the difference. 

	Default: `None`

* **psf_isolation_dist**: *number*.

	Spots contribute to `psf` if more than `psf_isolation_dist` from nearest spot. 

	Default: `20`

* **psf_min_spots**: *int*.

	Need this many isolated spots to determine `psf`. 

	Default: `300`

* **psf_shape**: *list_int*.

	Diameter of psf in y, x, z direction (in units of [xy_pixels, xy_pixels, z_pixels]). 

	Default: `181, 181, 19`

* **psf_annulus_width**: *number*.

	`psf` is assumed to be radially symmetric within each z-plane so assume all values within annulus of this size (in xy_pixels) to be the same. 

	Default: `1.4`

* **wiener_constant**: *number*.

	Constant used to compute wiener filter from `psf`. 

	Default: `50000`

* **wiener_pad_shape**: *list_int*.

	When applying the wiener filter, we pad the raw image to median value linearly with this many pixels at end of each dimension. 

	Default: `20, 20, 3`

* **r_smooth**: *maybe_list_int*.

	Radius of averaging filter to do smoothing of filtered image. Provide two numbers to do 2D smoothing and three numbers to do 3D smoothing. Typical *2D*: `2, 2`. Typical *3D*: `1, 1, 2`. Recommended use is in *3D* only as it incorporates information between z-planes which filtering with difference of hanning kernels does not. 

	 Size of `r_smooth` has big influence on time taken for smoothing. For a `2048 x 2048 x 50` image: 

	 * `r_smooth = 1, 1, 2`: 2.8 seconds 

	 * `r_smooth = 2, 2, 2`: 8.5 seconds 

	 Leave empty to do no smoothing. 

	Default: `None`

* **n_clip_warn**: *int*.

	If the number of pixels that are clipped when saving as uint16 is more than `n_clip_warn`, a warning message will occur. 

	Default: `1000`

* **n_clip_error**: *maybe_int*.

	If the number of pixels that are clipped when saving as uint16 is more than `n_clip_error` for `n_clip_error_images_thresh` images, the extract and filter step will be halted. If left blank, n_clip_error will be set to 1% of pixels of a single z-plane. 

	Default: `None`

* **n_clip_error_images_thresh**: *int*.

	If the number of pixels that are clipped when saving as uint16 is more than `n_clip_error` for `n_clip_error_images_thresh` images, the extract and filter step will be halted. 

	Default: `3`

## find_spots
The *find_spots* section contains parameters which specify how to convert the images produced in the extract section to point clouds.

* **radius_xy**: *int*.

	To be detected as a spot, a pixel needs to be above dilation with structuring element which is a square (`np.ones`) of width `2*radius_xy-1` in the xy plane. 

	Default: `2`

* **radius_z**: *int*.

	To be detected as a spot, a pixel needs to be above dilation with structuring element which is cuboid (`np.ones`) with width `2*radius_z-1` in z direction. Must be more than 1 to be 3D. 

	Default: `2`

* **max_spots_2d**: *int*.

	If number of spots detected on particular z-plane of an imaging round is greater than this, then will only select the `max_spots_2d` most intense spots on that z-plane. I.e. PCR works better if trying to fit fewer more intense spots. This only applies to imaging rounds and not ref_round/ref_channel as need lots of spots then. In 2D, allow more spots as only 1 z-plane 

	Default: `1500`

* **max_spots_3d**: *int*.

	Same as `max_spots_2d` for the 3D pipeline. In 3D, need to allow less spots on a z-plane as have many z-planes. 

	Default: `500`

* **isolation_radius_inner**: *number*.

	To determine if spots are isolated, filter image with annulus between `isolation_radius_inner` and `isolation_radius`. `isolation_radius_inner` should be approx the radius where intensity of spot crosses from positive to negative. It is in units of xy-pixels. This filtering will only be applied to spots detected in the ref_round/ref_channel. 

	Default: `4`

* **isolation_radius_xy**: *number*.

	Outer radius of annulus filtering kernel in xy direction in units of xy-pixels. 

	Default: `14`

* **isolation_radius_z**: *number*.

	Outer radius of annulus filtering kernel in z direction in units of z-pixels. 

	Default: `1`

* **isolation_thresh**: *maybe_number*.

	Spot is isolated if value of annular filtered image at spot location is below the `isolation_thresh` value. Leave blank to automatically determine value using `auto_isolation_thresh_multiplier`. multiplied by the threshold used to detect the spots i.e. the extract_auto_thresh value. 

	Default: `None`

* **auto_isolation_thresh_multiplier**: *number*.

	If `isolation_thresh` left blank, it will be set to `isolation_thresh = auto_isolation_thresh_multiplier * nb.extract.auto_thresh[:, r, c]`. 

	Default: `-0.2`

* **n_spots_warn_factor**: *number*.

	Used in *iss/find_spots/base/check_n_spots* 

	 A warning will be raised if for any tile, round, channel the number of spots detected is less than: 

	 `n_spots_warn = n_spots_warn_factor * max_spots * nb.basic_info.nz` 

	 where `max_spots` is `max_spots_2d` if *2D* and `max_spots_3d` if *3D*. 

	Default: `0.1`

* **n_spots_error_factor**: *number*.

	Used in *iss/find_spots/base/check_n_spots*. An error is raised if any of the following are satisfied: 

	 * For any given channel, the number of spots found was less than `n_spots_warn` for at least the fraction `n_spots_error_factor` of tiles/rounds. 

	 * For any given tile, the number of spots found was less than `n_spots_warn` for at least the fraction `n_spots_error_factor` of rounds/channels. 

	 * For any given round, the number of spots found was less than `n_spots_warn` for at least the fraction `n_spots_error_factor` of tiles/channels. 

	Default: `0.5`

## stitch
The *stitch* section contains parameters which specify how the overlaps between neighbouring tiles are found. Note that references to south in this section should really be north and west should be east.

* **expected_overlap**: *number*.

	Expected fractional overlap between tiles. Used to get initial shift search if not provided. 

	Default: `0.1`

* **auto_n_shifts**: *list_int*.

	If `shift_south_min/max` and/or `shift_west_min/max` not given, the initial shift search will have `auto_n_shifts` either side of the expected shift given the `expected_overlap` with step given by `shift_step`. First value gives $n_{shifts}$ in direction of overlap (y for south, x for west). Second value gives $n_{shifts}$ in other direction (x for south, y for west). Third value gives $n_{shifts}$ in z. 

	Default: `20, 20, 1`

* **shift_south_min**: *maybe_list_int*.

	Can manually specify initial shifts. Exhaustive search will include all shifts between min and max with step given by `shift_step`. Each entry should be a list of 3 values: [y, x, z]. Typical: `-1900, -100, -2` 

	Default: `None`

* **shift_south_max**: *maybe_list_int*.

	Can manually specify initial shifts. Exhaustive search will include all shifts between min and max with step given by `shift_step`. Each entry should be a list of 3 values: [y, x, z]. Typical: `-1700, 100, 2` 

	Default: `None`

* **shift_west_min**: *maybe_list_int*.

	Can manually specify initial shifts. Exhaustive search will include all shifts between min and max with step given by `shift_step`. Each entry should be a list of 3 values: [y, x, z]. Typical: `-100, -1900, -2` 

	Default: `None`

* **shift_west_max**: *maybe_list_int*.

	Can manually specify initial shifts. Shift range will run between min to max with step given by `shift_step`. Each entry should be a list of 3 values: [y, x, z]. Typical: `100, -1700, 2` 

	Default: `None`

* **shift_step**: *list_int*.

	Step size to use in y, x, z when finding shift between tiles. 

	Default: `5, 5, 3`

* **shift_widen**: *list_int*.

	If shift in initial search range has score which does not exceed `shift_score_thresh`, then range will be extrapolated with same step by `shift_widen` values in y, x, z direction. 

	Default: `10, 10, 1`

* **shift_max_range**: *list_int*.

	The range of shifts searched over will continue to be increased according to `shift_widen` until the shift range in the y, x, z direction reaches `shift_max_range`. If a good shift is still not found, a warning will be printed. 

	Default: `300, 300, 10`

* **neighb_dist_thresh**: *number*.

	Basically the distance in yx pixels below which neighbours are a good match. 

	Default: `2`

* **shift_score_thresh**: *maybe_number*.

	A shift between tiles must have a number of close neighbours exceeding this. If not given, it will be worked using the `shift_score_thresh` parameters below using the function *iss/stitch/shift/get_score_thresh*. 

	Default: `None`

* **shift_score_thresh_multiplier**: *number*.

	`shift_score_thresh` is set to `shift_score_thresh_multiplier` multiplied by the mean of scores of shifts a distance between `shift_score_thresh_min_dist` and `shift_score_thresh_max_dist` from the best shift. 

	Default: `2`

* **shift_score_thresh_min_dist**: *number*.

	`shift_score_thresh` is set to `shift_score_thresh_multiplier` multiplied by the mean of scores of shifts a distance between `shift_score_thresh_min_dist` and `shift_score_thresh_max_dist` from the best shift. 

	Default: `11`

* **shift_score_thresh_max_dist**: *number*.

	`shift_score_thresh` is set to `shift_score_thresh_multiplier` multiplied by the mean of scores of shifts a distance between `shift_score_thresh_min_dist` and `shift_score_thresh_max_dist` from the best shift. 

	Default: `20`

* **nz_collapse**: *int*.

	3D data is converted into `np.ceil(nz / nz_collapse)` 2D slices for exhaustive shift search to quicken it up. I.e. this is the maximum number of z-planes to be collapsed to a 2D slice when searching for the best shift. 

	Default: `30`

* **save_image_zero_thresh**: *int*.

	When saving stitched images, all pixels with absolute value less than or equal to `save_image_zero_thresh` will be set to 0. This helps reduce size of the .npz files and does not lose any important information. 

	Default: `20`

## register_initial
The *register_initial* section contains parameters which specify how the shifts from the ref_round/ref_channel to each imaging round/channel are found. These are then used as the starting point for determining the affine transforms in the *register* section.

* **shift_channel**: *maybe_int*.

	Channel to use to find shifts between rounds to use as starting point for PCR. Leave blank to set to `basic_info['ref_channel']`. 

	Default: `None`

* **shift_min**: *list_int*.

	Exhaustive search range will include all shifts between min and max with step given by `shift_step`. Each entry should be a list of 3 values: [y, x, z]. Typical: [-100, -100, -1] 

	Default: `-100, -100, -3`

* **shift_max**: *list_int*.

	Exhaustive search range will include all shifts between min and max with step given by `shift_step`. Each entry should be a list of 3 values: [y, x, z]. Typical: [100, 100, 1] 

	Default: `100, 100, 3`

* **shift_step**: *list_int*.

	Step size to use in y, x, z when performing the exhaustive search to find the shift between tiles. 

	Default: `5, 5, 3`

* **shift_widen**: *list_int*.

	If shift in initial search range has score which does not exceed `shift_score_thresh`, then the range will be extrapolated with same step by shift_widen values in y, x, z direction. 

	Default: `10, 10, 1`

* **shift_max_range**: *list_int*.

	The range of shifts searched over will continue to be increased according to `shift_widen` until the shift range in the y, x, z direction reaches `shift_max_range`. If a good shift is still not found, a warning will be printed. 

	Default: `500, 500, 10`

* **neighb_dist_thresh**: *number*.

	Basically the distance in yx pixels below which neighbours are a good match. 

	Default: `2`

* **shift_score_thresh**: *maybe_number*.

	A shift between tiles must have a number of close neighbours exceeding this. If not given, it will be worked using the `shift_score_thresh` parameters below using the function *iss/stitch/shift/get_score_thresh*. 

	Default: `None`

* **shift_score_thresh_multiplier**: *number*.

	`shift_score_thresh` is set to `shift_score_thresh_multiplier` multiplied by the mean of scores of shifts a distance between `shift_score_thresh_min_dist` and `shift_score_thresh_max_dist` from the best shift. 

	Default: `1.5`

* **shift_score_thresh_min_dist**: *number*.

	`shift_score_thresh` is set to `shift_score_thresh_multiplier` multiplied by the mean of scores of shifts a distance between `shift_score_thresh_min_dist` and `shift_score_thresh_max_dist` from the best shift. 

	Default: `11`

* **shift_score_thresh_max_dist**: *number*.

	`shift_score_thresh` is set to `shift_score_thresh_multiplier` multiplied by the mean of scores of shifts a distance between `shift_score_thresh_min_dist` and `shift_score_thresh_max_dist` from the best shift. 

	Default: `20`

* **nz_collapse**: *int*.

	3D data is converted into `np.ceil(nz / nz_collapse)` 2D slices for exhaustive shift search to quicken it up. I.e. this is the maximum number of z-planes to be collapsed to a 2D slice when searching for the best shift. 

	Default: `30`

## register
The *register* section contains parameters which specify how the affine transforms from the ref_round/ref_channel to each imaging round/channel are found from the shifts found in the *register_initial* section.

* **n_iter**: *int*.

	Maximum number iterations to run point cloud registration, PCR 

	Default: `100`

* **neighb_dist_thresh_2d**: *number*.

	Basically the distance in yx pixels below which neighbours are a good match. PCR updates transforms by minimising distances between neighbours which are closer than this. 

	Default: `3`

* **neighb_dist_thresh_3d**: *number*.

	The same as `neighb_dist_thresh_2d` but in 3D, we use a larger distance because the size of a z-pixel is greater than a xy pixel. 

	Default: `5`

* **matches_thresh_fract**: *number*.

	If PCR produces transforms with fewer neighbours (pairs with distance between them less than `neighb_dist_thresh`) than `matches_thresh = np.clip(matches_thresh_fract * n_spots, matches_thresh_min, matches_thresh_max)`, the transform will be re-evaluated with regularization so it is near the average transform. 

	Default: `0.25`

* **matches_thresh_min**: *int*.

	If PCR produces transforms with fewer neighbours (pairs with distance between them less than `neighb_dist_thresh`) than `matches_thresh = np.clip(matches_thresh_fract * n_spots, matches_thresh_min, matches_thresh_max)`, the transform will be re-evaluated with regularization so it is near the average transform. 

	Default: `25`

* **matches_thresh_max**: *int*.

	If PCR produces transforms with fewer neighbours (pairs with distance between them less than `neighb_dist_thresh`) than `matches_thresh = np.clip(matches_thresh_fract * n_spots, matches_thresh_min, matches_thresh_max)`, the transform will be re-evaluated with regularization so it is near the average transform. 

	Default: `300`

* **scale_dev_thresh**: *list_number*.

	If a transform has a chromatic aberration scaling that has an absolute deviation of more than `scale_dev_thresh[i]` from the median for that colour channel in dimension `i`, it will be re-evaluated with regularization. There is a threshold for the y, x, z scaling. 

	Default: `0.01, 0.01, 0.1`

* **shift_dev_thresh**: *list_number*.

	If a transform has a `shift[i]` that has an absolute deviation of more than `shift_dev_thresh[i]` from the median for that tile and round in any dimension `i`, it will be re-evaluated with regularization. There is a threshold for the y, x, z shift. `shift_dev_thresh[2]` is in z pixels. 

	Default: `15, 15, 5`

* **regularize_constant_scale**: *number*.

	Constant used for scaling and rotation when doing regularized least squares. 

	Default: `30000`

* **regularize_constant_shift**: *number*.

	Constant used for shift when doing regularized least squares. We want to allow for a rotation deviation of around 0.0003 and a shift deviation of around 9 hence the difference in constants. 

	Default: `9`

## call_spots
The *call_spots* section contains parameters which determine how the `bleed_matrix` and `gene_efficiency` are computed, as well as how a gene is assigned to each spot found on the ref_round/ref_channel.

* **bleed_matrix_method**: *str*.

	`bleed_matrix_method` can only be `single` or `separate`. `single`: a single bleed matrix is produced for all rounds. `separate`: a different bleed matrix is made for each round. 

	Default: `single`

* **color_norm_intensities**: *list_number*.

	Parameter used to get color normalisation factor. `color_norm_intensities` should be ascending and `color_norm_probs` should be descending and they should be the same size. The probability of normalised spot color being greater than `color_norm_intensities[i]` must be less than `color_norm_probs[i]` for all `i`. 

	Default: `0.5, 1, 5`

* **color_norm_probs**: *list_number*.

	Parameter used to get color normalisation factor. `color_norm_intensities` should be ascending and `color_norm_probs` should be descending and they should be the same size. The probability of normalised spot color being greater than `color_norm_intensities[i]` must be less than `color_norm_probs[i]` for all `i`. 

	Default: `0.01, 5e-4, 1e-5`

* **bleed_matrix_score_thresh**: *number*.

	In `scaled_k_means` part of `bleed_matrix` calculation, a mean vector for each dye is computed from all spots with a dot product to that mean greater than this. 

	Default: `0`

* **background_weight_shift**: *maybe_number*.

	Shift to apply to weighting of each background vector to limit boost of weak spots. The weighting of round r for the fitting of the background vector for channel c is `1 / (spot_color[r, c] + background_weight_shift)` so `background_weight_shift` ensures this does not go to infinity for small `spot_color[r, c]`. Typical `spot_color[r, c]` is 1 for intense spot so `background_weight_shift` is small fraction of this. Leave blank to determine using `norm_shift_auto_param`. 

	Default: `None`

* **dp_norm_shift**: *maybe_number*.

	When calculating the `dot_product_score`, this is the small shift to apply when normalising `spot_colors` to ensure don't divide by zero. Value is for a single round and is multiplied by `sqrt(n_rounds_used)` when computing `dot_product_score`. Expected norm of a spot_color for a single round is 1 so `dp_norm_shift` is a small fraction of this. Leave blank to determine using `norm_shift_auto_param`. 

	Default: `None`

* **norm_shift_auto_param**: *number*.

	Parameter to automatically determine `dp_norm_shift`, `background_weight_shift` and `gene_efficiency_intensity_thresh`. They are all set using `norm_shift = norm_shift_auto_param * median(get_spot_intensity(abs(norm_shift_image)))`. Where `norm_shift_image` is the middle z-plane (`nb.call_spots.norm_shift_z`) of the central tile (`nb.call_spots.norm_shift_tile`). Work out from this because `dp_norm_shift` is basically a small fraction of the expected L2 norm of `spot_color` in a single round and this is basically what spot_intensity gives. It is also clamped between the min and max values. 

	Default: `0.1`

* **norm_shift_min**: *number*.

	Minimum possible value of `dp_norm_shift` and `background_weight_shift`. 

	Default: `0.001`

* **norm_shift_max**: *number*.

	Maximum possible value of `dp_norm_shift` and `background_weight_shift`. 

	Default: `0.5`

* **norm_shift_precision**: *number*.

	`dp_norm_shift` and `background_weight_shift` will be rounded to nearest `norm_shift_precision`. 

	Default: `0.01`

* **norm_shift_to_intensity_scale**: *number*.

	`gene_efficiency_intensity_thresh` will be set to `norm_shift / norm_shift_to_intensity_scale` where `norm_shift` is determined using `norm_shift_auto_param` if not specified. 

	Default: `10`

* **gene_efficiency_min_spots**: *int*.

	If number of spots assigned to a gene less than or equal to this, `gene_efficiency[g]=1` for all rounds. 

	Default: `25`

* **gene_efficiency_n_iter**: *int*.

	`gene_efficiency` is computed from spots which pass a quality thresholding based on the bled_codes computed with the gene_efficiency of the previous iteration. This process will continue until the `gene_effiency` converges or `gene_efficiency_n_iter` iterations are reached. 0 means `gene_efficiency` will not be used. 

	Default: `10`

* **gene_efficiency_score_thresh**: *number*.

	Spots used to compute `gene_efficiency` must have `dot_product_score` greater than `gene_efficiency_score_thresh`, difference to second best score greater than `gene_efficiency_score_diff_thresh` and intensity greater than `gene_efficiency_intensity_thresh`. 

	Default: `0.6`

* **gene_efficiency_score_diff_thresh**: *number*.

	Spots used to compute `gene_efficiency` must have `dot_product_score` greater than `gene_efficiency_score_thresh`, difference to second best score greater than `gene_efficiency_score_diff_thresh` and intensity greater than `gene_efficiency_intensity_thresh`. 

	Default: `0.2`

* **gene_efficiency_intensity_thresh**: *maybe_number*.

	Spots used to compute `gene_efficiency` must have `dot_product_score` greater than `gene_efficiency_score_thresh`, difference to second best score greater than `gene_efficiency_score_diff_thresh` and intensity greater than `gene_efficiency_intensity_thresh`. Leave blank to determine from `norm_shift_auto_param`. 

	Default: `None`

## omp
The *omp* section contains parameters which are use to carry out orthogonal matching pursuit (omp) on every pixel, as well as how to convert the results of this to spot locations.

* **use_z**: *maybe_list_int*.

	Can specify z-planes to find spots on If 2 values provided, all z-planes between and including the values given will be used. 

	Default: `None`

* **weight_coef_fit**: *bool*.

	If `False`, gene coefficients are found through omp with normal least squares fitting. If `True`, gene coefficients are found through omp with weighted least squares fitting with rounds/channels which already containing genes contributing less. 

	Default: `False`

* **initial_intensity_thresh**: *maybe_number*.

	To save time in `call_spots_omp`, coefficients only found for pixels with intensity of absolute `spot_colors` greater than `initial_intensity_thresh`. Leave blank to set to determine using `initial_intensity_thresh_auto_param` It is also clamped between the `initial_intensity_thresh_min` and `initial_intensity_thresh_max`. 

	Default: `None`

* **initial_intensity_thresh_auto_param**: *number*.

	If `initial_intensity_thresh`, it will be set to `initial_intensity_thresh_auto_param  * nb.call_spots.median_abs_intensity`. 

	Default: `0.5`

* **initial_intensity_thresh_min**: *number*.

	Min allowed value of `initial_intensity_thresh`. 

	Default: `0.001`

* **initial_intensity_thresh_max**: *number*.

	Max allowed value of `initial_intensity_thresh`. 

	Default: `0.5`

* **initial_intensity_precision**: *number*.

	`initial_intensity_thresh` will be rounded to nearest `initial_intensity_precision` if not given. 

	Default: `0.001`

* **max_genes**: *int*.

	The maximum number of genes that can be assigned to each pixel i.e. number of iterations of omp. 

	Default: `30`

* **dp_thresh**: *number*.

	Pixels only have coefficient found for a gene if that gene has absolute `dot_product_score` greater than this i.e. this is the stopping criterion for the OMP. 

	Default: `0.225`

* **alpha**: *number*.

	Parameter for fitting_standard_deviation. By how much to increase variance as genes added. TODO: make this comment better 

	Default: `120`

* **beta**: *number*.

	Parameter for fitting_standard_deviation. The variance with no genes added (`coef=0`) is `beta**2`. 

	Default: `1`

* **initial_pos_neighbour_thresh**: *maybe_int*.

	Only save spots with number of positive coefficient neighbours greater than `initial_pos_neighbour_thresh`. Leave blank to determine using `initial_pos_neighbour_thresh_param`. It is also clipped between `initial_pos_neighbour_thresh_min` and `initial_pos_neighbour_thresh_max`. 

	Default: `None`

* **initial_pos_neighbour_thresh_param**: *number*.

	If `initial_pos_neighbour_thresh` not given, it is set to `initial_pos_neighbour_thresh_param` multiplied by number of positive values in nb.omp.spot_shape i.e. with `initial_pos_neighbour_thresh_param = 0.1`, it is set to 10% of the max value. 

	Default: `0.1`

* **initial_pos_neighbour_thresh_min**: *int*.

	Min allowed value of `initial_pos_neighbour_thresh`. 

	Default: `4`

* **initial_pos_neighbour_thresh_max**: *int*.

	Max allowed value of `initial_pos_neighbour_thresh`. 

	Default: `40`

* **radius_xy**: *int*.

	To detect spot in coefficient image of each gene, pixel needs to be above dilation with structuring element which is a square (`np.ones`) of width `2*radius_xy-1` in the xy plane. 

	Default: `3`

* **radius_z**: *int*.

	To detect spot in coefficient image of each gene, pixel needs to be above dilation with structuring element which is cuboid (`np.ones`) with width `2*radius_z-1` in z direction. Must be more than 1 to be 3D. 

	Default: `2`

* **shape_max_size**: *list_int*.

	spot_shape specifies the neighbourhood about each spot in which we count coefficients which contribute to score. It is either given through `file_names['omp_spot_shape']` or computed using the below parameters with shape prefix. Maximum Y, X, Z size of spot_shape. Will be cropped if there are zeros at the extremities. 

	Default: `27, 27, 9`

* **shape_pos_neighbour_thresh**: *int*.

	For spot to be used to find `spot_shape`, it must have this many pixels around it on the same z-plane that have a positive coefficient. If 3D, also, require 1 positive pixel on each neighbouring plane (i.e. 2 is added to this value). 

	Default: `9`

* **shape_isolation_dist**: *number*.

	Spots are isolated if nearest neighbour (across all genes) is further away than this. Only isolated spots are used to find `spot_shape`. 

	Default: `10`

* **shape_sign_thresh**: *number*.

	If the mean absolute coefficient sign is less than this in a region near a spot, we set the expected coefficient in `spot_shape` to be 0. Max mean absolute coefficient sign is 1 so must be less than this. 

	Default: `0.15`

## thresholds
The *thresholds* section contains the thresholds used to determine which spots pass a quality thresholding process such that we consider their gene assignments legitimate.

* **intensity**: *maybe_number*.

	Final accepted reference and OMP spots both require `intensity > thresholds[intensity]`. If not given, will be set to same value as `call_spots['gene_efficiency_intensity_thresh']`. intensity for a really intense spot is about 1 so intensity_thresh should be less than this. 

	Default: `None`

* **score_ref**: *number*.

	Final accepted spots are those which pass quality_threshold which is `nb.ref_spots.score > thresholds[score_ref]` and `nb.ref_spots.intensity > intensity_thresh`. quality_threshold requires score computed with *iss/call_spots/dot_prodduct/dot_product_score* to exceed this. Max score is 1 so must be below this. 

	Default: `0.25`

* **score_omp**: *number*.

	Final accepted OMP spots are those which pass quality_threshold which is: `score > thresholds[score_omp]` and `intensity > thresholds[intensity]`. `score` is given by: `score = (score_omp_multiplier * n_neighbours_pos + n_neighbours_neg) /   (score_omp_multiplier * n_neighbours_pos_max + n_neighbours_neg_max)` Max score is 1 so `score_thresh` should be less than this. 0.15 if more concerned for missed spots than false positives. 

	Default: `0.263`

* **score_omp_multiplier**: *number*.

	Final accepted OMP spots are those which pass quality_threshold which is: `score > thresholds[score_omp]` and `intensity > thresholds[intensity]`. `score` is given by: `score = (score_omp_multiplier * n_neighbours_pos + n_neighbours_neg) /   (score_omp_multiplier * n_neighbours_pos_max + n_neighbours_neg_max)` 0.45 if more concerned for missed spots than false positives. 

	Default: `0.95`

