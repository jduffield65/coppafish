import os
import time
import numpy as np
import warnings
import pytest
from typing import Any

from coppafish import Notebook, Viewer
from coppafish.robominnie import RoboMinnie
from coppafish.plot.register.diagnostics import RegistrationViewer


def get_robominnie_scores(rm: RoboMinnie) -> None:
    print(rm.compare_spots('ref'))
    overall_score = rm.overall_score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall reference spots score is < 75%'))

    print(rm.compare_spots('omp'))
    overall_score = rm.overall_score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))
    del rm


@pytest.mark.slow
def test_integration_001() -> Notebook:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, sequencing rounds, one tile.

    Returns:
        Notebook: complete coppafish Notebook.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(include_presequence=False, include_dapi=False)
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    robominnie.add_spots(n_spots=15_000)
    robominnie.save_raw_images(output_dir=output_dir)
    nb = robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie
    return nb


@pytest.mark.slow
def test_deterministic(iterations: int = 2) -> None:
    """
    Test that the coppafish output is always the same when run over and over, i.e. deterministic by comparing the 
    Notebooks.

    Args:
        iterations (int, optional): number of times to run the coppafish pipeline. Default: 2.
    """
    assert iterations > 1
    notebooks = []
    for i in range(iterations):
        notebooks.append(test_bg_subtraction())
        if i == 0:
            continue
        assert np.allclose(notebooks[i - 1].filter.bg_scale, notebooks[i].filter.bg_scale), \
            f"Notebooks omp.gene_no were not equal!"
        assert np.allclose(notebooks[i - 1].omp.gene_no, notebooks[i].omp.gene_no), \
            f"Notebooks omp.gene_no were not equal!"
        assert np.allclose(notebooks[i - 1].omp.local_yxz, notebooks[i].omp.local_yxz), \
            f"Notebooks omp.local_yxz were not equal!"


@pytest.mark.slow
def test_integration_002() -> None:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, DAPI image, presequence round, sequencing rounds, one tile.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie()
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    # Add spots to DAPI image as larger spots
    robominnie.add_spots(n_spots=15_000, spot_size_pixels_dapi=np.array([9, 9, 9]), include_dapi=True, 
                         spot_amplitude_dapi=0.05)
    # robominnie.Generate_Random_Noise(noise_mean_amplitude=0, noise_std=0.0004, noise_type='normal')
    robominnie.save_raw_images(output_dir=output_dir)
    robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie


@pytest.mark.slow
def test_integration_003(
    include_stitch: bool = True, include_omp: bool = True, run_tile_by_tile: bool = True, 
    ) -> Notebook:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor, DAPI, presequencing round and sequencing rounds, `2` connected tiles, aligned along the x axis.

    Args:
        include_stitch (bool, optional): run stitch. Default: true.
        include_omp (bool, optional): run OMP. Default: true.
        run_tile_by_tile (bool, optional): run each tile separately then combine notebooks on tile independent 
            pipeline. Default: true.

    Returns:
        Notebook: final notebook.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(n_tiles_x=2)
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    # Add spots to DAPI image as larger spots
    robominnie.add_spots(n_spots=25_000, include_dapi=True, spot_size_pixels_dapi=np.array([9, 9, 9]), 
                         spot_amplitude_dapi=0.05)
    robominnie.save_raw_images(output_dir=output_dir)
    nb = robominnie.run_coppafish(
        include_stitch=include_stitch, 
        include_omp=include_omp, 
        run_tile_by_tile=run_tile_by_tile
    )
    if not include_omp or not include_stitch:
        return nb
    get_robominnie_scores(robominnie)
    del robominnie
    return nb


@pytest.mark.slow
def test_integration_004() -> None:
    """
    Summary of input data: random spots and pink noise.
    
    Includes anchor round, DAPI image, presequence round, sequencing rounds, one tile. No DAPI channel registration.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie()
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    # Add spots to DAPI image as larger spots
    robominnie.add_spots(n_spots=15_000, spot_size_pixels_dapi=np.array([9, 9, 9]), include_dapi=True, 
                         spot_amplitude_dapi=0.05)
    # robominnie.Generate_Random_Noise(noise_mean_amplitude=0, noise_std=0.0004, noise_type='normal')
    robominnie.save_raw_images(output_dir=output_dir, register_with_dapi=False)
    robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie


@pytest.mark.slow
def test_bg_subtraction() -> None:
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    rng = np.random.RandomState(0)

    robominnie = RoboMinnie(brightness_scale_factor=2 * (0.1 + rng.rand(1, 9, 8)))
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    robominnie.add_spots(n_spots=15_000, 
                         gene_efficiency=0.5 * (rng.rand(20, 8) + 1), 
                         background_offset=1e-7*rng.rand(15_000, 7), 
                         include_dapi=True, 
                         spot_size_pixels_dapi=np.asarray([9, 9, 9]),
                         spot_amplitude_dapi=0.05)
    robominnie.save_raw_images(output_dir=output_dir, register_with_dapi=False)
    nb = robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie
    return nb


@pytest.mark.slow
def test_tile_by_tile_equality() -> None:
    """
    Test for coppafish notebook equality when running the pipeline tile by tile then merging versus all tiles at once 
    (old approach) using ``test_integration_003``.
    """
    def _approximately_equal(a: Any, b: Any) -> bool:
        if a is None and b is None:
            equal = True
        elif a is None or b is None:
            equal = False
        elif isinstance(a, (
                float, np.float16, np.float32, np.float64, int, np.int8, np.int16, np.int32, np.int64, np.uint16, 
                np.uint32, np.uint64, 
        )):
            equal = np.isclose(a, b, equal_nan=True)
        elif isinstance(a, (np.ndarray)):
            if isinstance(a.dtype, (bool, str, np.str_)):
                equal = (a == b).all()
            else:
                equal = np.allclose(a, b, equal_nan=True)
        elif isinstance(a, (str, bool, list)):
            equal = a == b
        else:
            ValueError(f"Failed to compare variables of types {type(a)=} and {type(b)=}")
        if not equal:
            print(f"{a=}\n{b=}")
            print(f"{type(a)=}\n{type(b)=}")
            print(f"{a.dtype=}\n{b.dtype=}")
            print(f"{a.shape=}\n{b.shape=}")
        return equal
    
    start_time_tile_by_tile = time.time()
    nb_1 = test_integration_003(include_omp=True, run_tile_by_tile=True)
    end_time_tile_by_tile = time.time()
    start_time = time.time()
    nb_0 = test_integration_003(include_omp=True, run_tile_by_tile=False)
    end_time = time.time()
    print(f"Pipeline time: {round(end_time - start_time, 1)}s")
    print(f"Pipeline tile by tile time: {round(end_time_tile_by_tile - start_time_tile_by_tile, 1)}s")
    assert nb_0.has_page("file_names") == nb_1.has_page("file_names")
    assert nb_0.has_page("basic_info") == nb_1.has_page("basic_info")
    assert nb_0.has_page("scale") == nb_1.has_page("scale")
    assert nb_0.has_page("extract") == nb_1.has_page("extract")
    assert nb_0.has_page("extract_debug") == nb_1.has_page("extract_debug")
    assert nb_0.has_page("filter") == nb_1.has_page("filter")
    assert nb_0.has_page("filter_debug") == nb_1.has_page("filter_debug")
    assert nb_0.has_page("find_spots") == nb_1.has_page("find_spots")
    assert nb_0.has_page("register") == nb_1.has_page("register")
    assert nb_0.has_page("register_debug") == nb_1.has_page("register_debug")
    assert nb_0.has_page("stitch") == nb_1.has_page("stitch")
    assert nb_0.has_page("ref_spots") == nb_1.has_page("ref_spots")
    assert nb_0.has_page("call_spots") == nb_1.has_page("call_spots")
    assert nb_0.has_page("omp") == nb_1.has_page("omp")
    assert nb_0.has_page("thresholds") == nb_1.has_page("thresholds")
    
    assert nb_0.has_page("basic_info")
    assert _approximately_equal(nb_0.basic_info.anchor_channel, nb_1.basic_info.anchor_channel)
    assert _approximately_equal(nb_0.basic_info.anchor_round, nb_1.basic_info.anchor_round)
    assert _approximately_equal(nb_0.basic_info.channel_camera, nb_1.basic_info.channel_camera)
    assert _approximately_equal(nb_0.basic_info.channel_laser, nb_1.basic_info.channel_laser)
    assert _approximately_equal(nb_0.basic_info.dapi_channel, nb_1.basic_info.dapi_channel)
    assert _approximately_equal(nb_0.basic_info.dye_names, nb_1.basic_info.dye_names)
    assert _approximately_equal(nb_0.basic_info.is_3d, nb_1.basic_info.is_3d)
    assert _approximately_equal(nb_0.basic_info.n_channels, nb_1.basic_info.n_channels)
    assert _approximately_equal(nb_0.basic_info.n_dyes, nb_1.basic_info.n_dyes)
    assert _approximately_equal(nb_0.basic_info.n_extra_rounds, nb_1.basic_info.n_extra_rounds)
    assert _approximately_equal(nb_0.basic_info.n_rounds, nb_1.basic_info.n_rounds)
    assert _approximately_equal(nb_0.basic_info.n_tiles, nb_1.basic_info.n_tiles)
    assert _approximately_equal(nb_0.basic_info.nz, nb_1.basic_info.nz)
    assert _approximately_equal(nb_0.basic_info.pixel_size_xy, nb_1.basic_info.pixel_size_xy)
    assert _approximately_equal(nb_0.basic_info.pixel_size_z, nb_1.basic_info.pixel_size_z)
    assert _approximately_equal(nb_0.basic_info.pre_seq_round, nb_1.basic_info.pre_seq_round)
    assert _approximately_equal(nb_0.basic_info.tile_centre, nb_1.basic_info.tile_centre)
    assert _approximately_equal(nb_0.basic_info.tile_pixel_value_shift, nb_1.basic_info.tile_pixel_value_shift)
    assert _approximately_equal(nb_0.basic_info.tile_sz, nb_1.basic_info.tile_sz)
    assert _approximately_equal(nb_0.basic_info.tilepos_yx, nb_1.basic_info.tilepos_yx)
    assert _approximately_equal(nb_0.basic_info.tilepos_yx_nd2, nb_1.basic_info.tilepos_yx_nd2)
    assert _approximately_equal(nb_0.basic_info.use_anchor, nb_1.basic_info.use_anchor)
    assert _approximately_equal(nb_0.basic_info.use_channels, nb_1.basic_info.use_channels)
    assert _approximately_equal(nb_0.basic_info.use_dyes, nb_1.basic_info.use_dyes)
    assert _approximately_equal(nb_0.basic_info.use_preseq, nb_1.basic_info.use_preseq)
    assert _approximately_equal(nb_0.basic_info.use_rounds, nb_1.basic_info.use_rounds)
    assert _approximately_equal(nb_0.basic_info.use_tiles, nb_1.basic_info.use_tiles)
    assert _approximately_equal(nb_0.basic_info.use_z, nb_1.basic_info.use_z)

    assert nb_0.has_page("file_names")
    assert _approximately_equal(nb_0.file_names.anchor, nb_1.file_names.anchor)
    assert _approximately_equal(nb_0.file_names.big_anchor_image, nb_1.file_names.big_anchor_image)
    assert _approximately_equal(nb_0.file_names.big_dapi_image, nb_1.file_names.big_dapi_image)
    assert _approximately_equal(nb_0.file_names.code_book, nb_1.file_names.code_book)
    assert _approximately_equal(nb_0.file_names.dye_camera_laser, nb_1.file_names.dye_camera_laser)
    assert _approximately_equal(nb_0.file_names.fluorescent_bead_path, nb_1.file_names.fluorescent_bead_path)
    assert _approximately_equal(nb_0.file_names.initial_bleed_matrix, nb_1.file_names.initial_bleed_matrix)
    assert _approximately_equal(nb_0.file_names.input_dir, nb_1.file_names.input_dir)
    assert _approximately_equal(nb_0.file_names.omp_spot_coef, nb_1.file_names.omp_spot_coef)
    assert _approximately_equal(nb_0.file_names.omp_spot_info, nb_1.file_names.omp_spot_info)
    assert _approximately_equal(nb_0.file_names.omp_spot_shape, nb_1.file_names.omp_spot_shape)
    assert _approximately_equal(nb_0.file_names.output_dir, nb_1.file_names.output_dir)
    assert _approximately_equal(nb_0.file_names.pciseq, nb_1.file_names.pciseq)
    assert _approximately_equal(nb_0.file_names.pre_seq, nb_1.file_names.pre_seq)
    assert _approximately_equal(nb_0.file_names.psf, nb_1.file_names.psf)
    assert _approximately_equal(nb_0.file_names.raw_extension, nb_1.file_names.raw_extension)
    assert _approximately_equal(nb_0.file_names.raw_metadata, nb_1.file_names.raw_metadata)
    assert _approximately_equal(nb_0.file_names.round, nb_1.file_names.round)
    assert _approximately_equal(nb_0.file_names.scale, nb_1.file_names.scale)
    assert _approximately_equal(nb_0.file_names.spot_details_info, nb_1.file_names.spot_details_info)
    assert _approximately_equal(nb_0.file_names.tile, nb_1.file_names.tile)
    assert _approximately_equal(nb_0.file_names.tile_dir, nb_1.file_names.tile_dir)

    assert nb_0.has_page("scale")
    assert _approximately_equal(nb_0.scale.scale, nb_1.scale.scale)
    assert _approximately_equal(nb_0.scale.scale_tile, nb_1.scale.scale_tile)
    assert _approximately_equal(nb_0.scale.scale_z, nb_1.scale.scale_z)
    assert _approximately_equal(nb_0.scale.scale_channel, nb_1.scale.scale_channel)
    assert _approximately_equal(nb_0.scale.scale_anchor, nb_1.scale.scale_anchor)
    assert _approximately_equal(nb_0.scale.scale_anchor_tile, nb_1.scale.scale_anchor_tile)
    assert _approximately_equal(nb_0.scale.scale_anchor_z, nb_1.scale.scale_anchor_z)
    assert _approximately_equal(nb_0.scale.r1, nb_1.scale.r1)
    assert _approximately_equal(nb_0.scale.r2, nb_1.scale.r2)
    assert _approximately_equal(nb_0.scale.r_smooth, nb_1.scale.r_smooth)
    assert _approximately_equal(nb_0.scale.r1, nb_1.scale.r1)
    
    assert nb_0.has_page("extract")
    assert _approximately_equal(nb_0.extract.continuous_dapi, nb_1.extract.continuous_dapi)
    assert _approximately_equal(nb_0.extract.file_type, nb_1.extract.file_type)
    assert nb_0.extract.revision_hash == nb_1.extract.revision_hash
    assert nb_0.extract.software_version == nb_1.extract.software_version
    
    assert nb_0.has_page("extract_debug")
    assert nb_0.extract_debug.pixel_unique_values == nb_1.extract_debug.pixel_unique_values
    assert nb_0.extract_debug.pixel_unique_counts == nb_1.extract_debug.pixel_unique_counts
    
    assert nb_0.has_page("filter")
    assert _approximately_equal(nb_0.filter.auto_thresh, nb_1.filter.auto_thresh)
    assert _approximately_equal(nb_0.filter.hist_counts, nb_1.filter.hist_counts)
    assert _approximately_equal(nb_0.filter.hist_values, nb_1.filter.hist_values)
    
    assert nb_0.has_page("find_spots")
    assert _approximately_equal(nb_0.find_spots.isolated_spots, nb_1.find_spots.isolated_spots)
    assert _approximately_equal(nb_0.find_spots.isolation_thresh, nb_1.find_spots.isolation_thresh)
    assert _approximately_equal(nb_0.find_spots.spot_no, nb_1.find_spots.spot_no)
    assert _approximately_equal(nb_0.find_spots.spot_yxz, nb_1.find_spots.spot_yxz)

    assert nb_0.has_page("register")
    # bg_scale is calculated properly at the register section
    assert _approximately_equal(nb_0.filter.bg_scale, nb_1.filter.bg_scale)
    assert _approximately_equal(nb_0.register.channel_transform, nb_1.register.channel_transform)
    assert _approximately_equal(nb_0.register.initial_transform, nb_1.register.initial_transform)
    assert _approximately_equal(nb_0.register.round_transform, nb_1.register.round_transform)
    assert _approximately_equal(nb_0.register.transform, nb_1.register.transform)
    
    assert nb_0.has_page("register_debug")
    assert _approximately_equal(nb_0.register_debug.channel_transform, nb_1.register_debug.channel_transform)
    assert _approximately_equal(nb_0.register_debug.converged, nb_1.register_debug.converged)
    assert _approximately_equal(nb_0.register_debug.mse, nb_1.register_debug.mse)
    assert _approximately_equal(nb_0.register_debug.n_matches, nb_1.register_debug.n_matches)
    assert _approximately_equal(nb_0.register_debug.position, nb_1.register_debug.position)
    assert _approximately_equal(nb_0.register_debug.round_shift, nb_1.register_debug.round_shift)
    assert _approximately_equal(nb_0.register_debug.round_shift_corr, nb_1.register_debug.round_shift_corr)
    assert _approximately_equal(nb_0.register_debug.round_transform_raw, nb_1.register_debug.round_transform_raw)
    
    assert nb_0.has_page("stitch")
    assert _approximately_equal(nb_0.stitch.east_final_shift_search, nb_1.stitch.east_final_shift_search)
    assert _approximately_equal(nb_0.stitch.east_outlier_score, nb_1.stitch.east_outlier_score)
    assert _approximately_equal(nb_0.stitch.east_outlier_shifts, nb_1.stitch.east_outlier_shifts)
    assert _approximately_equal(nb_0.stitch.east_pairs, nb_1.stitch.east_pairs)
    assert _approximately_equal(nb_0.stitch.east_score, nb_1.stitch.east_score)
    assert _approximately_equal(nb_0.stitch.east_score_thresh, nb_1.stitch.east_score_thresh)
    assert _approximately_equal(nb_0.stitch.east_shifts, nb_1.stitch.east_shifts)
    assert _approximately_equal(nb_0.stitch.east_start_shift_search, nb_1.stitch.east_start_shift_search)
    assert _approximately_equal(nb_0.stitch.north_final_shift_search, nb_1.stitch.north_final_shift_search)
    assert _approximately_equal(nb_0.stitch.north_outlier_score, nb_1.stitch.north_outlier_score)
    assert _approximately_equal(nb_0.stitch.north_outlier_shifts, nb_1.stitch.north_outlier_shifts)
    assert _approximately_equal(nb_0.stitch.north_pairs, nb_1.stitch.north_pairs)
    assert _approximately_equal(nb_0.stitch.north_score, nb_1.stitch.north_score)
    assert _approximately_equal(nb_0.stitch.north_score_thresh, nb_1.stitch.north_score_thresh)
    assert _approximately_equal(nb_0.stitch.north_shifts, nb_1.stitch.north_shifts)
    assert _approximately_equal(nb_0.stitch.north_start_shift_search, nb_1.stitch.north_start_shift_search)
    assert _approximately_equal(nb_0.stitch.tile_origin, nb_1.stitch.tile_origin)
    
    assert nb_0.has_page("ref_spots")
    assert _approximately_equal(nb_0.ref_spots.local_yxz, nb_1.ref_spots.local_yxz)
    assert _approximately_equal(nb_0.ref_spots.isolated, nb_1.ref_spots.isolated)
    assert _approximately_equal(nb_0.ref_spots.tile, nb_1.ref_spots.tile)
    assert _approximately_equal(nb_0.ref_spots.colors, nb_1.ref_spots.colors)
    assert _approximately_equal(nb_0.ref_spots.gene_no, nb_1.ref_spots.gene_no)
    assert _approximately_equal(nb_0.ref_spots.score, nb_1.ref_spots.score)
    assert _approximately_equal(nb_0.ref_spots.score_diff, nb_1.ref_spots.score_diff)
    assert _approximately_equal(nb_0.ref_spots.intensity, nb_1.ref_spots.intensity)
    assert _approximately_equal(nb_0.ref_spots.background_strength, nb_1.ref_spots.background_strength)
    assert _approximately_equal(nb_0.ref_spots.gene_probs, nb_1.ref_spots.gene_probs)
    assert _approximately_equal(nb_0.ref_spots.bg_colours, nb_1.ref_spots.bg_colours)

    assert nb_0.has_page("call_spots")
    assert (nb_0.call_spots.gene_names == nb_1.call_spots.gene_names).all()
    assert _approximately_equal(nb_0.call_spots.gene_codes, nb_1.call_spots.gene_codes)
    assert _approximately_equal(nb_0.call_spots.color_norm_factor, nb_1.call_spots.color_norm_factor)
    assert _approximately_equal(nb_0.call_spots.abs_intensity_percentile, nb_1.call_spots.abs_intensity_percentile)
    assert _approximately_equal(nb_0.call_spots.initial_bleed_matrix, nb_1.call_spots.initial_bleed_matrix)
    assert _approximately_equal(nb_0.call_spots.bleed_matrix, nb_1.call_spots.bleed_matrix)
    assert _approximately_equal(nb_0.call_spots.bled_codes, nb_1.call_spots.bled_codes)
    assert _approximately_equal(nb_0.call_spots.bled_codes_ge, nb_1.call_spots.bled_codes_ge)
    assert _approximately_equal(nb_0.call_spots.gene_efficiency, nb_1.call_spots.gene_efficiency)
    assert _approximately_equal(nb_0.call_spots.use_ge, nb_1.call_spots.use_ge)

    assert nb_0.has_page("omp")
    assert _approximately_equal(nb_0.omp.initial_intensity_thresh, nb_1.omp.initial_intensity_thresh)
    assert _approximately_equal(nb_0.omp.shape_tile, nb_1.omp.shape_tile)
    assert _approximately_equal(nb_0.omp.shape_spot_local_yxz, nb_1.omp.shape_spot_local_yxz)
    assert _approximately_equal(nb_0.omp.shape_spot_gene_no, nb_1.omp.shape_spot_gene_no)
    assert _approximately_equal(nb_0.omp.spot_shape_float, nb_1.omp.spot_shape_float)
    assert _approximately_equal(nb_0.omp.initial_pos_neighbour_thresh, nb_1.omp.initial_pos_neighbour_thresh)
    assert _approximately_equal(nb_0.omp.spot_shape, nb_1.omp.spot_shape)
    assert _approximately_equal(nb_0.omp.local_yxz, nb_1.omp.local_yxz)
    assert _approximately_equal(nb_0.omp.tile, nb_1.omp.tile)
    assert _approximately_equal(nb_0.omp.colors, nb_1.omp.colors)
    assert _approximately_equal(nb_0.omp.gene_no, nb_1.omp.gene_no)
    assert _approximately_equal(nb_0.omp.n_neighbours_pos, nb_1.omp.n_neighbours_pos)
    assert _approximately_equal(nb_0.omp.n_neighbours_neg, nb_1.omp.n_neighbours_neg)
    assert _approximately_equal(nb_0.omp.intensity, nb_1.omp.intensity)
    
    assert nb_0.basic_info.revision_hash == nb_1.basic_info.revision_hash
    assert nb_0.basic_info.software_version == nb_1.basic_info.software_version
    assert nb_0.scale.revision_hash == nb_1.scale.revision_hash
    assert nb_0.scale.software_version == nb_1.scale.software_version
    assert nb_0.filter.revision_hash == nb_1.filter.revision_hash
    assert nb_0.filter.software_version == nb_1.filter.software_version
    assert nb_0.find_spots.revision_hash == nb_1.find_spots.revision_hash
    assert nb_0.find_spots.software_version == nb_1.find_spots.software_version
    assert nb_0.register.revision_hash == nb_1.register.revision_hash
    assert nb_0.register.software_version == nb_1.register.software_version
    assert nb_0.stitch.revision_hash == nb_1.stitch.revision_hash
    assert nb_0.stitch.software_version == nb_1.stitch.software_version
    assert nb_0.ref_spots.revision_hash == nb_1.ref_spots.revision_hash
    assert nb_0.ref_spots.software_version == nb_1.ref_spots.software_version
    assert nb_0.call_spots.revision_hash == nb_1.call_spots.revision_hash
    assert nb_0.call_spots.software_version == nb_1.call_spots.software_version
    assert nb_0.omp.revision_hash == nb_1.omp.revision_hash
    assert nb_0.omp.software_version == nb_1.omp.software_version

    if not nb_0.has_page("thresholds"):
        return
    assert _approximately_equal(nb_0.thresholds.intensity, nb_1.thresholds.intensity)
    assert _approximately_equal(nb_0.thresholds.score_ref, nb_1.thresholds.score_ref)
    assert _approximately_equal(nb_0.thresholds.score_omp, nb_1.thresholds.score_omp)
    assert _approximately_equal(nb_0.thresholds.score_omp_multiplier, nb_1.thresholds.score_omp_multiplier)


@pytest.mark.slow
def test_viewers() -> None:
    """
    Make sure the coppafish plotting is working without crashing.
    
    Notes:
        - Requires a robominnie instance to have successfully run through first.
    """
    notebook_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                 'integration_dir/output_coppafish/notebook.npz')
    gene_colours_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                     'integration_dir/gene_colours.csv')
    notebook = Notebook(notebook_path)
    Viewer(notebook, gene_marker_file=gene_colours_path)
    RegistrationViewer(notebook)


if __name__ == '__main__':
    test_tile_by_tile_equality()
    test_viewers()
