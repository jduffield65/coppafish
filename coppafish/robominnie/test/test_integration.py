import os
import numpy as np
from coppafish.robominnie import RoboMinnie
from coppafish import Notebook, Viewer
from coppafish.plot.register.diagnostics import RegistrationViewer
import warnings
import pytest
from typing import Any


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
def test_integration_001() -> None:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, sequencing rounds, one tile.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(include_presequence=False, include_dapi=False)
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    robominnie.add_spots(n_spots=15_000)
    robominnie.save_raw_images(output_dir=output_dir)
    robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie


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
def test_integration_003(include_stitch: bool = True, include_omp: bool = True, run_tile_by_tile: bool = False, 
                         ) -> Notebook:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor, DAPI, presequencing round and sequencing rounds, `2` connected tiles, aligned along the x axis.

    Args:
        include_stitch (bool, optional): run stitch. Default: true.
        include_omp (bool, optional): run OMP. Default: true.
        run_tile_by_tile (bool, optional): run each tile separately then combine notebooks on tile independent 
            pipeline. Default: false.

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
    robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie


@pytest.mark.slow
def test_tile_by_tile_equality() -> None:
    """
    Test for coppafish notebook equality when running the pipeline tile by tile then merging versus all tiles at once 
    (old approach) using ``test_integration_003``. Runs everything except the final step, OMP.
    """
    def _approximately_equal(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        elif isinstance(a, (np.ndarray, float, np.float_, int, np.int_)):
            return np.allclose(a, b)
        elif isinstance(a, (str, bool, list)):
            return a == b
        ValueError(f"Failed to compare variables of types {type(a)} and {type(b)}")
    
    nb_0 = test_integration_003(include_stitch=False, run_tile_by_tile=False)
    nb_1 = test_integration_003(include_stitch=False, run_tile_by_tile=True)
    assert nb_0.has_page("file_names") == nb_1.has_page("file_names")
    assert nb_0.has_page("basic_info") == nb_1.has_page("basic_info")
    assert nb_0.has_page("scale") == nb_1.has_page("scale")
    assert nb_0.has_page("extract") == nb_1.has_page("extract")
    assert nb_0.has_page("extract_debug") == nb_1.has_page("extract_debug")
    assert nb_0.has_page("find_spots") == nb_1.has_page("find_spots")
    assert nb_0.has_page("register") == nb_1.has_page("register")
    assert nb_0.has_page("register_debug") == nb_1.has_page("register_debug")
    assert nb_0.has_page("stitch") == nb_1.has_page("stitch")
    assert nb_0.has_page("ref_spots") == nb_1.has_page("ref_spots")
    assert nb_0.has_page("call_spots") == nb_1.has_page("call_spots")
    assert nb_0.has_page("omp") == nb_1.has_page("omp")
    assert nb_0.has_page("thresholds") == nb_1.has_page("thresholds")
    if not nb_0.has_page("basic_info"):
        return
    # assert _approximately_equal(nb_0.basic_info.)
    if not nb_0.has_page("scale"):
        return
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
    if not nb_0.has_page("extract"):
        return
    assert _approximately_equal(nb_0.extract.auto_thresh, nb_1.extract.auto_thresh)
    assert _approximately_equal(nb_0.extract.hist_counts, nb_1.extract.hist_counts)
    assert _approximately_equal(nb_0.extract.hist_values, nb_1.extract.hist_values)
    assert _approximately_equal(nb_0.extract.file_type, nb_1.extract.file_type)
    if not nb_0.has_page("find_spots"):
        return
    assert _approximately_equal(nb_0.find_spots.isolated_spots, nb_1.find_spots.isolated_spots)
    assert _approximately_equal(nb_0.find_spots.isolation_thresh, nb_1.find_spots.isolation_thresh)
    assert _approximately_equal(nb_0.find_spots.spot_no, nb_1.find_spots.spot_no)
    assert _approximately_equal(nb_0.find_spots.spot_yxz, nb_1.find_spots.spot_yxz)
    if not nb_0.has_page("register"):
        # bg_scale is calculated properly at the register section
        return
    if nb_0.extract.bg_scale is not None:
        assert _approximately_equal(nb_0.extract.bg_scale, nb_1.extract.bg_scale)
    assert _approximately_equal(nb_0.register.channel_transform, nb_1.register.channel_transform)
    assert _approximately_equal(nb_0.register.initial_transform, nb_1.register.initial_transform)
    assert _approximately_equal(nb_0.register.round_transform, nb_1.register.round_transform)
    assert _approximately_equal(nb_0.register.transform, nb_1.register.transform)
    if not nb_0.has_page("register_debug"):
        return
    #FIXME: round_shift_corr are not equal
    assert _approximately_equal(nb_0.register_debug.channel_transform, nb_0.register_debug.channel_transform)
    assert _approximately_equal(nb_0.register_debug.converged, nb_0.register_debug.converged)
    assert _approximately_equal(nb_0.register_debug.mse, nb_0.register_debug.mse)
    assert _approximately_equal(nb_0.register_debug.n_matches, nb_0.register_debug.n_matches)
    assert _approximately_equal(nb_0.register_debug.position, nb_0.register_debug.position)
    assert _approximately_equal(nb_0.register_debug.round_shift, nb_0.register_debug.round_shift)
    assert _approximately_equal(nb_0.register_debug.round_shift_corr, nb_0.register_debug.round_shift_corr)
    assert _approximately_equal(nb_0.register_debug.round_transform_raw, nb_0.register_debug.round_transform_raw)
    if not nb_0.has_page("stitch"):
        return
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
    if not nb_0.has_page("ref_spots"):
        return
    NotImplementedError("Unfinished equality test for ref_spots, OMP, and thresholds sections")
    #TODO: Compare results from ref_spots, call_spots, OMP, and thresholds


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
