import os
import numpy as np
from coppafish.robominnie import RoboMinnie
from coppafish import Notebook, Viewer
from coppafish.plot.register.diagnostics import RegistrationViewer
import warnings
import pytest


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
def test_integration_003() -> None:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor, DAPI, presequencing round and sequencing rounds, `2` connected tiles, aligned along the x axis.
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
    robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie


@pytest.mark.slow
def test_integration_004():
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
def test_bg_subtraction():
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
def test_viewers():
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
    test_integration_003()
    test_viewers()
