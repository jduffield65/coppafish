import os
import numpy as np
from coppafish.robominnie import RoboMinnie
import warnings
import pytest


@pytest.mark.slow
def test_integration_001() -> None:
    """
    Summary of input data: random spots and random, white noise.

    Includes anchor round, sequencing rounds, one tile.

    Compares ground truth spots to OMP spots and reference spots.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(include_presequence=False, include_dapi=False)
    robominnie.generate_gene_codes(n_genes=15)
    robominnie.generate_pink_noise(noise_amplitude=0.0015, noise_spatial_scale=0.1)
    robominnie.add_spots(n_spots=15_000, bleed_matrix=np.diag(np.ones(7)), spot_size_pixels=np.array([1.5, 1.5, 1.5]))
    # Save the synthetic data in coppafish format as raw .npy files
    # NOTE: We are shortening the pipeline runtime by making the initial intensity threshold strict for OMP
    robominnie.save_raw_images(output_dir=output_dir, overwrite=True, omp_iterations=1, 
                               omp_initial_intensity_thresh_percentile=90)
    robominnie.run_coppafish()

    robominnie.compare_spots('ref', score_threshold=0, intensity_threshold=1)
    overall_score = robominnie.overall_score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))
    assert overall_score > 0.5, 'Integration reference spots score < 50%!'

    robominnie.compare_spots('omp', intensity_threshold=1)
    overall_score = robominnie.overall_score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))


@pytest.mark.slow
def test_integration_002() -> None:
    """
    Summary of input data: random spots and random, white noise.

    Includes anchor round, DAPI image, presequence round, sequencing rounds, one tile.

    Compares ground truth spots to OMP spots and reference spots.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie()
    robominnie.generate_gene_codes(n_genes=15)
    robominnie.generate_pink_noise(noise_amplitude=0.0015, noise_spatial_scale=0.1)
    # Add spots to DAPI image as larger spots
    robominnie.add_spots(n_spots=10_000, bleed_matrix=np.diag(np.ones(7)), spot_size_pixels=np.array([1.5, 1.5, 1.5]), 
                         spot_size_pixels_dapi=np.array([9, 9, 9]), include_dapi=True, spot_amplitude_dapi=0.05)
    # robominnie.Generate_Random_Noise(noise_mean_amplitude=0, noise_std=0.0004, noise_type='normal')
    # NOTE: We are shortening the pipeline runtime by making the initial intensity threshold strict for OMP
    robominnie.save_raw_images(output_dir=output_dir, overwrite=True, omp_iterations=1,
                               omp_initial_intensity_thresh_percentile=90)
    robominnie.run_coppafish()

    robominnie.compare_spots('ref')
    overall_score = robominnie.overall_score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))
    assert overall_score > 0.5, 'Integration reference spots score < 50%!'

    robominnie.compare_spots('omp')
    overall_score = robominnie.overall_score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))


@pytest.mark.slow
def test_integration_003() -> None:
    """
    Summary of input data: random spots and random, white noise.

    Includes anchor, DAPI, presequencing round and sequencing rounds, `2` connected tiles, aligned along the x axis.

    Compares ground truth spots to OMP spots and reference spots.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(include_presequence=False, include_dapi=False, n_tiles_x=2, tile_overlap=0.15)
    robominnie.generate_gene_codes(n_genes=15)
    robominnie.generate_pink_noise(noise_amplitude=0.0015, noise_spatial_scale=0.1)
    # robominnie.view_images()
    # Add spots to DAPI image as larger spots
    robominnie.add_spots(n_spots=25_000, bleed_matrix=np.diag(np.ones(7)), 
                         spot_size_pixels=np.array([1.5, 1.5, 1.5]), include_dapi=True, 
                         spot_size_pixels_dapi=np.array([9, 9, 9]), spot_amplitude_dapi=0.05)
    # robominnie.generate_random_noise(noise_mean_amplitude=0, noise_std=0.0004, noise_type='normal')
    # robominnie.view_images()
    # NOTE: We are shortening the pipeline runtime by making the initial intensity threshold strict for OMP
    robominnie.save_raw_images(output_dir=output_dir, overwrite=True, omp_iterations=1,
                               omp_initial_intensity_thresh_percentile=90)
    robominnie.run_coppafish()

    robominnie.compare_spots('ref')
    # Basic scoring system for integration test
    overall_score = robominnie.overall_score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))

    tps, wps, fps, fns = robominnie.compare_spots('omp')
    print(tps)
    print(wps)
    print(fps)
    print(fns)
    # Basic scoring system for integration test
    overall_score = robominnie.overall_score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))


if __name__ == '__main__':
    test_integration_003()
