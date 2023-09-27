import os
import numpy as np
from coppafish.robominnie import RoboMinnie
import warnings
import pytest


@pytest.mark.slow
def test_integration_001() -> None:
    """
    Summary of input data: random spots and random, white noise.\n
    Includes anchor round, sequencing rounds, one tile.\n
    Compares ground truth spots to OMP spots and reference spots.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(include_anchor=True, include_presequence=False, include_dapi=False, n_tiles=1, seed=94)
    robominnie.Generate_Gene_Codes(n_genes=15)
    robominnie.Generate_Pink_Noise(noise_amplitude=0.0015, noise_spatial_scale=0.1)
    robominnie.Add_Spots(n_spots=10_000, bleed_matrix=np.diag(np.ones(7)), \
                         spot_size_pixels=np.array([1.5, 1.5, 1.5]))
    robominnie.Generate_Random_Noise(noise_mean_amplitude=0, noise_std=0.0004, noise_type='normal')
    robominnie.Fix_Image_Minimum(minimum=0)
    # Save the synthetic data in coppafish format as raw .npy files
    # NOTE: We are shortening the pipeline runtime by making the initial intensity threshold strict for OMP
    robominnie.Save_Raw_Images(output_dir=output_dir, overwrite=True, omp_iterations=2, \
                               omp_initial_intensity_thresh_percentile=90)
    robominnie.Run_Coppafish(save_ref_spots_data=True)

    robominnie.Compare_Ref_Spots()
    # Basic scoring system for integration test
    overall_score = robominnie.Overall_Score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))
    assert overall_score > 0.5, 'Integration reference spots score < 50%!'

    robominnie.Compare_OMP_Spots()
    # Basic scoring system for integration test
    overall_score = robominnie.Overall_Score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))
    assert overall_score > 0.5, 'Integration OMP spots score < 50%!'


@pytest.mark.slow
def test_integration_002() -> None:
    """
    Summary of input data: random spots and random, white noise.\n
    Includes anchor round, DAPI image, presequence round, sequencing rounds, one tile.\n
    Compares ground truth spots to OMP spots and reference spots.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'integration_dir')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(include_anchor=True, n_planes=10, include_presequence=True, include_dapi=True, 
                            n_tiles=1, seed=94)
    robominnie.Generate_Gene_Codes(n_genes=15)
    robominnie.Generate_Pink_Noise(noise_amplitude=0.0015, noise_spatial_scale=0.1)
    # robominnie.View_Images()
    # Add spots to DAPI image as larger spots
    robominnie.Add_Spots(n_spots=10_000, bleed_matrix=np.diag(np.ones(7)), 
                         spot_size_pixels=np.array([1.5, 1.5, 1.5]), include_dapi=True, 
                         spot_size_pixels_dapi=np.array([9, 9, 9]), spot_amplitude_dapi=0.05)
    # robominnie.Generate_Random_Noise(noise_mean_amplitude=0, noise_std=0.0004, noise_type='normal')
    robominnie.Fix_Image_Minimum(minimum=0)
    # robominnie.View_Images()
    # Save the synthetic data in coppafish format as raw .npy files
    # NOTE: We are shortening the pipeline runtime by making the initial intensity threshold strict for OMP
    robominnie.Save_Raw_Images(output_dir=output_dir, overwrite=True, omp_iterations=1, \
                               omp_initial_intensity_thresh_percentile=90)
    robominnie.Run_Coppafish(save_ref_spots_data=True)

    robominnie.Compare_Ref_Spots()
    # Basic scoring system for integration test
    overall_score = robominnie.Overall_Score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))
    assert overall_score > 0.5, 'Integration reference spots score < 50%!'

    robominnie.Compare_OMP_Spots()
    # Basic scoring system for integration test
    overall_score = robominnie.Overall_Score()
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.75:
        warnings.warn(UserWarning('Integration test passed, but the overall OMP spots score is < 75%'))
    assert overall_score > 0.5, 'Integration OMP spots score < 50%!'


if __name__ == '__main__':
    test_integration_002()
