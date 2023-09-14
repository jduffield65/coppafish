import numpy as np
from coppafish.robominnie import RoboMinnie
import warnings

output_dir = '/home/paul/Documents/coppafish/robominnie-install/synthetic_output'
codebook_path = '/home/paul/Documents/coppafish/synthetic/codebook.txt'

# Produce the most basic synthetic data to run coppafish on and check if the spots are being detected
# No off-diagonals bleed matrix
bleed_matrix = np.diag([.3, .7, 1, 1, 1, 1, .8])
n_spots = 10_000
spot_size = np.array([1.5, 1.5, 1.5])

robominnie = RoboMinnie(n_channels=7, n_tiles=1, n_rounds=7, n_planes=4, include_anchor=True, include_preseq=False)
robominnie.Generate_Pink_Noise(noise_amplitude=0.0015, noise_spatial_scale=0.16)
robominnie.Add_Spots(n_spots=n_spots, bleed_matrix=bleed_matrix, gene_codebook_path=codebook_path, 
    spot_size_pixels=spot_size)
robominnie.Fix_Image_Minimum(minimum=0)
# Save the synthetic data in coppafish format as raw .npy files
robominnie.Save_Coppafish(output_dir=output_dir, overwrite=True)
robominnie.Run_Coppafish()
true_positives, wrong_positives, false_positives, false_negatives = robominnie.Compare_Spots_OMP()
print('OMP results:')
print(
    f'\tTrue positives: {true_positives}, wrong positives: {wrong_positives},' + \
    f' false positives: {false_positives}, false negatives: {false_negatives}'
)
# Super basic scoring system for integration test
overall_score = true_positives / (true_positives + wrong_positives + false_positives + false_negatives)
print(f'Overall score: {round(overall_score*100, 1)}%')
if overall_score < 0.9:
    warnings.warn('Integration test passed, but the overall score is < 90%')
assert overall_score > 0.5, 'Integration test score < 50%!'
