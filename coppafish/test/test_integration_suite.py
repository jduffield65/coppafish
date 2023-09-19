import os
import numpy as np
from coppafish.robominnie import RoboMinnie
import warnings

codebook_contents_73g = """Bcl11b 1234560
Cadps2 2345601
Calb1 3456012
Calb2 4560123
Cck 5601234
Cdh13 6012345
Chodl 0246135
Chrm2 1350246
Cnr1 2461350
Col25a1 3502461
Cort 4613502
Cox6a2 5024613
Cplx2 6135024
Cpne5 0362514
Crh 1403625
Crhbp 2514036
Cryab 3625140
Cxcl14 4036251
Enpp2 5140362
Gabrd 6251403
Gad1 0415263
Gda 1526304
Grin3a 2630415
Hapln1 3041526
Htr3a 4152630
Id2 5263041
Kcnk2 6304152
Kctd12 0531642
Kit 1642053
Lamp5 2053164
Lhx6 3164205
Ndnf 4205316
Neurod6 5316420
Nos1 6420531
Nov 0654321
Npy 6410424
Npy2r 1065432
Nr4a2 2106543
Nrn1 3210654
Ntng1 4321065
Pcp4 5432106
Pde1a 6543210
Penk 0142241
Plcxd2 1253352
Plp1 2364463
Pnoc 3405504
Prkca 4516615
Pthlh 5620026
Pvalb 6031130
Rab3c 0265620
Rasgrf2 1306031
Reln 2410142
Rgs10 3521253
Rgs12 4632364
Rgs4 5043405
Satb1 6154516
Sema3c 0311306
Serpini1 1422410
Slc17a8 2533521
Slc6a1 3644632
Snca 4055043
Sncg 5166154
Sst 6200265
Synpr 0434055
Tac1 1545166
Tac2 2656200
Th 3060311
Thsd7a 4101422
Trp53i11 5212533
Vip 6323644
Wfs1 0550434
Yjefn3 1661545
"""


def _overall_score(TPs : int, WPs : int, FPs : int, FNs : int):
    return TPs / (TPs + WPs + FPs + FNs)


def test_integration_001() -> None:
    """
    Summary of input data: random spots and random, white noise.
    No presequence round, anchor round, single tile.
    """
    output_dir = '/tmp/integration'
    codebook_path = ''
    bleed_matrix = np.diag(np.ones(7))
    n_spots = 10_000
    spot_size = np.array([1.5, 1.5, 1.5])

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Save codebook as .txt file
    codebook_path = os.path.join(output_dir, 'codebook.txt')
    with open(codebook_path, 'w') as f:
        f.write(codebook_contents_73g)

    robominnie = RoboMinnie(include_anchor=True, seed=94)
    robominnie.Add_Spots(n_spots=n_spots, bleed_matrix=bleed_matrix, gene_codebook_path=codebook_path, 
        spot_size_pixels=spot_size)
    robominnie.Generate_Random_Noise(noise_mean_amplitude=0, noise_std=0.001, noise_type='normal')
    robominnie.Fix_Image_Minimum(minimum=0)
    # Save the synthetic data in coppafish format as raw .npy files
    robominnie.Save_Coppafish(output_dir=output_dir, overwrite=True)
    robominnie.Run_Coppafish()

    true_positives, wrong_positives, false_positives, false_negatives = robominnie.Compare_OMP_Spots()
    print('OMP results:')
    print(
        f'\tTrue positives: {true_positives}, wrong positives: {wrong_positives},' + \
        f' false positives: {false_positives}, false negatives: {false_negatives}'
    )
    # Super basic scoring system for integration test
    overall_score = _overall_score(true_positives, wrong_positives, false_positives, false_negatives)
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.9:
        warnings.warn('Integration test passed, but the overall score is < 90%')
    assert overall_score > 0.5, 'Integration OMP spots score < 50%!'


def test_integration_002() -> None:
    """
    Summary of input data: random spots, pink noise (represents biological features) and random, white noise. \
        No presequence round, anchor round, single tile. Run everything except OMP. Comparing ground truth spots \
        to reference spots
    """
    output_dir = '/tmp/integration'
    codebook_path = ''
    bleed_matrix = np.diag(np.ones(7))
    n_spots = 10_000
    spot_size = np.array([1.5, 1.5, 1.5])

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Save codebook as .txt file
    codebook_path = os.path.join(output_dir, 'codebook.txt')
    with open(codebook_path, 'w') as f:
        f.write(codebook_contents_73g)

    robominnie = RoboMinnie(include_anchor=True, seed=146)
    robominnie.Generate_Pink_Noise(noise_amplitude=0.0015, noise_spatial_scale=0.16)
    robominnie.Add_Spots(n_spots=n_spots, bleed_matrix=bleed_matrix, gene_codebook_path=codebook_path, 
        spot_size_pixels=spot_size)
    robominnie.Generate_Random_Noise(noise_mean_amplitude=0, noise_std=0.001, noise_type='normal')
    robominnie.Fix_Image_Minimum(minimum=0)
    # Save the synthetic data in coppafish format as raw .npy files
    robominnie.Save_Coppafish(output_dir=output_dir, overwrite=True)
    robominnie.Run_Coppafish(include_omp=False)

    true_positives, wrong_positives, false_positives, false_negatives = robominnie.Compare_Ref_Spots()
    print('Reference spots results:')
    print(
        f'\tTrue positives: {true_positives}, wrong positives: {wrong_positives},' + \
        f' false positives: {false_positives}, false negatives: {false_negatives}'
    )
    # Super basic scoring system for integration test
    overall_score = _overall_score(true_positives, wrong_positives, false_positives, false_negatives)
    print(f'Overall score: {round(overall_score*100, 1)}%')
    if overall_score < 0.9:
        warnings.warn('Integration test passed, but the overall score is < 90%')
    assert overall_score > 0.5, 'Integration reference spots score < 50%!'


if __name__ == '__main__':
    test_integration_001()
    test_integration_002()
