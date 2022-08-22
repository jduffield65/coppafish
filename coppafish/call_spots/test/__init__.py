from .test_bleed_matrix import TestScaledKMeans, TestGetBleedMatrix, TestGetDyeChannelIntensityGuess
from .test_base import TestColorNormalisation, TestGetGeneEfficiency
try:
    from .test_optimised import TestDotProductScore, TestFitBackground, TestGetSpotIntensity
except ImportError:
    pass
