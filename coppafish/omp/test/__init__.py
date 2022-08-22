from .test_all import TestFittingStandardDeviation, TestCountSpotNeighbours
try:
    from .test_optimised import TestFitCoefs, TestGetAllCoefs, TestGetBestGene
except ImportError:
    pass
