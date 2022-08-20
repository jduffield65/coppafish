from .test_morphology import TestMorphology
from .test_npy import TestNPY
try:
    from .test_optimised import TestOptimisedImfilter
except ImportError:
    pass
