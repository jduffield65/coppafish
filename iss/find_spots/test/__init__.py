from .test_base import TestIsolated
from .test_neighbour import TestNeighbour
try:
    from .test_optimised import TestDetectSpots
except ImportError:
    pass
