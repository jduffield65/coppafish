try:
    from .detect_optimised import detect_spots
except ImportError:
    from .detect import detect_spots

from .base import get_isolated, check_neighbour_intensity, spot_yxz, spot_isolated, get_isolated_points, \
    load_spot_info, filter_intense_spots
from .check_spots import check_n_spots
