try:
    from .base_optimised import get_spot_colors, all_pixel_yxz, apply_transform
except ImportError:
    from .base import get_spot_colors, all_pixel_yxz, apply_transform

# TODO: Optimise these functions
from .base import remove_background, normalise_rc
