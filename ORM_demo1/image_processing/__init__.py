from .intensity_transform import apply_gray
from .spatial_domain import apply_gaussian_filter
from .segmentation import apply_sobel
from .segmentation import apply_prewitt

__all__ = [
    'apply_gray',
    'apply_gaussian_filter',
    'apply_sobel',
    'apply_prewitt',
]
