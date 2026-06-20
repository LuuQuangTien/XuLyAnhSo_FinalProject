import cv2
import numpy as np
from image_processing.utils.image_utils import to_grayscale

def process(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh sang thang độ xám."""
    return to_grayscale(image)
