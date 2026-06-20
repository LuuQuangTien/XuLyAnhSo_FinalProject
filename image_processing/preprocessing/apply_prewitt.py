import cv2
import numpy as np
from image_processing.utils.image_utils import to_grayscale

def process(image: np.ndarray) -> np.ndarray:
    """Áp dụng bộ lọc Prewitt để phát hiện cạnh."""
    gray = to_grayscale(image)
    
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    
    # Kết hợp hai hướng
    prewitt = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)
    return prewitt
