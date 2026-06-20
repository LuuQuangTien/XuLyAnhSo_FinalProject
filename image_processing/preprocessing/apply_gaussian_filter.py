import cv2
import numpy as np

def process(image: np.ndarray) -> np.ndarray:
    """Áp dụng bộ lọc Gaussian Blur để giảm nhiễu."""
    return cv2.GaussianBlur(image, (5, 5), 0)
