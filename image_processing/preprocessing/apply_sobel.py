import cv2
import numpy as np
from image_processing.utils.image_utils import to_grayscale

def process(image: np.ndarray) -> np.ndarray:
    """Áp dụng bộ lọc Sobel để phát hiện cạnh."""
    gray = to_grayscale(image)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return sobel
