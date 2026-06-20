import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh sang ảnh xám an toàn, chống lặp lại code ở nhiều nơi."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
