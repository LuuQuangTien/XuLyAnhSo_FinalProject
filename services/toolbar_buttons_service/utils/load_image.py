# Hàm hỗ trợ đọc file ảnh từ ổ cứng (Hỗ trợ đường dẫn Unicode).
import cv2
import numpy as np

def execute(image_path):
    """Load image with Unicode-safe path handling."""
    if not image_path:
        return None
    img_array = np.fromfile(image_path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
