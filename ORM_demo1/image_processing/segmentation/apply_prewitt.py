import numpy as np
import cv2
from image_processing.intensity_transform import apply_gray

def process(image, **kwargs):
    if image is None:
        return None
    
    # Chuyển đổi sang ảnh xám
    gray = apply_gray.process(image)
        
    # Định nghĩa kernel Prewitt
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    # Tính đạo hàm
    gx = cv2.filter2D(gray.astype(np.float32), -1, kx)
    gy = cv2.filter2D(gray.astype(np.float32), -1, ky)
    
    # Tính độ lớn vector biên
    g = np.sqrt(gx**2 + gy**2)
    
    # Chuẩn hóa về [0, 255]
    g = np.clip(g, 0, 255)
    
    return g.astype(np.uint8)
