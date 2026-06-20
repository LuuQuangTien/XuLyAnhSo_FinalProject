import numpy as np
import cv2
from image_processing.intensity_transform import apply_gray

def process(image, **kwargs):
    
    if image is None:
        return None
    
    # Sử dụng code từ file apply_gray thay vì cv2.cvtColor
    gray = apply_gray.process(image)
        
    # Định nghĩa kernel Sobel theo lý thuyết Chương 2
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Sử dụng cv2.filter2D (phép tích chập) để tính đạo hàm theo x và y
    # Đây là cách thể hiện tính thuật toán (nhân chập ma trận)
    gx = cv2.filter2D(gray.astype(np.float32), -1, kx)
    gy = cv2.filter2D(gray.astype(np.float32), -1, ky)
    
    # Tính độ lớn vector biên: G = sqrt(gx^2 + gy^2)
    g = np.sqrt(gx**2 + gy**2)
    
    # Chuẩn hóa về [0, 255]
    g = np.clip(g, 0, 255)
    
    return g.astype(np.uint8)
