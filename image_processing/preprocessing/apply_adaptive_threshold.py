import cv2
import numpy as np
from image_processing.utils.image_utils import to_grayscale

def process(image: np.ndarray, block_size: int = 31, C: int = 10, sharpen: bool = True, blur: bool = True) -> np.ndarray:
    """
    Áp dụng Adaptive Threshold để nhị phân hóa ảnh.
    Nếu blur=False, bỏ qua hoàn toàn làm mờ (Rất quan trọng khi đọc bong bóng để không làm bay nét chì mờ).
    """
    gray = to_grayscale(image)
    
    if blur:
        if sharpen:
            # Sharpening (Unsharp Mask) để làm rõ nét chì mờ và mép viền
            gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
            sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
        else:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        blurred = gray # Bỏ qua blur hoàn toàn để giữ nguyên cường độ nét chì


    # Đảm bảo block_size luôn là số lẻ
    if block_size % 2 == 0:
        block_size += 1
        
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size, C
    )
    return thresh
