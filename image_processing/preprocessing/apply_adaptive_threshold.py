import cv2
import numpy as np
from image_processing.utils.image_utils import to_grayscale

def normalize_illumination(gray_image):
    h, w = gray_image.shape
    # Tối ưu tốc độ: Thu nhỏ ảnh để ước lượng bề mặt ánh sáng (Background Surface)
    scale = 0.1
    # Dùng INTER_AREA để lấy mẫu màu chính xác hơn khi thu nhỏ
    small = cv2.resize(gray_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Kernel size 15 tương đương 150 pixel trên ảnh gốc, đủ lớn để che lấp nét chữ/bong bóng
    k = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    
    # Dùng Closing để xóa nét đen, chỉ giữ lại nền giấy và dải bóng râm/hắt sáng
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    # Làm mượt bề mặt ánh sáng để tránh tạo viền (artifacts)
    bg_small = cv2.GaussianBlur(bg_small, (k|1, k|1), 0)
    
    # Phóng to bản đồ ánh sáng về kích thước gốc
    background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Chuẩn hóa (Normalization): Ảnh gốc / Bản đồ ánh sáng * 255
    background = background.astype(np.float32) + 1e-5
    normalized = 255.0 * (gray_image.astype(np.float32) / background)
    
    return np.clip(normalized, 0, 255).astype(np.uint8)

def process(image: np.ndarray, block_size: int = 31, C: int = 10, sharpen: bool = True, blur: bool = True, debug_path: str = "") -> np.ndarray:
    """
    Áp dụng Adaptive Threshold kết hợp Illumination Normalization để khử bóng râm hoàn toàn.
    """
    gray = to_grayscale(image)
    if debug_path:
        cv2.imwrite(f"{debug_path}01_raw_gray.jpg", gray)
    
    # Kích hoạt chuẩn hóa ánh sáng (Khử bóng râm, hắt sáng)
    gray = normalize_illumination(gray)
    if debug_path:
        cv2.imwrite(f"{debug_path}02_illumination_normalized.jpg", gray)
    
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
        
    if debug_path:
        cv2.imwrite(f"{debug_path}03_blurred_{block_size}.jpg", blurred)

    # Đảm bảo block_size luôn là số lẻ
    if block_size % 2 == 0:
        block_size += 1
        
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size, C
    )
    
    if debug_path:
        cv2.imwrite(f"{debug_path}04_final_thresh_{block_size}.jpg", thresh)
        
    return thresh
