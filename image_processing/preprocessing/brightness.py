import cv2
import numpy as np
import math

def auto_brighten_image(image: np.ndarray, threshold: int = 160, target_brightness: int = 160) -> np.ndarray:
    """
    Tối ưu hóa độ sáng cho ảnh chụp OMR:
    1. Tính toán Gamma động (Dynamic Gamma) để đưa độ sáng trung bình về mức chuẩn.
    2. Sử dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) để cân bằng 
       vùng sáng tối cục bộ (giải quyết cực tốt hiện tượng sấp bóng, bóng tay che trên giấy).
    """
    if image is None:
        return image
        
    is_color = len(image.shape) == 3
    if is_color:
        # Sử dụng hệ màu LAB để chỉ can thiệp độ sáng (kênh L), giữ nguyên màu sắc thật (kênh A, B)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        gray = l
    else:
        gray = image
        
    mean_brightness = np.mean(gray)
    
    if mean_brightness < threshold:
        # 1. Gamma Correction Động
        mean_b = max(mean_brightness, 1.0)
        
        # Công thức: (mean_b / 255) ^ power = (target_brightness / 255)
        power = math.log(target_brightness / 255.0) / math.log(mean_b / 255.0)
        
        # Giới hạn power [0.3, 0.9] để tránh nhiễu hạt (noise) do kéo sáng quá gắt
        power = np.clip(power, 0.3, 0.9)
        
        table = np.array([((i / 255.0) ** power) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray_gamma = cv2.LUT(gray, table)
        
        # 2. CLAHE (Local Contrast Enhancement)
        # Xóa các mảng tối cục bộ trên tờ giấy (rất hay gặp khi chụp bằng điện thoại)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray_gamma)
        
        if is_color:
            lab_corrected = cv2.merge((gray_clahe, a, b))
            return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        else:
            return gray_clahe
            
    return image
