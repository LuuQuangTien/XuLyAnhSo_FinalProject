# Thuật toán xử lý ảnh: Chuyển đổi sang ảnh xám (Grayscale).
import numpy as np

def process(image, **kwargs):
    
    if image is None:
        return None

    if image.ndim == 2:
        return image  # Đã là ảnh xám

    # image[:,:,0]=B, image[:,:,1]=G, image[:,:,2]=R (thứ tự BGR)
    gray = (0.114 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.299 * image[:, :, 2])
    return gray.astype(np.uint8)
    #uint8 đảm bảo kết quả là số từ 0 - 255
