import cv2
import numpy as np

def _gaussian_kernel_1d(size, sigma=0):
    if sigma <= 0:
        # Công thức chuẩn của OpenCV để tự động tính sigma dựa trên size
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    
    # Tạo lưới tọa độ từ -(size//2) đến size//2
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    # Áp dụng công thức hàm Gaussian 1D: g(x) = e^(-x^2 / (2 * sigma^2))
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    # Chuẩn hóa để tổng bằng 1
    return kernel / np.sum(kernel)

def process(image, kernel_size=3, sigma=0, **kwargs):
    
    if image is None:
        return None
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Tự tạo Kernel Gaussian 1D bằng công thức toán học rồi nhân lại thành 2D (tính chất tách được)
    k1d = _gaussian_kernel_1d(kernel_size, sigma)
    kernel = np.outer(k1d, k1d).astype(np.float32)
    
    # Thực hiện tích chập bằng cv2.filter2D (tối ưu hóa tích chập ma trận)
    return cv2.filter2D(image, -1, kernel)
