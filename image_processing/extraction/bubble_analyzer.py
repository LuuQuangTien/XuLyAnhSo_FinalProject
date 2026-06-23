import cv2
import numpy as np

def create_bubble_mask(radius):
    size = radius * 2
    mask = np.zeros((size, size), dtype="uint8")
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    return mask

def count_bubble_pixels(thresh, cx, cy, radius, precomputed_mask=None):
    sy, ey = max(0, cy - radius), min(thresh.shape[0], cy + radius)
    sx, ex = max(0, cx - radius), min(thresh.shape[1], cx + radius)
    roi = thresh[sy:ey, sx:ex]
    if roi.size == 0: return 0
    
    if precomputed_mask is not None and roi.shape == precomputed_mask.shape:
        return cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=precomputed_mask))
    
    mask = np.zeros(roi.shape, dtype="uint8")
    mcx, mcy = cx - sx, cy - sy
    cv2.circle(mask, (mcx, mcy), radius, 255, -1)
    
    return cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=mask))

def count_projection_peak(thresh, cx, cy, radius):
    """
    Kỹ thuật Projection Profile (Phân bố 1D).
    Trích xuất một dải ngang xuyên qua tâm bong bóng để tìm đỉnh tháp (Peak).
    Miễn nhiễm hoàn toàn với lỗi lệch lưới theo chiều ngang.
    """
    strip_h = max(3, int(radius * 0.8))
    search_w = int(radius * 1.5)
    
    sy, ey = max(0, cy - strip_h), min(thresh.shape[0], cy + strip_h)
    sx, ex = max(0, cx - search_w), min(thresh.shape[1], cx + search_w)
    strip = thresh[sy:ey, sx:ex]
    
    if strip.size == 0: return 0
    
    proj = np.sum(strip, axis=0) / 255.0
    return int(np.max(proj))

def create_bubble_mask(radius):
    if isinstance(radius, (tuple, list)):
        rx, ry = int(radius[0]), int(radius[1])
    else:
        rx = ry = int(radius)
        
    mask = np.zeros((2 * ry, 2 * rx), dtype="uint8")
    cv2.ellipse(mask, (rx, ry), (rx, ry), 0, 0, 360, 255, -1)
    return mask

def evaluate_bubble_xor(thresh, cx, cy, radius, precomputed_mask=None):
    if isinstance(radius, (tuple, list)):
        rx, ry = int(radius[0]), int(radius[1])
    else:
        rx = ry = int(radius)
        
    sy, ey = max(0, int(cy - ry)), min(thresh.shape[0], int(cy + ry))
    sx, ex = max(0, int(cx - rx)), min(thresh.shape[1], int(cx + rx))
    roi = thresh[sy:ey, sx:ex]
    if roi.size == 0: return float('inf')
    
    if precomputed_mask is not None and roi.shape == precomputed_mask.shape:
        xor_result = cv2.bitwise_xor(roi, precomputed_mask)
        xor_inside = cv2.bitwise_and(xor_result, precomputed_mask)
        return cv2.countNonZero(xor_inside)
        
    mask = np.zeros(roi.shape, dtype="uint8")
    mcx, mcy = int(cx - sx), int(cy - sy)
    cv2.ellipse(mask, (mcx, mcy), (rx, ry), 0, 0, 360, 255, -1)
    
    xor_result = cv2.bitwise_xor(roi, mask)
    xor_inside = cv2.bitwise_and(xor_result, mask)
    return cv2.countNonZero(xor_inside)

def get_bubble_centroid_vector(thresh, cx, cy, radius):
    if isinstance(radius, (tuple, list)):
        rx, ry = int(radius[0]), int(radius[1])
    else:
        rx = ry = int(radius)
        
    sx, sy = max(0, int(cx - rx)), max(0, int(cy - ry))
    ex, ey = min(thresh.shape[1], int(cx + rx)), min(thresh.shape[0], int(cy + ry))
    
    roi = thresh[sy:ey, sx:ex]
    if roi.size == 0: return 0, 0
    
    mask = np.zeros(roi.shape, dtype="uint8")
    mcx, mcy = int(cx - sx), int(cy - sy)
    cv2.ellipse(mask, (mcx, mcy), (rx, ry), 0, 0, 360, 255, -1)
    
    roi_masked = cv2.bitwise_and(roi, mask)
    M = cv2.moments(roi_masked)
    if M["m00"] > 0:
        actual_cx = sx + M["m10"] / M["m00"]
        actual_cy = sy + M["m01"] / M["m00"]
        return int(round(actual_cx - cx)), int(round(actual_cy - cy))
    
    return 0, 0
