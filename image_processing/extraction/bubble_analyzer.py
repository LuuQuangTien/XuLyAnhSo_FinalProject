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
