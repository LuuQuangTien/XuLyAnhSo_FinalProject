import cv2
import numpy as np
from image_processing.utils.image_utils import to_grayscale
from image_processing.utils.geometry_utils import order_points

def crop_paper_from_background(image, log_txt):
    """
    Tiền-tiền xử lý (Pre-cropping): Dùng 3 vòng bảo vệ Toán học để tìm và cắt tờ giấy khỏi nền.
    - Vòng 1: Bilateral Filter + Auto-Canny (Xử lý nhiễu ánh sáng/hoa văn mặt bàn).
    - Vòng 2: Adaptive Epsilon (Ép buộc approxPolyDP tìm ra 4 góc bằng cách nới lỏng sai số).
    - Vòng 3: Extreme Points (Lấy vỏ bọc Convex Hull và ép 4 điểm Cực Hạn làm 4 góc).
    Nếu tất cả đều thất bại, trả về ảnh gốc (Fallback).
    """
    height, width = image.shape[:2]
    image_area = width * height
    
    gray = to_grayscale(image)
    
    # 1. Bilateral Filter: Làm mịn vân gỗ, nhiễu hột nhưng GIỮ LẠI ĐỘ SẮC NÉT của mép giấy
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    cv2.imwrite(log_txt.replace(".txt", "_precrop_blurred.jpg"), blurred)
    
    # Auto-Canny: Tự động căn ngưỡng Canny dựa trên độ chói trung bình của ảnh
    v = np.median(blurred)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)
    
    # Dilation để nối liền các nét đứt ở mép giấy
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    paper_contour = None
    
    for c in cnts:
        area = cv2.contourArea(c)
        # Bỏ qua những cụm rác quá nhỏ (< 20% diện tích bức hình)
        if area < image_area * 0.2:
            continue
            
        peri = cv2.arcLength(c, True)
        
        # 2. Vòng lặp Ép Góc (Adaptive Epsilon)
        found_4_pts = False
        for eps_ratio in np.arange(0.01, 0.1, 0.01):
            approx = cv2.approxPolyDP(c, eps_ratio * peri, True)
            if len(approx) == 4:
                paper_contour = approx
                found_4_pts = True
                with open(log_txt, "a", encoding="utf-8") as f:
                    f.write(f"[PRE-CROP] Tìm thấy viền giấy hoàn hảo ở sai số epsilon={eps_ratio:.2f}\n")
                break
                
        if found_4_pts:
            break
            
        # 3. Chốt chặn Cực Hạn (Convex Hull + Extreme Points Fallback)
        hull = cv2.convexHull(c)
        hull_pts = hull[:, 0, :]
        
        if len(hull_pts) >= 4:
            rect = np.zeros((4, 2), dtype="float32")
            s = hull_pts.sum(axis=1)
            rect[0] = hull_pts[np.argmin(s)] # Trái-Trên (TL)
            rect[2] = hull_pts[np.argmax(s)] # Phải-Dưới (BR)
            
            diff = np.diff(hull_pts, axis=1)
            rect[1] = hull_pts[np.argmin(diff)] # Phải-Trên (TR)
            rect[3] = hull_pts[np.argmin(diff)] # Trái-Dưới (BL)
            
            quad_area = cv2.contourArea(rect)
            if quad_area > image_area * 0.2:
                paper_contour = np.array([rect[0], rect[1], rect[2], rect[3]], dtype="float32").reshape(4, 1, 2)
                with open(log_txt, "a", encoding="utf-8") as f:
                    f.write("[PRE-CROP] Cứu vớt thành công bằng Convex Hull Extreme Points!\n")
                break
                
    if paper_contour is not None:
        pts = paper_contour.reshape(4, 2).astype("float32")
        rect = order_points(pts)
        
        (tl, tr, br, bl) = rect
        w1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        w2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(w1), int(w2))
        
        h1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        h2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(h1), int(h2))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[PRE-CROP] Đã cắt thẳng tờ giấy. Kích thước mới: {maxWidth}x{maxHeight}\n")
            
        return warped
        
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write("[PRE-CROP] Thất bại! Kể cả dùng Convex Hull cũng không tìm được viền giấy. Giữ nguyên ảnh gốc.\n")
        
    return image
