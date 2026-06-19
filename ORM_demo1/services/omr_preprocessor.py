# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import os

def apply_adaptive_threshold(gray_image):
    """
    Áp dụng Adaptive Threshold để khử bóng râm và chênh lệch sáng.
    """
    # Sharpening (Unsharp Mask) để làm rõ nét chì mờ và mép viền
    gaussian = cv2.GaussianBlur(gray_image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray_image, 1.5, gaussian, -0.5, 0)
    
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        31, 10
    )
    return thresh

def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def angle_between(p1, p2, p3):
    """Tính góc tại p2, bởi 2 vector p2->p1 và p2->p3 (trả về độ)"""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def _get_debug_paths(debug_dir, debug_prefix):
    if debug_dir and os.path.exists(debug_dir):
        return (os.path.join(debug_dir, f"{debug_prefix}align_debug.txt"),
                os.path.join(debug_dir, f"{debug_prefix}debug_align.jpg"))
    return ("align_debug.txt", "debug_align.jpg")

def fix_corners(pts, gray, debug_dir=None, debug_prefix=""):
    """
    pts: numpy array shape (4, 2), ordered as TL, TR, BR, BL
    Kiểm tra tứ giác, nếu bị méo (do bắt nhầm góc), tự động suy luận tọa độ và tìm lại góc bị sai.
    Hỗ trợ tự chữa lành tối đa 2 góc.
    """
    log_txt, _ = _get_debug_paths(debug_dir, debug_prefix)
    
    for attempt in range(2):
        angles = []
        angles.append(angle_between(pts[3], pts[0], pts[1])) # TL
        angles.append(angle_between(pts[0], pts[1], pts[2])) # TR
        angles.append(angle_between(pts[1], pts[2], pts[3])) # BR
        angles.append(angle_between(pts[2], pts[3], pts[0])) # BL
        
        # Một hình chữ nhật chụp qua perspective thường có góc từ 75 đến 105 độ
        if min(angles) > 75 and max(angles) < 105:
            return pts # Các góc khá ổn, không cần sửa
            
        # ĐIỂM CHỐT YẾU: Góc bị sai sẽ là góc bị biến dạng nhiều nhất so với 90 độ
        # (Nếu bị thụt vào trong thì góc đó sẽ tù >130 độ, nếu bị kéo ra ngoài thì nhọn <50 độ)
        bad_idx = int(np.argmax(np.abs(np.array(angles) - 90)))
        
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"\n[GEOMETRIC HEALING] Attempt {attempt+1}. Bad quad detected. Angles: {np.round(angles, 1)}\n")
            f.write(f"Bad corner index: {bad_idx} (Angle: {angles[bad_idx]:.1f})\n")
            
        # Dự đoán tọa độ điểm bị sai dựa vào 3 điểm còn lại
        # P_bad = P_prev + P_next - P_opp (Tịnh tiến vector hình bình hành)
        prev_idx = (bad_idx - 1) % 4
        next_idx = (bad_idx + 1) % 4
        opp_idx  = (bad_idx + 2) % 4
        
        P_prev = pts[prev_idx]
        P_next = pts[next_idx]
        P_opp  = pts[opp_idx]
        
        P_pred = P_prev + P_next - P_opp
        
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"Predicted coordinate: {P_pred}\n")
            
        # Tìm kiếm cục bộ (Local Search) quanh tọa độ dự đoán
        h, w = gray.shape
        r = 250 # Bán kính tìm kiếm (Tăng lên 250 để bù đắp sai số phối cảnh Perspective)
        sx = int(max(0, P_pred[0] - r))
        ex = int(min(w, P_pred[0] + r))
        sy = int(max(0, P_pred[1] - r))
        ey = int(min(h, P_pred[1] + r))
        
        roi = gray[sy:ey, sx:ex]
        if roi.size == 0: return pts
        
        # Dùng Otsu Thresholding siêu nhạy cho khu vực cục bộ
        _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi_thresh = cv2.medianBlur(roi_thresh, 3) # Lọc nhiễu
        
        cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_pt = None
        min_dist = float('inf')
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50: continue # Lọc rác nhỏ
            
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx_roi = int(M["m10"] / M["m00"])
            cy_roi = int(M["m01"] / M["m00"])
            
            cx = sx + cx_roi
            cy = sy + cy_roi
            
            dist = np.hypot(cx - P_pred[0], cy - P_pred[1])
            if dist < min_dist:
                min_dist = dist
                best_pt = [cx, cy]
                
        if best_pt is not None:
            pts[bad_idx] = best_pt
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"Healing successful! Replaced with: {best_pt} (Dist: {min_dist:.1f})\n")
        else:
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write("Healing failed! No suitable contour found in ROI.\n")
            break # Nếu không tìm thấy thì dừng, tránh lặp vô ích
            
    return pts

def filter_noise_corners_by_area(candidates, log_txt):
    """
    Thuật toán Bóc Vỏ (Peeling): 
    Lọc bỏ các điểm nhiễu ở rìa ảnh bằng cách kiểm tra độ đồng đều diện tích của 4 góc ngoài cùng.
    """
    valid_candidates = candidates.copy()
    
    while len(valid_candidates) > 4:
        pts = np.array([[c[0], c[1]] for c in valid_candidates])
        rect = order_points(pts)
        
        quad_candidates = []
        for r_pt in rect:
            for c in valid_candidates:
                if c[0] == r_pt[0] and c[1] == r_pt[1]:
                    quad_candidates.append(c)
                    break
                    
        # Nếu trùng điểm (do nhiễu tập trung quá gần nhau), dừng bóc vỏ
        if len(set([c[0]*10000+c[1] for c in quad_candidates])) < 4:
            break
            
        areas = [c[2] for c in quad_candidates]
        max_a, min_a = max(areas), min(areas)
        
        # Nếu 4 góc có diện tích tương đồng (sai lệch < 1.8 lần) -> Coi như đây là 4 góc chuẩn!
        if max_a / min_a <= 1.8: 
            break
            
        # Bóc bỏ góc có diện tích dị biệt nhất so với trung vị
        median_a = np.median(areas)
        worst_quad_idx = int(np.argmax(np.abs(np.array(areas) - median_a)))
        worst_point = quad_candidates[worst_quad_idx]
        
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"Peeling noise corner: {worst_point[:2]} Area: {worst_point[2]}\n")
            
        valid_candidates.remove(worst_point)
        
    return valid_candidates

def crop_paper_from_background(image, log_txt):
    """
    Tiền-tiền xử lý (Pre-cropping): Dùng Canny Edge Detection để tìm tờ giấy trắng trên nền tối.
    Nếu tìm thấy, cắt lấy tờ giấy (Perspective Transform) để loại bỏ hoàn toàn nhiễu từ nền bàn.
    Nếu thất bại (giấy rách, dính sát viền), trả về ảnh gốc (Fallback).
    """
    height, width = image.shape[:2]
    image_area = width * height
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny để dò viền
    edges = cv2.Canny(blurred, 50, 150)
    
    # Phóng to viền một chút để nối liền các đoạn đứt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lấy 5 viền lớn nhất
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    paper_contour = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < image_area * 0.2: # Phải chiếm ít nhất 20% khung hình
            continue
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Nếu đường viền có 4 góc, đó chính là tờ giấy!
        if len(approx) == 4:
            paper_contour = approx
            break
            
    if paper_contour is not None:
        pts = paper_contour.reshape(4, 2).astype("float32")
        rect = order_points(pts)
        
        # Tính kích thước tờ giấy mới
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
            f.write(f"[PRE-CROP] Successfully cropped paper from background. New size: {maxWidth}x{maxHeight}\n")
            
        return warped
        
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write("[PRE-CROP] Failed to find paper contour. Falling back to full image.\n")
        
    return image

def align_document(image, debug_dir=None, debug_prefix=""):
    """
    Tìm 4 điểm đen định vị ở 4 góc và nắn thẳng hình ảnh (Perspective Transform).
    """
    log_txt, log_jpg = _get_debug_paths(debug_dir, debug_prefix)
    
    # Bước 0: Tiền-tiền-xử-lý (Pre-cropping) tờ giấy trắng
    image = crop_paper_from_background(image, log_txt)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    image_area = width * height
    
    # Sharpening (Unsharp Mask) giúp các góc vuông sắc nét hơn, chống bị bo tròn do mờ (Blurry)
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    
    # Dùng Gaussian Blur vừa phải để làm mờ nhiễu (Hạ từ 9x9 xuống 5x5 để giữ độ sắc cạnh)
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
    
    # Adaptive threshold với block lớn để giữ khối đen
    block_size = int(width / 20)
    if block_size % 2 == 0: block_size += 1
    if block_size < 31: block_size = 31
        
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size, 15
    )
    
    # KHÔNG dùng dilate hoặc erode để tránh dính lẹo
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"Image Size: {width} x {height}, Area: {image_area}\n")
        f.write(f"Found {len(cnts)} total contours.\n")
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            contour_area = cv2.contourArea(c)
            if contour_area == 0: continue
            
            # Tính Min Area Rect để khử nhiễu xoay
            rect_min = cv2.minAreaRect(c)
            min_rect_area = rect_min[1][0] * rect_min[1][1]
            if min_rect_area == 0: continue
            
            # Hình vuông: extent ~ 1.0 | Hình tròn (bong bóng): extent ~ 0.78
            extent = contour_area / min_rect_area
            aspect_ratio = float(w) / h if h != 0 else 0
            area_ratio = contour_area / float(image_area)
            
            # CẬP NHẬT: Kiểm tra độ tương phản cao (High Contrast). 
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            
            if mean_intensity > 210:
                f.write(f"Rejected (Low Contrast): ({int(x+w/2)}, {int(y+h/2)}), Intensity: {mean_intensity:.1f}\n")
                continue
                
            if 0.0001 < area_ratio < 0.005 and 0.5 <= aspect_ratio <= 2.0 and extent > 0.5:
                cX, cY = int(x + w/2), int(y + h/2)
                centers.append((cX, cY, contour_area))
                f.write(f"Valid Center added: ({cX}, {cY}), Area: {contour_area}, Ratio: {area_ratio:.5f}, Extent: {extent:.2f}, Intensity: {mean_intensity:.1f}\n")
            elif contour_area > 100:
                f.write(f"Rejected (Size/Extent): ({int(x+w/2)}, {int(y+h/2)}), Area: {contour_area}, Ratio: {area_ratio:.5f}, Extent: {extent:.2f}\n")
                
        f.write(f"\nTotal Valid Centers: {len(centers)}\n")
        
    # Tạo ảnh debug
    debug_img = image.copy()
    for (cX, cY, _) in centers:
        cv2.circle(debug_img, (cX, cY), 10, (255, 0, 0), -1)
        
    if len(centers) < 4:
        cv2.putText(debug_img, "FAILED: Less than 4 centers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imwrite(log_jpg, debug_img)
        return image
        
    # Áp dụng thuật toán bóc vỏ để loại nhiễu
    filtered_centers = filter_noise_corners_by_area(centers, log_txt)
    
    # Chốt 4 điểm sau khi bóc vỏ
    pts = np.array([[c[0], c[1]] for c in filtered_centers])
    rect = order_points(pts)
    
    # --- GEOMETRIC HEALING ---
    rect = fix_corners(rect, gray, debug_dir, debug_prefix)
    # -------------------------
    
    (tl, tr, br, bl) = rect
    
    # Vẽ 4 điểm được chọn lên ảnh debug
    cv2.circle(debug_img, tuple(tl.astype(int)), 20, (0, 255, 0), -1) # Xanh lá
    cv2.circle(debug_img, tuple(tr.astype(int)), 20, (0, 255, 0), -1)
    cv2.circle(debug_img, tuple(br.astype(int)), 20, (0, 255, 0), -1)
    cv2.circle(debug_img, tuple(bl.astype(int)), 20, (0, 255, 0), -1)
    
    # Kiểm tra an toàn: 4 điểm ngoài cùng phải tạo thành hình chữ nhật bao phủ phần lớn bức ảnh
    w_rect = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    h_rect = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"Selected 4 corners: TL={tl}, TR={tr}, BR={br}, BL={bl}\n")
        f.write(f"Rect Width: {w_rect:.2f} (Required: {width * 0.4:.2f})\n")
        f.write(f"Rect Height: {h_rect:.2f} (Required: {height * 0.4:.2f})\n")
    
    if w_rect < width * 0.4 or h_rect < height * 0.4:
        # Nếu quá bé, tức là nó bắt nhầm mớ bong bóng thay vì 4 góc giấy
        cv2.putText(debug_img, "FAILED: Rect too small", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imwrite(log_jpg, debug_img)
        return image
        
    cv2.putText(debug_img, "SUCCESS: Aligned", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imwrite(log_jpg, debug_img)
    
    maxWidth = int(max(w_rect, np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))))
    maxHeight = int(max(h_rect, np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))))
    
    # KHIỂM TRA BẢO VỆ: Nếu 4 điểm tìm được tạo thành hình quá nhỏ (dưới 40% kích thước ảnh)
    # thì chắc chắn đó là 4 bọt bong bóng bên trong chứ không phải 4 góc giấy. Trả về ảnh gốc!
    if maxWidth < width * 0.4 or maxHeight < height * 0.4:
        return image
        
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def find_blocks(thresh: np.ndarray, debug_dir=None, debug_prefix="", original_img=None) -> list:
    """Tìm các khối chữ nhật (ROI) trên ảnh nhị phân."""
    # Dùng morphology close để nối liền các nét đứt của viền bảng
    kernel = np.ones((5, 5), np.uint8)
    closed_thresh = cv2.morphologyEx(thresh.copy(), cv2.MORPH_CLOSE, kernel)
    
    # --- PROJECTION PROFILE CUTTING BẰNG W_STD (Cắt hình học tự động tỷ lệ) ---
    # Thay vì cắt mù, ta tính độ rộng chuẩn (W_std) của một block Answer, sau đó chỉ cắt những block béo phì.
    
    debug_gaps_img = None
    if debug_dir and os.path.exists(debug_dir) and original_img is not None:
        debug_gaps_img = original_img.copy()
        
    new_closed_thresh = closed_thresh.copy()
    
    for _ in range(5): # Tối đa cắt 5 lần
        cnts_tmp, _ = cv2.findContours(new_closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_widths = []
        for c in cnts_tmp:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 15000 and float(w)/h < 2.5:
                valid_widths.append(w)
                
        if not valid_widths:
            break
            
        W_std = np.median(valid_widths)
        cut_happened = False
        
        for c in cnts_tmp:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 15000 and float(w)/h < 2.5:
                # 1. Cắt dọc nếu khối rộng bất thường (rộng hơn 1.35 lần W_std)
                if w > 1.35 * W_std:
                    num_blocks = max(1, round(w / W_std))
                    roi = new_closed_thresh[y:y+h, x:x+w]
                    proj = np.sum(roi, axis=0) / 255.0
                    
                    # Cắt tại các vị trí k * W_std (với k=1..num_blocks)
                    for k in range(1, int(num_blocks) + (1 if num_blocks==1 else 0)):
                        expected_x = int(k * W_std)
                        if expected_x >= w: break
                        
                        # Tìm rãnh trắng (valley) xung quanh vị trí dự kiến (khoảng +/- 40px)
                        search_start = max(0, expected_x - 40)
                        search_end = min(w, expected_x + 40)
                        
                        if search_end > search_start:
                            search_region = proj[search_start:search_end]
                            valley_min = np.min(search_region)
                            # Tìm tất cả các điểm có giá trị xấp xỉ đáy thung lũng (khoảng trống giữa 2 block)
                            valley_indices = np.where(search_region <= valley_min + 10)[0]
                            
                            if len(valley_indices) > 0:
                                # valley_indices[0] chính là viền dọc cuối của block hiện tại
                                # valley_indices[-1] chính là viền dọc bắt đầu của block tiếp theo!
                                first_idx = search_start + valley_indices[0]
                                last_idx = search_start + valley_indices[-1]
                                
                                # Xóa sổ toàn bộ vùng gap này (tẩy đen) để cắt đứt mọi bóng râm/nhiễu
                                cv2.rectangle(new_closed_thresh, (x + first_idx, y), (x + last_idx, y + h), 0, -1)
                                
                                if debug_gaps_img is not None:
                                    cv2.rectangle(debug_gaps_img, (x + first_idx, y), (x + last_idx, y + h), (0, 255, 255), -1) # Bôi vàng vùng gap bị xóa
                                cut_happened = True
        if not cut_happened:
            break
            
    closed_thresh = new_closed_thresh
    # --- END CUTTING ---
    
    cnts, _ = cv2.findContours(closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
        # Hạ ngưỡng diện tích xuống 15000 để bắt được khối SBD và Mã đề
        # Nới lỏng điều kiện aspect_ratio < 2.5 để cho phép khối SBD và Mã đề bị dính liền thành 1 khối ngang (aspect ~ 1.4)
        # Nhưng vẫn đủ nghiêm ngặt để diệt trừ các hộp text dài ngoẵng (như "HỌ VÀ TÊN" có aspect > 7.0)
        if w * h > 15000 and aspect_ratio < 2.5: 
            blocks.append((x, y, w, h))
            
    if debug_dir and os.path.exists(debug_dir) and original_img is not None:
        debug_img = original_img.copy()
        with open(os.path.join(debug_dir, f"{debug_prefix}blocks_debug.txt"), "w", encoding="utf-8") as f:
            f.write(f"Total contours found in closed_thresh: {len(cnts)}\n")
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h if h != 0 else 0
                area = w * h
                if area > 15000:
                    if aspect_ratio < 2.5:
                        f.write(f"ACCEPTED: Area={area}, Aspect={aspect_ratio:.2f}, Box=({x},{y},{w},{h})\n")
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3) # Xanh lá cho block hợp lệ
                        cv2.putText(debug_img, f"OK {aspect_ratio:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        f.write(f"REJECTED (Aspect > 2.5): Area={area}, Aspect={aspect_ratio:.2f}, Box=({x},{y},{w},{h})\n")
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2) # Đỏ cho block bị loại (như Họ Tên)
                        cv2.putText(debug_img, f"REJ {aspect_ratio:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}debug_blocks.jpg"), debug_img)
        # Thêm ảnh debug cho bước hoàn thành tiền xử lý (closed_thresh)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}debug_closed_thresh.jpg"), closed_thresh)
        if debug_gaps_img is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}debug_gaps.jpg"), debug_gaps_img)
            
    return blocks

