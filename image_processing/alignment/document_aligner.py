"""
Module căn chỉnh tài liệu (Document Alignment).
Chứa toàn bộ logic tìm 4 điểm neo, bẻ phẳng ảnh (Perspective Transform),
cắt giấy khỏi nền và chữa lành góc bị sai.
"""
import cv2
import numpy as np
import os
import glob
import tempfile

from image_processing.utils.image_utils import to_grayscale
from image_processing.utils.geometry_utils import order_points, angle_between

from image_processing.alignment.ai_helper import get_ai_paper_bounding_box
from image_processing.alignment.precrop import crop_paper_from_background
from image_processing.alignment.geometric_search import _get_debug_paths, fix_corners, find_best_geometric_quad

def clear_ai_cache():
    try:
        cache_files = glob.glob(os.path.join(tempfile.gettempdir(), "omr_ai_cache_*.*"))
        for f in cache_files:
            os.remove(f)
    except:
        pass

def align_document(image, debug_dir=None, debug_prefix="", method="four_corners", use_ai=False):
    """
    Tìm 4 điểm đen định vị ở 4 góc và nắn thẳng hình ảnh (Perspective Transform).
    Returns: (aligned_image, error_msg)
    """
    error_msg = ""
    h_orig, w_orig = image.shape[:2]
    if w_orig > h_orig:
        return image, "Lỗi: Ảnh bị chụp nằm ngang. Vui lòng chụp ảnh theo chiều dọc (không xoay ngang điện thoại)."
    log_txt, log_jpg = _get_debug_paths(debug_dir, debug_prefix)
    
    image = crop_paper_from_background(image, log_txt)
    
    gray = to_grayscale(image)
    mean_brightness = np.mean(gray)
    
    height, width = image.shape[:2]
    image_area = width * height
    
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
    
    block_size = int(width / 20)
    if block_size % 2 == 0: block_size += 1
    if block_size < 31: block_size = 31
    
    # Dynamic C: Hạ hằng số C xuống thấp để không "xóa sổ" các điểm đen trên nền tối
    if mean_brightness < 90:
        C_val = 5
    elif mean_brightness < 130:
        C_val = 10
    else:
        C_val = 15
        
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size, C_val
    )
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"Image Size: {width} x {height}, Area: {image_area}\n")
        f.write(f"Mean Brightness: {mean_brightness:.1f}, Using adaptiveThreshold C={C_val}\n")
        f.write(f"Found {len(cnts)} total contours.\n")
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            contour_area = cv2.contourArea(c)
            if contour_area == 0: continue
            
            rect_min = cv2.minAreaRect(c)
            min_rect_area = rect_min[1][0] * rect_min[1][1]
            if min_rect_area == 0: continue
            
            extent = contour_area / min_rect_area
            aspect_ratio = float(w) / h if h != 0 else 0
            area_ratio = contour_area / float(image_area)
            
            roi_gray = gray[y:y+h, x:x+w]
            mask_roi = np.zeros(roi_gray.shape, dtype=np.uint8)
            c_local = c - [x, y]
            cv2.drawContours(mask_roi, [c_local], -1, 255, -1)
            mean_intensity = cv2.mean(roi_gray, mask=mask_roi)[0] if roi_gray.size > 0 else 0
            
            if mean_intensity > 210:
                f.write(f"Rejected (Low Contrast): ({int(x+w/2)}, {int(y+h/2)}), Intensity: {mean_intensity:.1f}\n")
                continue
            # BƯỚC 1: TÌM 3 GÓC LỚN (COARSE SEARCH). Hạ ngưỡng diện tích xuống 0.00008 để bắt các góc vẽ tay bị nhỏ.
            if 0.00008 < area_ratio < 0.005 and 0.5 <= aspect_ratio <= 2.5 and extent > 0.45:
                # Dùng Moments (centroid) thay vì tâm BoundingRect để tránh lệch khi chấm đen bị biến dạng phối cảnh
                M_contour = cv2.moments(c)
                if M_contour["m00"] > 0:
                    cX = M_contour["m10"] / M_contour["m00"]
                    cY = M_contour["m01"] / M_contour["m00"]
                else:
                    cX, cY = float(x + w/2), float(y + h/2)
                centers.append((cX, cY, contour_area))
                f.write(f"Valid Center added: ({cX:.1f}, {cY:.1f}), Area: {contour_area}, Ratio: {area_ratio:.5f}, Extent: {extent:.2f}, Intensity: {mean_intensity:.1f}\n")
            elif contour_area > 100:
                f.write(f"Rejected (Size/Extent): ({int(x+w/2)}, {int(y+h/2)}), Area: {contour_area}, Ratio: {area_ratio:.5f}, Extent: {extent:.2f}\n")
                
        f.write(f"\nTotal Valid Centers: {len(centers)}\n")
        
    debug_img = image.copy()
    for (cX, cY, _) in centers:
        cv2.circle(debug_img, (int(round(cX)), int(round(cY))), 10, (255, 0, 0), -1)
        
    if len(centers) < 3:
        cv2.putText(debug_img, "FAILED: Less than 3 centers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imwrite(log_jpg, debug_img)
        return image, "Lỗi: Không nhận diện đủ 3 góc định vị (Ảnh mờ, mất góc hoặc quá lệch)"
        
    filtered_centers = find_best_geometric_quad(centers, image, log_txt, debug_dir, debug_prefix, use_ai=use_ai, debug_img=debug_img)
    
    if len(filtered_centers) < 4:
        cv2.putText(debug_img, "FAILED: Could not find 4th corner", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imwrite(log_jpg, debug_img)
        return image, "Lỗi: Nội suy góc thứ 4 thất bại. Hãy chụp rõ góc dưới cùng bên phải."
    
    # Mảng 4 điểm và diện tích tương ứng
    pts_and_areas = [(c[0], c[1], c[2]) for c in filtered_centers]
    pts = np.array([[c[0], c[1]] for c in pts_and_areas], dtype="float32")
    rect = order_points(pts)
    
    # ---------------------------------------------------------
    # AUTO-ROTATION LOGIC DỰA VÀO GÓC ĐỊNH HƯỚNG (GÓC NHỎ NHẤT)
    # ---------------------------------------------------------
    # order_points trả về [TL, TR, BR, BL] trên không gian 2D của bức ảnh hiện tại.
    # Ta tìm diện tích (contour_area) của 4 điểm này để biết góc đánh dấu (nhỏ nhất) đang nằm ở đâu.
    ordered_areas = []
    for pt in rect:
        # Ánh xạ điểm trong rect về pts_and_areas (dùng khoảng cách gần nhất)
        dists = [np.hypot(pt[0] - p[0], pt[1] - p[1]) for p in pts_and_areas]
        min_idx_dist = np.argmin(dists)
        ordered_areas.append(pts_and_areas[min_idx_dist][2])
        
    min_area_idx = np.argmin(ordered_areas)
    
    # TỪ CHỐI CHẤM NẾU ẢNH BỊ LẬT HOẶC XOAY (DỰA VÀO VỊ TRÍ GÓC MARKER NHỎ NHẤT)
    if min_area_idx != 2:
        error_reasons = {
            0: "lật ngược (180 độ)",
            1: "xoay ngang",
            3: "xoay ngang"
        }
        
        # Nếu góc nhỏ nhất nằm ở vị trí xoay ngang (1 hoặc 3)
        # Bổ sung Heuristic: Với giấy vẽ tay, học sinh có thể vẽ góc BL nhỏ hơn cả góc BR.
        # Ta kiểm tra tỷ lệ hình học: Nếu tứ giác rõ ràng là hình chữ nhật đứng (Height > Width), 
        # thì chắc chắn nó không bị xoay ngang. Ta bỏ qua lỗi này!
        w_top = np.hypot(rect[0][0] - rect[1][0], rect[0][1] - rect[1][1])
        h_left = np.hypot(rect[0][0] - rect[3][0], rect[0][1] - rect[3][1])
        
        is_portrait = h_left > w_top * 1.1
        
        if (min_area_idx in [1, 3] and is_portrait):
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write("[ORIENTATION CHECK] Cảnh báo: Diện tích góc nhỏ nhất sai vị trí, nhưng hình dáng tứ giác là dọc (Portrait). Bỏ qua lỗi xoay ngang do giấy vẽ tay.\n")
        else:
            reason = error_reasons.get(min_area_idx, "sai chiều")
            cv2.putText(debug_img, f"FAILED: Anh bi {reason}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imwrite(log_jpg, debug_img)
            return image, f"Lỗi: Ảnh bị {reason}. Vui lòng xoay đúng chiều trước khi đưa vào hệ thống."
        
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"\n[ORIENTATION CHECK] Areas: TL={ordered_areas[0]}, TR={ordered_areas[1]}, BR={ordered_areas[2]}, BL={ordered_areas[3]}\n")
        f.write("[ORIENTATION CHECK] Ảnh đúng chiều (Marker nhỏ nhất nằm đúng ở góc Bottom-Right).\n\n")

    # Chữa lành góc nếu cần
    rect = fix_corners(rect, gray, debug_dir, debug_prefix)
    
    (tl, tr, br, bl) = rect
    
    # FINAL GEOMETRY CHECK: Đảm bảo tứ giác sau cùng thực sự là một hình chữ nhật (không bị lệch góc nghiêm trọng)
    final_angles = [
        angle_between(bl, tl, tr),
        angle_between(tl, tr, br),
        angle_between(tr, br, bl),
        angle_between(br, bl, tl)
    ]
    
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"Final Quad Angles: {np.round(final_angles, 1)}\n")
        
    if min(final_angles) < 70 or max(final_angles) > 110:
        cv2.putText(debug_img, "FAILED: Not a valid rectangle", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imwrite(log_jpg, debug_img)
        return image, f"Lỗi: Không nhận diện được 4 góc chính xác (Góc quá méo: {np.round(final_angles, 1)})"
        
    cv2.circle(debug_img, tuple(tl.astype(int)), 20, (0, 255, 0), -1)
    cv2.circle(debug_img, tuple(tr.astype(int)), 20, (0, 255, 0), -1)
    cv2.circle(debug_img, tuple(br.astype(int)), 20, (0, 255, 0), -1)
    cv2.circle(debug_img, tuple(bl.astype(int)), 20, (0, 255, 0), -1)
    
    # Vẽ các đường thẳng nối 4 góc để dễ quan sát (Hình bình hành/Tứ giác)
    pts_to_draw = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(debug_img, [pts_to_draw], isClosed=True, color=(0, 255, 0), thickness=4)
    
    w_rect = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    h_rect = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"Selected 4 corners: TL={tl}, TR={tr}, BR={br}, BL={bl}\n")
        f.write(f"Rect Width: {w_rect:.2f} (Required: {width * 0.4:.2f})\n")
        f.write(f"Rect Height: {h_rect:.2f} (Required: {height * 0.4:.2f})\n")
    
    if w_rect < width * 0.4 or h_rect < height * 0.4:
        cv2.putText(debug_img, "FAILED: Rect too small", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imwrite(log_jpg, debug_img)
        return image, "Lỗi: Khung định vị quá nhỏ, không hợp lệ"
        
    cv2.putText(debug_img, "SUCCESS: Aligned", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imwrite(log_jpg, debug_img)
    
    maxWidth = int(max(w_rect, np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))))
    maxHeight = int(max(h_rect, np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))))
    
    if maxWidth < width * 0.4 or maxHeight < height * 0.4:
        return image, "Lỗi: Kích thước khung nắn thẳng quá nhỏ"
        
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped, error_msg
