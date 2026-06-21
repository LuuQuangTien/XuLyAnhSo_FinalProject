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
from image_processing.alignment.geometric_search import _get_debug_paths, deduce_4th_corner_from_3, fix_corners, find_best_geometric_quad

def clear_ai_cache():
    try:
        cache_files = glob.glob(os.path.join(tempfile.gettempdir(), "omr_ai_cache_*.*"))
        for f in cache_files:
            os.remove(f)
    except:
        pass

def align_document(image, debug_dir=None, debug_prefix="", method="four_corners"):
    """
    Tìm 4 điểm đen định vị ở 4 góc và nắn thẳng hình ảnh (Perspective Transform).
    Returns: (aligned_image, error_msg)
    """
    error_msg = ""
    h_orig, w_orig = image.shape[:2]
    if w_orig > h_orig:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        error_msg = "Ảnh nằm ngang, đã tự động xoay dọc."
    log_txt, log_jpg = _get_debug_paths(debug_dir, debug_prefix)
    
    image = crop_paper_from_background(image, log_txt)
    
    gray = to_grayscale(image)
    height, width = image.shape[:2]
    image_area = width * height
    
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
    
    block_size = int(width / 20)
    if block_size % 2 == 0: block_size += 1
    if block_size < 31: block_size = 31
        
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size, 15
    )
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"Image Size: {width} x {height}, Area: {image_area}\n")
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
        
    debug_img = image.copy()
    for (cX, cY, _) in centers:
        cv2.circle(debug_img, (cX, cY), 10, (255, 0, 0), -1)
        
    if method == "three_corners":
        if len(centers) < 3:
            cv2.putText(debug_img, "FAILED: Less than 3 centers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imwrite(log_jpg, debug_img)
            return image, "Lỗi: Không nhận diện đủ 3 góc định vị (Ảnh mờ, mất góc hoặc quá lệch)"
            
        # SỬ DỤNG AI SILUETA ĐỂ LỌC NHIỄU CHO MẪU 3 GÓC
        bbox = get_ai_paper_bounding_box(image, log_txt, debug_prefix)
        if bbox is not None:
            bx, by, bw, bh = bbox
            
            # Xuất ảnh debug AI Box cho chế độ 3 góc
            ai_debug_img = image.copy()
            cv2.rectangle(ai_debug_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 4)
            ai_debug_path = log_txt.replace("align_debug.txt", "ai_box.jpg")
            cv2.imwrite(ai_debug_path, ai_debug_img)
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"[3-CORNERS] Đã xuất ảnh debug AI Box ra: {ai_debug_path}\n")
                
            filtered_centers = []
            pad = 50 # Nới rộng bbox một chút để an toàn
            for c in centers:
                cx, cy, area = c
                if bx - pad <= cx <= bx + bw + pad and by - pad <= cy <= by + bh + pad:
                    filtered_centers.append(c)
            
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"[3-CORNERS] AI lọc viền: giữ lại {len(filtered_centers)}/{len(centers)} chấm đen.\n")
                
            if len(filtered_centers) >= 3:
                centers = filtered_centers
            else:
                with open(log_txt, "a", encoding="utf-8") as f:
                    f.write(f"[3-CORNERS] AI lọc xong chỉ còn {len(filtered_centers)} điểm, bỏ qua AI filter để dùng Fallback.\n")
                    
        from image_processing.alignment.geometric_search import find_best_geometric_triplet
        
        # Pass bbox if AI was successful, otherwise None
        top_3 = find_best_geometric_triplet(centers, log_txt, bbox if 'bbox' in locals() else None)
        pts = deduce_4th_corner_from_3(top_3, log_txt)
        pts = np.array(pts, dtype="float32")
        rect = order_points(pts)
    else:
        if len(centers) < 4:
            cv2.putText(debug_img, "FAILED: Less than 4 centers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imwrite(log_jpg, debug_img)
            return image, "Lỗi: Không nhận diện đủ 4 góc định vị (Ảnh mờ, mất góc hoặc quá lệch)"
            
        filtered_centers = find_best_geometric_quad(centers, image, log_txt, debug_dir, debug_prefix)
        pts = np.array([[c[0], c[1]] for c in filtered_centers])
        rect = order_points(pts)
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
