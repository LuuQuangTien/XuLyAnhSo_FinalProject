"""
Module trích xuất các khối (Block Extractor).
Tìm các vùng chữ nhật (ROI) trên ảnh nhị phân: khối đáp án, SBD, Mã đề.
"""
import cv2
import numpy as np
import os


def find_blocks(thresh: np.ndarray, min_area=15000, max_aspect_ratio=2.5, debug_dir=None, debug_prefix="", original_img=None) -> tuple[list, str]:
    """Tìm các khối chữ nhật (ROI) trên ảnh nhị phân. Trả về (danh sách khối, thông báo lỗi)."""
    # Dùng morphology close để nối liền các nét đứt của viền bảng
    kernel = np.ones((5, 5), np.uint8)
    closed_thresh = cv2.morphologyEx(thresh.copy(), cv2.MORPH_CLOSE, kernel)
    
    # --- PROJECTION PROFILE CUTTING BẰNG W_STD (Cắt hình học tự động tỷ lệ) ---
    debug_gaps_img = None
    if debug_dir and os.path.exists(debug_dir) and original_img is not None:
        debug_gaps_img = original_img.copy()
        
    new_closed_thresh = closed_thresh.copy()
    
    for _ in range(5):
        cnts_tmp, _ = cv2.findContours(new_closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_widths = []
        for c in cnts_tmp:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > min_area and float(w)/h < max_aspect_ratio:
                valid_widths.append(w)
                
        if not valid_widths:
            break
            
        W_std = np.median(valid_widths)
        cut_happened = False
        
        for c in cnts_tmp:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > min_area and float(w)/h < max_aspect_ratio:
                if w > 1.35 * W_std:
                    num_blocks = max(1, round(w / W_std))
                    roi = new_closed_thresh[y:y+h, x:x+w]
                    proj = np.sum(roi, axis=0) / 255.0
                    
                    for k in range(1, int(num_blocks) + (1 if num_blocks==1 else 0)):
                        expected_x = int(k * W_std)
                        if expected_x >= w: break
                        
                        search_start = max(0, expected_x - 40)
                        search_end = min(w, expected_x + 40)
                        
                        if search_end > search_start:
                            search_region = proj[search_start:search_end]
                            valley_min = np.min(search_region)
                            valley_indices = np.where(search_region <= valley_min + 10)[0]
                            
                            if len(valley_indices) > 0:
                                first_idx = search_start + valley_indices[0]
                                last_idx = search_start + valley_indices[-1]
                                
                                cv2.rectangle(new_closed_thresh, (x + first_idx, y), (x + last_idx, y + h), 0, -1)
                                
                                if debug_gaps_img is not None:
                                    cv2.rectangle(debug_gaps_img, (x + first_idx, y), (x + last_idx, y + h), (0, 255, 255), -1)
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
        if w * h > min_area and aspect_ratio < max_aspect_ratio: 
            blocks.append((x, y, w, h))
            
    if debug_dir and os.path.exists(debug_dir) and original_img is not None:
        debug_img = original_img.copy()
        with open(os.path.join(debug_dir, f"{debug_prefix}blocks_debug.txt"), "w", encoding="utf-8") as f:
            f.write(f"Total contours found in closed_thresh: {len(cnts)}\n")
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h if h != 0 else 0
                area = w * h
                if area > min_area:
                    if aspect_ratio < max_aspect_ratio:
                        f.write(f"ACCEPTED: Area={area}, Aspect={aspect_ratio:.2f}, Box=({x},{y},{w},{h})\n")
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(debug_img, f"OK {aspect_ratio:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        f.write(f"REJECTED (Aspect > {max_aspect_ratio}): Area={area}, Aspect={aspect_ratio:.2f}, Box=({x},{y},{w},{h})\n")
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(debug_img, f"REJ {aspect_ratio:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}debug_blocks.jpg"), debug_img)
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}debug_closed_thresh.jpg"), closed_thresh)
        if debug_gaps_img is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}debug_gaps.jpg"), debug_gaps_img)
            
    error_msg = ""
    if len(blocks) == 0:
        error_msg = "Lỗi: Trích xuất khối thất bại (Ảnh quá nhiễu hoặc sai form)"
    return blocks, error_msg
