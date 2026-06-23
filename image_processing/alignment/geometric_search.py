import cv2
import numpy as np
import itertools
import tempfile
import os

from image_processing.utils.geometry_utils import angle_between
from image_processing.alignment.ai_helper import get_ai_paper_bounding_box

def _get_debug_paths(debug_dir, debug_prefix):
    if debug_dir and os.path.exists(debug_dir):
        return (os.path.join(debug_dir, f"{debug_prefix}align_debug.txt"),
                os.path.join(debug_dir, f"{debug_prefix}debug_align.jpg"))
    temp_dir = tempfile.gettempdir()
    return (os.path.join(temp_dir, f"{debug_prefix}align_debug.txt"),
            os.path.join(temp_dir, f"{debug_prefix}debug_align.jpg"))

def _point_to_line_dist(point, line_p1, line_p2):
    """Khoảng cách từ điểm đến đường thẳng (mở rộng vô hạn, không phải đoạn thẳng)."""
    d = line_p2 - line_p1
    n = np.array([-d[1], d[0]])
    norm_n = np.linalg.norm(n)
    if norm_n == 0:
        return float('inf')
    return abs(np.dot(point - line_p1, n)) / norm_n

def _line_angle_diff(a1, a2):
    """Chênh lệch góc giữa 2 đường thẳng (0-90°, đường thẳng không phân biệt hướng)."""
    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff)

def _line_intersection(p1, p2, p3, p4):
    """Tìm giao điểm của 2 đường thẳng (mở rộng vô hạn). Trả về None nếu song song."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-10:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


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
        
        if min(angles) > 75 and max(angles) < 105:
            return pts
            
        bad_idx = int(np.argmax(np.abs(np.array(angles) - 90)))
        
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"\n[GEOMETRIC HEALING] Attempt {attempt+1}. Bad quad detected. Angles: {np.round(angles, 1)}\n")
            f.write(f"Bad corner index: {bad_idx} (Angle: {angles[bad_idx]:.1f})\n")
            
        prev_idx = (bad_idx - 1) % 4
        next_idx = (bad_idx + 1) % 4
        opp_idx  = (bad_idx + 2) % 4
        
        P_prev = pts[prev_idx]
        P_next = pts[next_idx]
        P_opp  = pts[opp_idx]
        
        P_pred = P_prev + P_next - P_opp
        
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"Predicted coordinate: {P_pred}\n")
            
        h, w = gray.shape
        r = 250
        sx = int(max(0, P_pred[0] - r))
        ex = int(min(w, P_pred[0] + r))
        sy = int(max(0, P_pred[1] - r))
        ey = int(min(h, P_pred[1] + r))
        
        roi = gray[sy:ey, sx:ex]
        if roi.size == 0: return pts
        
        _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi_thresh = cv2.medianBlur(roi_thresh, 3)
        
        cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_pt = None
        min_dist = float('inf')
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50: continue
            
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
                f.write(f"Healing attempt found contour! Replaced with: {best_pt} (Dist: {min_dist:.1f})\n")
        else:
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write("Healing failed! No suitable contour found in ROI.\n")
            break
            
    return pts

def find_best_geometric_quad(candidates, image, log_txt, debug_dir=None, debug_prefix="", use_ai=False, debug_img=None):
    """
    Thuật toán Hình Học Tổ Hợp: 
    - Lấy top 5 điểm gần 4 góc của ảnh nhất.
    - Duyệt qua mọi tổ hợp 4 góc để chọn ra tứ giác vuông vức nhất và có kích thước góc đều nhất.
    - Nếu có sự tranh chấp (nhiều tổ hợp có điểm tương đương), gọi AI U2Net để tìm Bounding Box và phân xử.
    """
    if len(candidates) < 3:
        return candidates
        
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
    tl_pool = sorted(candidates, key=lambda c: np.hypot(c[0] - 0, c[1] - 0))[:5]
    tr_pool = sorted(candidates, key=lambda c: np.hypot(c[0] - width, c[1] - 0))[:5]
    bl_pool = sorted(candidates, key=lambda c: np.hypot(c[0] - 0, c[1] - height))[:5]
    
    def calculate_quad_score(quad, relaxed=False):
        p_tl, p_tr, p_br, p_bl = quad
            
        pts = np.array([p_tl[:2], p_tr[:2], p_br[:2], p_bl[:2]], dtype=float)
        
        angles = [
            angle_between(pts[3], pts[0], pts[1]), # TL
            angle_between(pts[0], pts[1], pts[2]), # TR
            angle_between(pts[1], pts[2], pts[3]), # BR
            angle_between(pts[2], pts[3], pts[0])  # BL
        ]
        
        if min(angles) < 70 or max(angles) > 110:
            return -1, pts, 0
            
        # Diện tích của tứ giác tạo bởi 4 điểm này (Càng lớn tức là càng bung ra ngoài mép)
        quad_area = cv2.contourArea(np.array(pts, dtype=np.float32))
        if quad_area < (width * height) * 0.1:
            return -1, pts, 0
            
        # Kiểm tra độ đồng đều diện tích của 4 CHẤM ĐEN
        areas = sorted([p_tl[2], p_tr[2], p_br[2], p_bl[2]])
        large_areas_ratio = areas[3] / (areas[1] + 1e-6)
        
        # Ngưỡng tỷ lệ diện tích
        max_ratio = 6.0 if relaxed else 3.0
        if large_areas_ratio > max_ratio:
            return -1, pts, 0
            
        # Tính điểm phạt
        angle_diff = sum(abs(a - 90) for a in angles) / 4.0
        h_rect = max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1]))
        w_rect = max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))
        aspect_ratio = h_rect / (w_rect + 1e-6)
        aspect_penalty_factor = 0.3 if relaxed else 0.5
        aspect_penalty = abs(aspect_ratio - 1.414) * aspect_penalty_factor
        angle_penalty_factor = 0.03 if relaxed else 0.05
        area_penalty_factor = 0.02 if relaxed else 0.05
        penalty_factor = 1.0 - (angle_diff * angle_penalty_factor) - (large_areas_ratio * area_penalty_factor) - aspect_penalty
        if penalty_factor <= 0:
            return -1, pts, 0
        return quad_area * penalty_factor, pts, penalty_factor

    quad_candidates = []
    if len(candidates) >= 4:
        br_pool = sorted(candidates, key=lambda c: np.hypot(c[0] - width, c[1] - height))[:5]
        for p_tl, p_tr, p_bl, p_br in itertools.product(tl_pool, tr_pool, bl_pool, br_pool):
            if len(set([ (p[0], p[1]) for p in [p_tl, p_tr, p_bl, p_br] ])) == 4:
                quad_candidates.append([p_tl, p_tr, p_br, p_bl])
                
    for p_tl, p_tr, p_bl in itertools.product(tl_pool, tr_pool, bl_pool):
        if len(set([ (p[0], p[1]) for p in [p_tl, p_tr, p_bl] ])) < 3: continue
        pred_br_x = p_tr[0] + p_bl[0] - p_tl[0]
        pred_br_y = p_tr[1] + p_bl[1] - p_tl[1]
        roi_size = 150
        sx = int(max(0, pred_br_x - roi_size))
        ex = int(min(width, pred_br_x + roi_size))
        sy = int(max(0, pred_br_y - roi_size))
        ey = int(min(height, pred_br_y + roi_size))
        roi = gray[sy:ey, sx:ex]
        if roi.size == 0: continue
        _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_tiny = None
        min_dist = float('inf')
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 30 or area > 5000: continue
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h != 0 else 0
            extent = area / float(w * h)
            if 0.5 <= aspect_ratio <= 2.0 and extent > 0.5:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = sx + int(M["m10"] / M["m00"])
                    cy = sy + int(M["m01"] / M["m00"])
                    dist = np.hypot(cx - pred_br_x, cy - pred_br_y)
                    if dist < min_dist:
                        min_dist = dist
                        best_tiny = (cx, cy, area)
        if best_tiny:
            quad_candidates.append([p_tl, p_tr, best_tiny, p_bl])
        else:
            # Tự động nội suy góc thứ 4 để tạo thành hình bình hành
            quad_candidates.append([p_tl, p_tr, (pred_br_x, pred_br_y, 0), p_bl])

    valid_quads = []
    for quad in quad_candidates:
        score, pts, pf = calculate_quad_score(quad, relaxed=False)
        if score > 0:
            valid_quads.append({'quad': quad, 'pts': np.array(pts, dtype=np.float32), 'score': score, 'pf': pf})
            
    if len(valid_quads) > 0:
        valid_quads.sort(key=lambda x: x['score'], reverse=True)
        best = valid_quads[0]
        best_quad_area = cv2.contourArea(best['pts'])
        
        # Fast Search chỉ thành công nếu tứ giác đạt độ tin cậy hình học cao (không bị méo mó quá nhiều)
        # Nếu penalty_factor < 0.75, có thể là nó đang bắt nhầm bong bóng bên trong do ảnh bị chụp nghiêng (Perspective Zoom)
        if best_quad_area > (width * height) * 0.2 and best['pf'] > 0.75:
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"[GEOMETRIC SEARCH] Tìm thấy bằng FAST SEARCH: Score={best['score']:.0f}, PenaltyFactor={best['pf']:.2f}\n")
            return best['quad']

    with open(log_txt, "a", encoding="utf-8") as f:
        f.write("[GEOMETRIC SEARCH] Fast Search thất bại hoặc bị bóp méo hình học. Kích hoạt ROBUST SEARCH...\n")
        
    quad_candidates_robust = []
    for p_tl, p_tr, p_bl in itertools.product(tl_pool, tr_pool, bl_pool):
        if len(set([ (p[0], p[1]) for p in [p_tl, p_tr, p_bl] ])) < 3: continue
        
        pred_br_x = p_tr[0] + p_bl[0] - p_tl[0]
        pred_br_y = p_tr[1] + p_bl[1] - p_tl[1]
        
        sx = int(max(0, min(p_tr[0], p_bl[0]) - 150))
        sy = int(max(0, min(p_tr[1], p_bl[1]) - 150))
        ex = width
        ey = height
        if debug_img is not None:
            cv2.rectangle(debug_img, (sx, sy), (ex, ey), (0, 255, 255), 2)
        roi = gray[sy:ey, sx:ex]
        
        found_robust = False
        if roi.size > 0:
            _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 20 or area > 20000: continue
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h if h != 0 else 0
                extent = area / float(w * h)
                if 0.3 <= aspect_ratio <= 3.0 and extent > 0.3:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = sx + int(M["m10"] / M["m00"])
                        cy = sy + int(M["m01"] / M["m00"])
                        if np.hypot(cx - p_tl[0], cy - p_tl[1]) < 100: continue
                        if np.hypot(cx - p_tr[0], cy - p_tr[1]) < 100: continue
                        if np.hypot(cx - p_bl[0], cy - p_bl[1]) < 100: continue
                        quad_candidates_robust.append([p_tl, p_tr, (cx, cy, area), p_bl])
                        found_robust = True
                        
        if not found_robust:
            # Tự động nội suy góc thứ 4 cho ROBUST SEARCH
            quad_candidates_robust.append([p_tl, p_tr, (pred_br_x, pred_br_y, 0), p_bl])
                    
    valid_quads_robust = []
    for quad in quad_candidates_robust:
        score, pts, pf = calculate_quad_score(quad, relaxed=True)
        if score > 0:
            valid_quads_robust.append({'quad': quad, 'pts': np.array(pts, dtype=np.float32), 'score': score, 'pf': pf})
            
    if len(valid_quads_robust) == 0:
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write("[GEOMETRIC SEARCH] ROBUST SEARCH không tìm thấy góc phù hợp!\n")
        return []
        
    valid_quads_robust.sort(key=lambda x: x['score'], reverse=True)
    top_candidates = valid_quads_robust[:3]
    if len(top_candidates) == 1:
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[GEOMETRIC SEARCH] Tìm thấy bằng ROBUST SEARCH: {top_candidates[0]['score']:.2f}\n")
        return top_candidates[0]['quad']
        
    if not use_ai:
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[GEOMETRIC SEARCH] TRANH CHẤP: Có {len(top_candidates)} ứng cử viên nhưng AI bị tắt.\n")
        return top_candidates[0]['quad']

    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"[GEOMETRIC SEARCH] TRANH CHẤP: Có {len(top_candidates)} ứng cử viên. Gọi AI Silueta!\n")
        
    bbox = get_ai_paper_bounding_box(image, log_txt, debug_prefix)
    if bbox is not None:
        bx, by, bw, bh = bbox
        best_ai_score = -999999
        best_quad = None
        for cand in top_candidates:
            pts = cand['pts']
            ai_score = 0
            ai_corners = [np.array([bx, by]), np.array([bx + bw, by]), np.array([bx + bw, by + bh]), np.array([bx, by + bh])]
            for i, pt in enumerate(pts):
                dist = np.linalg.norm(np.array([pt[0], pt[1]]) - ai_corners[i])
                ai_score -= dist
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f" - Ứng cử viên {pts.tolist()}: BaseScore={cand['score']:.0f}, AI_Score={ai_score:.1f}\n")
            if ai_score > best_ai_score:
                best_ai_score = ai_score
                best_quad = cand['quad']
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[GEOMETRIC SEARCH] AI đã chọn xong tổ hợp có AI_Score cao nhất: {best_ai_score:.1f}\n")
            
        # LƯU ẢNH LOG CHO AI NẾU CÓ ẢNH GỐC
        if image is not None:
            debug_img = image.copy()
            # Vẽ Bounding Box của AI (Màu Đỏ, nét vẽ 4 để dễ nhìn)
            cv2.rectangle(debug_img, (bx, by), (bx+bw, by+bh), (0, 0, 255), 4)
            
            # Vẽ Tổ hợp góc tốt nhất AI đã chọn (Màu Xanh lá)
            if best_quad:
                pts_draw = np.array([pt[:2] for pt in best_quad], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug_img, [pts_draw], True, (0, 255, 0), 3)
                for pt in best_quad:
                    cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)
                    
            if debug_dir and os.path.exists(debug_dir):
                ai_debug_path = os.path.join(debug_dir, f"{debug_prefix}ai_box.jpg")
            else:
                ai_debug_path = os.path.join(tempfile.gettempdir(), f"{debug_prefix}ai_box.jpg")
            cv2.imwrite(ai_debug_path, debug_img)
            
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"[GEOMETRIC SEARCH] Đã xuất ảnh debug AI Box ra: {ai_debug_path}\n")
                
        return best_quad
    else:
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[GEOMETRIC SEARCH] Fallback dùng tổ hợp đầu tiên do AI trả về None.\n")
            
    # Fallback nếu AI lỗi hoặc không có file
    return top_candidates[0]['quad']
