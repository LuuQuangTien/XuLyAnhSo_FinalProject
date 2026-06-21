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

def deduce_4th_corner_from_3(pts, log_txt):
    pts_f = [np.array(p[:2], dtype=float) for p in pts]
    
    d01 = np.sum((pts_f[0] - pts_f[1])**2)
    d12 = np.sum((pts_f[1] - pts_f[2])**2)
    d20 = np.sum((pts_f[2] - pts_f[0])**2)
    
    dists = [d12, d20, d01] 
    right_idx = np.argmax(dists)
    
    p0 = pts_f[right_idx]
    others = [pts_f[i] for i in range(3) if i != right_idx]
    p1, p2 = others[0], others[1]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    
    if cross > 0:
        tr, bl = p1, p2
    else:
        tr, bl = p2, p1
        
    tl = p0
    br = bl + (tr - tl)
    
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"[3-CORNERS] TL=({int(tl[0])},{int(tl[1])}), TR=({int(tr[0])},{int(tr[1])}), BL=({int(bl[0])},{int(bl[1])})\n")
        f.write(f"-> Deduced BR=({int(br[0])},{int(br[1])})\n")
        
    return [[int(tl[0]), int(tl[1])], [int(tr[0]), int(tr[1])], [int(br[0]), int(br[1])], [int(bl[0]), int(bl[1])]]

def find_best_geometric_triplet(centers, log_txt, bbox=None):
    """
    Tìm tổ hợp 3 điểm (từ top 15) tạo thành một góc vuông (~90 độ) tốt nhất.
    Sử dụng AI Bounding Box (nếu có) để phạt các điểm nằm xa 4 góc của tờ giấy.
    """
    import itertools
    top_n = sorted(centers, key=lambda x: x[2], reverse=True)[:15]
    valid_triplets = []
    
    ai_corners = None
    if bbox is not None:
        bx, by, bw, bh = bbox
        ai_corners = [
            np.array([bx, by]),              # TL
            np.array([bx + bw, by]),         # TR
            np.array([bx + bw, by + bh]),    # BR
            np.array([bx, by + bh])          # BL
        ]
    
    for triplet in itertools.combinations(top_n, 3):
        pts_f = [np.array(p[:2], dtype=float) for p in triplet]
        d01 = np.sum((pts_f[0] - pts_f[1])**2)
        d12 = np.sum((pts_f[1] - pts_f[2])**2)
        d20 = np.sum((pts_f[2] - pts_f[0])**2)
        dists = [d12, d20, d01] 
        right_idx = np.argmax(dists)
        
        p0 = pts_f[right_idx]
        others = [pts_f[i] for i in range(3) if i != right_idx]
        p1, p2 = others[0], others[1]
        
        v1 = p1 - p0
        v2 = p2 - p0
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: continue
        
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        if 75 <= angle <= 105:
            # Score base: sum of areas
            score = triplet[0][2] + triplet[1][2] + triplet[2][2]
            
            # Penalize by distance to AI corners
            if ai_corners is not None:
                total_dist = 0
                for pt in pts_f:
                    # Khoảng cách tới góc AI gần nhất
                    min_dist = min(np.linalg.norm(pt - c) for c in ai_corners)
                    total_dist += min_dist
                # Trừ điểm càng nặng nếu càng xa các góc AI
                score -= total_dist * 5.0
            
            # Penalize bad aspect ratio (A4 is ~1.414)
            ratio = max(norm_v1, norm_v2) / (min(norm_v1, norm_v2) + 1e-6)
            if ratio > 2.0 or ratio < 1.0:
                score -= 5000 * abs(ratio - 1.414)
                
            valid_triplets.append((triplet, score, angle))
            
    if valid_triplets:
        valid_triplets.sort(key=lambda x: x[1], reverse=True)
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[3-CORNERS] Tìm thấy {len(valid_triplets)} tổ hợp góc vuông. Chọn tổ hợp tốt nhất (Góc: {valid_triplets[0][2]:.1f}, Score: {valid_triplets[0][1]:.1f}).\n")
        return valid_triplets[0][0]
    
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"[3-CORNERS] CẢNH BÁO: Không tìm thấy 3 điểm nào tạo thành góc vuông hợp lệ. Fallback về 3 điểm lớn nhất.\n")
    return top_n[:3]


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

def find_best_geometric_quad(candidates, image, log_txt, debug_dir=None, debug_prefix=""):
    """
    Thuật toán Hình Học Tổ Hợp: 
    - Lấy top 5 điểm gần 4 góc của ảnh nhất.
    - Duyệt qua mọi tổ hợp 4 góc để chọn ra tứ giác vuông vức nhất và có kích thước góc đều nhất.
    - Nếu có sự tranh chấp (nhiều tổ hợp có điểm tương đương), gọi AI U2Net để tìm Bounding Box và phân xử.
    """
    if len(candidates) < 4:
        return candidates
        
    height, width = image.shape[:2]
        
    tl_pool = sorted(candidates, key=lambda c: np.hypot(c[0] - 0, c[1] - 0))[:5]
    tr_pool = sorted(candidates, key=lambda c: np.hypot(c[0] - width, c[1] - 0))[:5]
    bl_pool = sorted(candidates, key=lambda c: np.hypot(c[0] - 0, c[1] - height))[:5]
    br_pool = sorted(candidates, key=lambda c: np.hypot(c[0] - width, c[1] - height))[:5]
    
    valid_quads = []
    
    for p_tl, p_tr, p_bl, p_br in itertools.product(tl_pool, tr_pool, bl_pool, br_pool):
        points_set = set([ (p[0], p[1]) for p in [p_tl, p_tr, p_bl, p_br] ])
        if len(points_set) < 4:
            continue
            
        pts = np.array([p_tl[:2], p_tr[:2], p_br[:2], p_bl[:2]], dtype=float)
        
        angles = [
            angle_between(pts[3], pts[0], pts[1]), # TL
            angle_between(pts[0], pts[1], pts[2]), # TR
            angle_between(pts[1], pts[2], pts[3]), # BR
            angle_between(pts[2], pts[3], pts[0])  # BL
        ]
        
        if min(angles) < 70 or max(angles) > 110:
            continue
            
        # Diện tích của tứ giác tạo bởi 4 điểm này (Càng lớn tức là càng bung ra ngoài mép)
        quad_area = cv2.contourArea(np.array(pts, dtype=np.float32))
        if quad_area < (width * height) * 0.1:
            continue
            
        # Kiểm tra độ đồng đều diện tích của 4 CHẤM ĐEN
        areas = [p_tl[2], p_tr[2], p_br[2], p_bl[2]]
        max_a, min_a = max(areas), min(areas)
        area_ratio = max_a / (min_a + 1e-6)
        
        # Nếu các chấm đen chênh lệch diện tích quá 5 lần -> Chắc chắn có điểm rác
        if area_ratio > 5.0:
            continue
            
        # Tính điểm phạt
        angle_diff = sum(abs(a - 90) for a in angles) / 4.0
        
        # Hệ số điểm: Bắt đầu từ 1.0 (100% quad_area)
        penalty_factor = 1.0 - (angle_diff * 0.02) - (area_ratio * 0.05)
        
        if penalty_factor <= 0:
            continue
            
        total_score = quad_area * penalty_factor
        valid_quads.append({
            'quad': [p_tl, p_tr, p_br, p_bl],
            'pts': pts,
            'score': total_score
        })
        
    if not valid_quads:
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write("[GEOMETRIC SEARCH] Thất bại! Không có tổ hợp nào tạo thành hình chữ nhật hợp lệ.\n")
        return candidates
        
    # Sắp xếp theo điểm giảm dần
    valid_quads.sort(key=lambda x: x['score'], reverse=True)
    
    # Lấy Top Candidates (Nhữnng ứng viên có điểm >= 70% so với ứng viên top 1)
    max_score = valid_quads[0]['score']
    top_candidates = [q for q in valid_quads if q['score'] >= max_score * 0.7]
    
    if len(top_candidates) == 1:
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[GEOMETRIC SEARCH] Tìm thấy 1 tổ hợp xuất sắc nhất với điểm: {max_score:.2f}\n")
        return top_candidates[0]['quad']
        
    # CÓ SỰ TRANH CHẤP! KÍCH HOẠT TRỌNG TÀI AI
    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(f"[GEOMETRIC SEARCH] TRANH CHẤP: Có {len(top_candidates)} ứng cử viên xuất sắc. Gọi AI Silueta ra phân xử!\n")
        
    bbox = get_ai_paper_bounding_box(image, log_txt, debug_prefix)
    
    if bbox is not None:
        bx, by, bw, bh = bbox
        best_ai_score = -999999
        best_quad = None
        
        # Chấm điểm lại từng ứng cử viên
        for cand in top_candidates:
            pts = cand['pts']
            ai_score = 0
            
            # Lấy 4 góc của AI Bounding Box
            ai_corners = [
                np.array([bx, by]),              # TL
                np.array([bx + bw, by]),         # TR
                np.array([bx + bw, by + bh]),    # BR
                np.array([bx, by + bh])          # BL
            ]
            
            # Mỗi điểm góc (nút) trong tứ giác
            for i, pt in enumerate(pts):
                px, py = pt[0], pt[1]
                
                # Tính khoảng cách từ điểm tới góc tương ứng của AI Box
                dist = np.linalg.norm(np.array([px, py]) - ai_corners[i])
                
                # Trừ điểm càng nặng nếu góc của tứ giác càng xa góc của tờ giấy
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
