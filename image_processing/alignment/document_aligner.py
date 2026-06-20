"""
Module căn chỉnh tài liệu (Document Alignment).
Chứa toàn bộ logic tìm 4 điểm neo, bẻ phẳng ảnh (Perspective Transform),
cắt giấy khỏi nền và chữa lành góc bị sai.
"""
import cv2
import numpy as np
import os
import tempfile
import glob
import json

import itertools
from image_processing.utils.image_utils import to_grayscale
from image_processing.utils.geometry_utils import order_points, angle_between


def _get_debug_paths(debug_dir, debug_prefix):
    if debug_dir and os.path.exists(debug_dir):
        return (os.path.join(debug_dir, f"{debug_prefix}align_debug.txt"),
                os.path.join(debug_dir, f"{debug_prefix}debug_align.jpg"))
    import tempfile
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


def clear_ai_cache():
    try:
        cache_files = glob.glob(os.path.join(tempfile.gettempdir(), "omr_ai_cache_*.*"))
        for f in cache_files:
            os.remove(f)
    except:
        pass


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
        
    model_path = os.path.join(os.getcwd(), "assets", "models", "silueta.onnx")
    if os.path.exists(model_path):
        try:
            bx, by, bw, bh = None, None, None, None
            cache_file = ""
            
            # KIỂM TRA CACHE TRƯỚC KHI CHẠY AI
            if debug_prefix:
                cache_file = os.path.join(tempfile.gettempdir(), f"omr_ai_cache_{debug_prefix}.json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, "r") as fc:
                            data = json.load(fc)
                        bx, by, bw, bh = data['bx'], data['by'], data['bw'], data['bh']
                        with open(log_txt, "a", encoding="utf-8") as f:
                            f.write(f"[GEOMETRIC SEARCH] Đã sử dụng Bounding Box từ CACHE: {bx, by, bw, bh}\n")
                    except: pass
                    
            # NẾU KHÔNG CÓ TRONG CACHE -> CHẠY AI
            if bx is None:
                import onnxruntime as ort
                
                # Cấu hình để tắt các cảnh báo không cần thiết từ onnxruntime
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3
                
                # Tải model vào onnxruntime
                net = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
                input_name = net.get_inputs()[0].name
                
                # Đưa ảnh vào mạng AI (Sử dụng OpenCV blobFromImage để tiền xử lý dễ dàng)
                blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (320, 320), (0.485*255, 0.456*255, 0.406*255), swapRB=True, crop=False)
                
                # Chạy Inference bằng ONNX Runtime
                out = net.run(None, {input_name: blob})[0]
                
                mask = out[0, 0, :, :]
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Lấy Bounding Box của vật thể lớn nhất
                _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if cnts:
                    best_cnt = max(cnts, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(best_cnt)
                    
                    # LƯU VÀO CACHE CHO LẦN SAU
                    if cache_file:
                        try:
                            with open(cache_file, "w") as fc:
                                json.dump({'bx': bx, 'by': by, 'bw': bw, 'bh': bh}, fc)
                        except: pass
            
            if bx is not None:
                with open(log_txt, "a", encoding="utf-8") as f:
                    f.write(f" - AI Bounding Box: x={bx}, y={by}, w={bw}, h={bh}\n")
                
                best_ai_score = -999999
                best_quad = None
                
                # Chấm điểm lại từng ứng cử viên
                for cand in top_candidates:
                    pts = cand['pts']
                    ai_score = 0
                    
                    # Mỗi điểm góc (nút) trong tứ giác
                    for pt in pts:
                        px, py = pt[0], pt[1]
                        
                        # Khoảng cách từ điểm tới Box của AI (Nếu nằm trong box, khoảng cách = 0)
                        dx = max(0, bx - px, px - (bx + bw))
                        dy = max(0, by - py, py - (by + bh))
                        dist = np.hypot(dx, dy)
                        
                        # Thưởng 1000 điểm nếu nằm lọt thỏm trong tờ giấy
                        if dist == 0:
                            ai_score += 1000
                        else:
                            # Trừ điểm càng nặng nếu càng cách xa tờ giấy
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
                        import tempfile
                        ai_debug_path = os.path.join(tempfile.gettempdir(), f"{debug_prefix}ai_box.jpg")
                    cv2.imwrite(ai_debug_path, debug_img)
                    
                    with open(log_txt, "a", encoding="utf-8") as f:
                        f.write(f"[GEOMETRIC SEARCH] Đã xuất ảnh debug AI Box ra: {ai_debug_path}\n")
                        
                return best_quad
                
        except Exception as e:
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"[GEOMETRIC SEARCH] Lỗi khi chạy AI Silueta: {e}. Fallback dùng tổ hợp đầu tiên.\n")
    else:
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"[GEOMETRIC SEARCH] Không tìm thấy silueta.onnx! Fallback dùng tổ hợp đầu tiên.\n")
            
    # Fallback nếu AI lỗi hoặc không có file
    return top_candidates[0]['quad']


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
            rect[3] = hull_pts[np.argmax(diff)] # Trái-Dưới (BL)
            
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
            
        centers.sort(key=lambda x: x[2], reverse=True)
        top_3 = centers[:3]
        pts = deduce_4th_corner_from_3(top_3, log_txt)
        pts = np.array(pts)
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
