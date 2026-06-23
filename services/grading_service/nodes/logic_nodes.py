import cv2
import numpy as np
from image_processing.extraction.bubble_analyzer import count_bubble_pixels, create_bubble_mask, count_projection_peak, evaluate_bubble_xor, get_bubble_centroid_vector

class Heuristic120Node:
    def execute(self, blocks, image_shape, **params):
        if not blocks or not image_shape:
            return {"sbd_block": None, "made_block": None, "question_blocks": blocks, "error": "Lỗi: Phân loại khối thất bại"}
            
        img_h, img_w = image_shape[:2]
        
        # Load parameters with defaults
        top_split_y = params.get("top_split_y", 0.45)
        bottom_split_y = params.get("bottom_split_y", 0.2)
        tall_block_ratio = params.get("tall_block_ratio", 2.5)
        right_split_x = params.get("right_split_x", 0.45)
        sbd_split_ratio = params.get("sbd_split_ratio", 0.65)
        
        fb_sbd = params.get("fallback_sbd", [0.535, 0.018, 0.270, 0.232])
        fb_made = params.get("fallback_made", [0.820, 0.018, 0.155, 0.232])
        fb_q_y = params.get("fallback_q_y", 0.270)
        fb_q_h = params.get("fallback_q_h", 0.708)
        fb_q_w = params.get("fallback_q_w", 0.218)
        fb_q_xs = params.get("fallback_q_xs", [0.024, 0.268, 0.512, 0.755])
        
        # 1. Lọc các khối bị đè lên nhau (loại khối to dị dạng, giữ khối nhỏ chuẩn)
        filtered_blocks = []
        blocks_sorted_by_area = sorted(blocks, key=lambda b: b[2]*b[3], reverse=False)
        for b in blocks_sorted_by_area:
            bx, by, bw, bh = b
            overlap = False
            for fb in filtered_blocks:
                fx, fy, fw, fh = fb
                ix = max(bx, fx)
                iy = max(by, fy)
                iw = min(bx+bw, fx+fw) - ix
                ih = min(by+bh, fy+fh) - iy
                if iw > 0 and ih > 0:
                    intersection = iw * ih
                    if intersection / min(bw*bh, fw*fh) > 0.3:
                        overlap = True
                        break
            if not overlap:
                filtered_blocks.append(b)
                
        # 2. Phân tách khối phần đầu (Top) và phần thân (Bottom)
        top_blocks = []
        bottom_blocks = []
        for b in filtered_blocks:
            bx, by, bw, bh = b
            cy = by + bh/2
            if cy < img_h * top_split_y:
                top_blocks.append(b)
            if cy > img_h * bottom_split_y: 
                bottom_blocks.append(b)
                
        # KIỂM TRA LỘN NGƯỢC 180 ĐỘ (Upside Down Detection)
        tall_top_blocks = [b for b in top_blocks if b[3] > b[2] * tall_block_ratio]
        if len(tall_top_blocks) >= 3:
            return {"sbd_block": None, "made_block": None, "question_blocks": blocks, "error": "Lỗi: Ảnh bị lộn ngược 180 độ. Yêu cầu chụp đúng chiều."}

        # 3. Lọc khối đầu (Chỉ lấy ở nửa bên phải - SBD & Mã đề)
        top_right_blocks = [b for b in top_blocks if (b[0] + b[2]/2) > img_w * right_split_x]
        top_right_blocks = sorted(top_right_blocks, key=lambda b: b[0]) # Trái sang phải
        
        # 4. Lọc khối thân (Chỉ giữ lại 4 khối to nhất là 4 cột trắc nghiệm)
        bottom_blocks = sorted(bottom_blocks, key=lambda b: b[2]*b[3], reverse=True)[:4]
        bottom_blocks = sorted(bottom_blocks, key=lambda b: b[0]) # Trái sang phải
        
        sbd_block = None
        made_block = None
        
        # Gán SBD và Mã đề
        if len(top_right_blocks) >= 2:
            sbd_block = top_right_blocks[0] # SBD nằm bên trái mã đề
            made_block = top_right_blocks[1]
        elif len(top_right_blocks) == 1:
            bx, by, bw, bh = top_right_blocks[0]
            # Nếu 2 khối bị dính chặt vào nhau, bổ đôi theo tỷ lệ cấu hình
            sbd_block = (bx, by, int(bw * sbd_split_ratio), bh)
            made_block = (bx + int(bw * sbd_split_ratio), by, bw - int(bw * sbd_split_ratio), bh)
            
        error = ""
        if not sbd_block or not made_block or len(bottom_blocks) < 4:
            # Kích hoạt Kế hoạch B: Lưới Toán học (Mathematical Grid Fallback)
            sbd_block = (int(img_w * fb_sbd[0]), int(img_h * fb_sbd[1]), int(img_w * fb_sbd[2]), int(img_h * fb_sbd[3]))
            made_block = (int(img_w * fb_made[0]), int(img_h * fb_made[1]), int(img_w * fb_made[2]), int(img_h * fb_made[3]))
            bottom_blocks = [
                (int(img_w * x), int(img_h * fb_q_y), int(img_w * fb_q_w), int(img_h * fb_q_h)) for x in fb_q_xs
            ]
            
        return {"sbd_block": sbd_block, "made_block": made_block, "question_blocks": bottom_blocks, "error": error}

class SBDReaderNode:
    def execute(self, thresh, block, image=None, debug_dir=None, debug_prefix="", **params):
        if not block: return {"result": ""}
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        else:
            gray = None
            
        bx, by, bw, bh = block
        rows = params.get("rows", 10)
        cols = params.get("cols", 6)
        
        rows_y = [int(by + bh * ((params["row_start"] + i * params["row_step"]) / params["row_total"])) for i in range(rows)]
        cols_x = [int(bx + bw * ((params["col_start"] + j * params["col_step"]) / params["col_total"])) for j in range(cols)]
        
        # Khắc phục lỗi "bành ra" (aspect ratio distortion) bằng mask hình Ellipse
        aspect_orig = 0.55 if cols > 4 else 0.30  # SBD=220/400, MADE=120/400
        radius_x = max(5, int(bw * params.get("radius_ratio", 0.035)))
        radius_y = max(5, int(bh * params.get("radius_ratio", 0.035) * aspect_orig))
        radius = (radius_x, radius_y)
        
        # BỎ LÕI 70%, DÙNG 100% BÁN KÍNH CHO XOR
        circle_area = np.pi * radius_x * radius_y
        max_xor_allowed = circle_area * (1.0 - params.get("fill_threshold", 0.40))
        sbd_mask = create_bubble_mask(radius)
        
        digits = []
        all_col_indices = []
        global_dx = 0  # Tổng độ lệch thực tế
        velocity = 0   # Vận tốc trôi (drift per column) do perspective distortion
        
        for j, cx in enumerate(cols_x):
            # Dự đoán tâm của cột hiện tại dựa trên đà (momentum) của cột trước
            predicted_offset = global_dx + velocity
            current_cx = cx + predicted_offset
            
            def _get_sbd_bubbled(cx_target, dy_target=0):
                tots = [evaluate_bubble_xor(thresh, cx_target, ry + dy_target, radius, sbd_mask) for ry in rows_y]
                val_tots = [t for t in tots if t < max_xor_allowed]
                if not val_tots: return [], tots
                min_xor = min(val_tots)
                r_thresh = 0.15 * circle_area + 0.85 * min_xor
                return [i for i, t in enumerate(tots) if t <= r_thresh and t < max_xor_allowed], tots
                
            valid_indices, tots = _get_sbd_bubbled(current_cx)
            
            # Khắc phục 1: Nếu dính phải đường kẻ dọc (gây ra nhiều valid_indices do nhiễu), thử tìm xung quanh
            if len(valid_indices) != 1:
                for test_dx in [-4, 4, -6, 6, -8, 8, -2, 2]:
                    test_valid, test_tots = _get_sbd_bubbled(current_cx + test_dx)
                    if len(test_valid) == 1:
                        valid_indices = test_valid
                        tots = test_tots
                        current_cx += test_dx
                        break
                        
            # Khắc phục 2: Nếu vẫn mất dấu, dò tìm mỏ neo bằng ô đậm nhất
            if len(valid_indices) != 1:
                best_i = min(range(len(rows_y)), key=lambda i: tots[i])
                if tots[best_i] < circle_area * 0.90:
                    dx, dy = get_bubble_centroid_vector(thresh, current_cx, rows_y[best_i], radius)
                    if abs(dx) > 0 or abs(dy) > 0:
                        s_valid, _ = _get_sbd_bubbled(current_cx + dx, dy)
                        if len(s_valid) == 1:
                            valid_indices = s_valid
                            current_cx += dx
                            
            # TIE-BREAKER: Local Contrast Stretching nếu phân vân do tẩy xóa hoặc nét quá mờ (0 hoặc >1 nét)
            # Khác với MCQ, SBD và Mã đề bắt buộc phải có 1 đáp án/cột, nên ta dùng Tie-breaker để "cứu" cả trường hợp trống.
            if len(valid_indices) != 1 and gray is not None:
                x_start = max(0, int(current_cx - radius_x * 2))
                x_end = min(gray.shape[1], int(current_cx + radius_x * 2))
                y_start = max(0, int(min(rows_y) - radius_y * 2))
                y_end = min(gray.shape[0], int(max(rows_y) + radius_y * 2))
                
                roi = gray[y_start:y_end, x_start:x_end]
                if roi.size > 0:
                    roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
                    _, roi_thresh = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    
                    def _get_sbd_bubbled_roi():
                        tots_roi = [evaluate_bubble_xor(roi_thresh, current_cx - x_start, ry - y_start, radius, sbd_mask) for ry in rows_y]
                        val_tots = [t for t in tots_roi if t < max_xor_allowed]
                        if not val_tots: return [], tots_roi
                        min_xor = min(val_tots)
                        r_thresh = 0.15 * circle_area + 0.85 * min_xor
                        return [i for i, t in enumerate(tots_roi) if t <= r_thresh and t < max_xor_allowed], tots_roi
                        
                    tie_valid, _ = _get_sbd_bubbled_roi()
                    if len(tie_valid) == 1:
                        valid_indices = tie_valid
                            
            if len(valid_indices) == 1:
                best_i = valid_indices[0]
                dx, dy = get_bubble_centroid_vector(thresh, current_cx, rows_y[best_i], radius)
                
                # Tổng lượng dịch chuyển so với DỰ ĐOÁN ban đầu
                total_shift = (current_cx + dx) - (cx + predicted_offset)
                
                # Giới hạn sự thay đổi để tránh nhảy sai cột
                correction = max(-7, min(7, total_shift))
                true_offset = predicted_offset + correction
                
                # Cập nhật vận tốc trôi
                velocity = true_offset - global_dx
                global_dx = true_offset
                
                digits.append(str(valid_indices[0]))
                all_col_indices.append(valid_indices)
            else:
                digits.append("?")
                all_col_indices.append(valid_indices)
                # Nếu mất dấu, tiếp tục trôi theo vận tốc cũ
                global_dx = predicted_offset
                
        result_str = "".join(digits)
        
        # LƯU ẢNH LỖI ĐỂ DEBUG NẾU ĐỌC SBD/MÃ ĐỀ THẤT BẠI
        if "?" in result_str and debug_dir and debug_prefix:
            try:
                import os
                crop = image[int(by):int(by+bh), int(bx):int(bx+bw)].copy() if image is not None else thresh[int(by):int(by+bh), int(bx):int(bx+bw)].copy()
                
                if len(crop.shape) == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                    
                # Vẽ đánh dấu lên ảnh lỗi để biết tại sao phân vân
                for j, indices in enumerate(all_col_indices):
                    local_cx = cols_x[j] - int(bx)
                    if len(indices) == 1:
                        local_ry = rows_y[indices[0]] - int(by)
                        cv2.circle(crop, (local_cx, local_ry), radius_x + 2, (0, 255, 0), 2)
                    elif len(indices) > 1:
                        for idx in indices:
                            local_ry = rows_y[idx] - int(by)
                            cv2.circle(crop, (local_cx, local_ry), radius_x + 2, (0, 165, 255), 2)
                    else:
                        cv2.rectangle(crop, (local_cx - radius_x, 0), (local_cx + radius_x, crop.shape[0]), (0, 0, 255), 1)

                prefix = "SBD" if params.get("cols", 6) > 4 else "MADE"
                safe_result = result_str.replace("?", "X")
                out_path = os.path.join(debug_dir, f"{debug_prefix}_ERROR_{prefix}_{safe_result}.jpg")
                cv2.imwrite(out_path, crop)
            except Exception:
                pass
                
        return {"result": result_str}

class SBDVisualizerNode:
    def execute(self, image, thresh, block, result=None, **params):
        if not block: return {"image": image}
        bx, by, bw, bh = block
        rows = params.get("rows", 10)
        cols = params.get("cols", 6)
        
        rows_y = [int(by + bh * ((params["row_start"] + i * params["row_step"]) / params["row_total"])) for i in range(rows)]
        cols_x = [int(bx + bw * ((params["col_start"] + j * params["col_step"]) / params["col_total"])) for j in range(cols)]
        
        radius = max(5, int(bw * params.get("radius_ratio", 0.035)))
        method = params.get("method", "inner_core")
        
        if result and len(result) == cols:
            # Ưu tiên kết quả đã chốt từ SBDReader (có logic fix phức tạp)
            for j, cx in enumerate(cols_x):
                char = result[j]
                if char.isdigit():
                    idx = int(char)
                    if 0 <= idx < len(rows_y):
                        cv2.circle(image, (cx, rows_y[idx]), radius + 2, (0, 255, 0), 3)
                else:
                    self._draw_fallback_col(image, thresh, cx, rows_y, radius, params)
        else:
            for j, cx in enumerate(cols_x):
                self._draw_fallback_col(image, thresh, cx, rows_y, radius, params)
                
        return {"image": image}

    def _draw_fallback_col(self, image, thresh, cx, rows_y, radius, params):
        method = params.get("method", "inner_core")
        if method == "projection":
            strip_h = max(3, int(radius * 0.8))
            min_pixels = strip_h * 2 * params.get("fill_threshold", 0.4)
            max_pixels = 0
            best_i = -1
            for i, ry in enumerate(rows_y):
                total = count_projection_peak(thresh, cx, ry, radius)
                if total > max_pixels and total > min_pixels:
                    max_pixels = total
                    best_i = i
            if best_i != -1:
                ry = rows_y[best_i]
                cv2.rectangle(image, (cx - radius, ry - strip_h), (cx + radius, ry + strip_h), (0, 255, 0), 2)
        else:
            inner_radius = max(3, int(radius * 0.70))
            min_pixels = int(np.pi * (inner_radius ** 2) * params.get("fill_threshold", 0.15))
            sbd_mask = create_bubble_mask(inner_radius)
            totals = []
            for i, ry in enumerate(rows_y):
                total = count_bubble_pixels(thresh, cx, ry, inner_radius, sbd_mask)
                totals.append(total)
            
            valid_totals = [t for t in totals if t > min_pixels]
            if valid_totals:
                max_total = max(valid_totals)
                rel_threshold = max_total * 0.85
                valid_indices = [i for i, t in enumerate(totals) if t >= rel_threshold and t > min_pixels]
                
                if len(valid_indices) == 1:
                    cv2.circle(image, (cx, rows_y[valid_indices[0]]), radius + 2, (0, 255, 0), 3)
                elif len(valid_indices) > 1:
                    # Phá
                    for idx in valid_indices:
                        cv2.circle(image, (cx, rows_y[idx]), radius + 2, (0, 165, 255), 3)


class BubbleGridDetectorNode:
    def execute(self, thresh, blocks, start_q=1, image=None, **params):
        """Trích xuất bong bóng được tô. Output: mapping q_num -> list indices"""
        if not blocks: return {"detected_bubbles": {}}
        
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        else:
            gray = None
            
        results = {}
        median_bh = np.median([b[3] for b in blocks]) if blocks else 0
        _, _, first_bw, _ = blocks[0]
        q_radius = max(5, int(first_bw * params.get("radius_ratio", 0.05)))
        method = params.get("method", "inner_core")
        
        if method == "projection":
            strip_h = max(3, int(q_radius * 0.8))
            min_pixels = strip_h * 2 * params.get("fill_threshold", 0.4)
            questions_per_block = params.get("questions_per_block", 30)
            for block_idx, (bx, by, bw, bh_orig) in enumerate(blocks):
                bh = median_bh if params.get("use_median_height", True) and median_bh > 0 else bh_orig
                rows_y = [int(by + bh * ((params["row_start"] + i * params["row_step"]) / params["row_total"])) for i in range(questions_per_block)]
                cols_x = [int(bx + bw * ((params["col_start"] + j * params["col_step"]) / params["col_total"])) for j in range(4)]
                for i, ry in enumerate(rows_y):
                    q_num = start_q + block_idx * questions_per_block + i
                    bubbled_list = []
                    for j, cx in enumerate(cols_x):
                        total = count_projection_peak(thresh, cx, ry, q_radius)
                        if total > min_pixels:
                            bubbled_list.append(j)
                    results[str(q_num)] = bubbled_list
        else:
            # Khắc phục lỗi "bành ra" bằng Ellipse Mask
            aspect_orig = 220 / 930 # MCQ chuẩn
            q_radius_x = max(5, int(first_bw * params.get("radius_ratio", 0.05)))
            q_radius_y = max(5, int(median_bh * params.get("radius_ratio", 0.05) * aspect_orig))
            q_radius_ellipse = (q_radius_x, q_radius_y)
            
            # BỎ LÕI 70%, DÙNG 100% BÁN KÍNH CHO XOR
            circle_area = np.pi * q_radius_x * q_radius_y
            max_xor_allowed = circle_area * (1.0 - params.get("fill_threshold", 0.40))
            q_mask = create_bubble_mask(q_radius_ellipse)
            questions_per_block = params.get("questions_per_block", 30)
            
            for block_idx, (bx, by, bw, bh_orig) in enumerate(blocks):
                bh = median_bh if params.get("use_median_height", True) and median_bh > 0 else bh_orig
                rows_y = [int(by + bh * ((params["row_start"] + i * params["row_step"]) / params["row_total"])) for i in range(questions_per_block)]
                cols_x = [int(bx + bw * ((params["col_start"] + j * params["col_step"]) / params["col_total"])) for j in range(4)]
    
                global_dy = 0  # Tích lũy độ lệch dọc qua các hàng
                for i, ry in enumerate(rows_y):
                    q_num = start_q + block_idx * questions_per_block + i
                    current_ry = ry + global_dy
                    
                    # Bước 1: Thu thập điểm XOR của 4 ô (Càng nhỏ càng tốt)
                    def _get_bubbled(ry_target, dx_target=0):
                        tots = [evaluate_bubble_xor(thresh, cx + dx_target, ry_target, q_radius_ellipse, q_mask) for cx in cols_x]
                        val_tots = [t for t in tots if t < max_xor_allowed]
                        if not val_tots: return [], tots
                        min_xor = min(val_tots)
                        r_thresh = 0.15 * circle_area + 0.85 * min_xor
                        return [j for j, t in enumerate(tots) if t <= r_thresh and t < max_xor_allowed], tots

                    bubbled_list, tots = _get_bubbled(current_ry)
                    
                    if len(bubbled_list) != 1:
                        # Tìm ô có lượng nét chì nhiều nhất (XOR nhỏ nhất) trong tất cả 4 ô
                        best_j = min(range(len(cols_x)), key=lambda j: tots[j])
                        # Chỉ tịnh tiến nếu có nét chì thực sự (XOR < 90% diện tích)
                        if tots[best_j] < circle_area * 0.90:
                            dx, dy = get_bubble_centroid_vector(thresh, cols_x[best_j], current_ry, q_radius_ellipse)
                            if dx != 0 or dy != 0:
                                s_bubbled, _ = _get_bubbled(current_ry + dy, dx_target=dx)
                                if len(s_bubbled) == 1:
                                    bubbled_list = s_bubbled
                                    global_dy += max(-2, min(2, dy))
                                    
                    if len(bubbled_list) == 1:
                        # Cập nhật mỏ neo liên tục từ các hàng đã chấm tốt
                        best_j = bubbled_list[0]
                        dx, dy = get_bubble_centroid_vector(thresh, cols_x[best_j], current_ry, q_radius_ellipse)
                        global_dy += max(-2, min(2, dy))
                    elif gray is not None and len(bubbled_list) > 1:
                        # TIE-BREAKER: Kéo giãn tương phản cục bộ chỉ khi có TỪ 2 ĐÁP ÁN TRỞ LÊN
                        # Tránh việc ép một câu trống (0 đáp án) thành có đáp án từ nhiễu giấy.
                        x_start = max(0, int(min(cols_x) - q_radius_x * 2))
                        x_end = min(gray.shape[1], int(max(cols_x) + q_radius_x * 2))
                        y_start = max(0, int(current_ry - q_radius_y * 2))
                        y_end = min(gray.shape[0], int(current_ry + q_radius_y * 2))
                        
                        roi = gray[y_start:y_end, x_start:x_end]
                        if roi.size > 0:
                            # Ép pixel sáng nhất thành trắng, tối nhất thành đen thui
                            roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
                            # Otsu tự động tìm ngưỡng chuẩn nhất chỉ cho riêng 4 ô này
                            _, roi_thresh = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                            
                            def _get_bubbled_roi():
                                tots = [evaluate_bubble_xor(roi_thresh, cx - x_start, current_ry - y_start, q_radius_ellipse, q_mask) for cx in cols_x]
                                val_tots = [t for t in tots if t < max_xor_allowed]
                                if not val_tots: return [], tots
                                min_xor = min(val_tots)
                                tie_valid = [j for j, t in enumerate(tots) if t == min_xor]
                                return tie_valid, tots
                                
                            tie_bubbled_list, _ = _get_bubbled_roi()
                            if len(tie_bubbled_list) == 1:
                                bubbled_list = tie_bubbled_list
                                
                    results[str(q_num)] = bubbled_list
                
        return {"detected_bubbles": results}


class MCQScorerNode:
    def execute(self, detected_bubbles, answers, made=None):
        """So khớp bong bóng với đáp án Excel. Output: Điểm và danh sách lỗi."""
        options = ["A", "B", "C", "D"]
        results = {}
        score = 0
        total_q = 0
        wrong_list, missing_list, ruined_list = [], [], []
        error_msg = ""
        
        current_answers = answers
        if answers and isinstance(next(iter(answers.values()), None), dict):
            if made and made in answers:
                current_answers = answers[made]
            else:
                error_msg = f"Lỗi: Không tìm thấy mã đề '{made}' trong đáp án. Đã dừng chấm điểm."
                return {
                    "score": 0,
                    "total_q": 0,
                    "results": results,
                    "wrong_list": [],
                    "missing_list": [],
                    "ruined_list": [],
                    "error": error_msg
                }
                
        for q_num, bubbled_list in detected_bubbles.items():
            chosen_answer = ",".join([options[b] for b in bubbled_list]) if bubbled_list else "Chưa chọn"
            results[q_num] = chosen_answer
            
            correct_answer = current_answers.get(str(q_num))
            if correct_answer:
                total_q += 1
                expected_count = len(correct_answer.split(","))
                if correct_answer == chosen_answer:
                    score += 1
                else:
                    if len(bubbled_list) == 0: missing_list.append(str(q_num))
                    elif len(bubbled_list) > expected_count: ruined_list.append(str(q_num))
                    else: wrong_list.append(str(q_num))
                    
        return {
            "score": score,
            "total_q": total_q,
            "results": results,
            "wrong_list": wrong_list,
            "missing_list": missing_list,
            "ruined_list": ruined_list,
            "error": error_msg
        }


class MCQVisualizerNode:
    def execute(self, image, detected_bubbles, answers, blocks, made=None, start_q=1, **params):
        """Vẽ kết quả xanh/đỏ lên ảnh gốc."""
        if not blocks: return {"image": image}
        
        options = ["A", "B", "C", "D"]
        current_answers = answers
        if answers and isinstance(next(iter(answers.values()), None), dict):
            if made and made in answers:
                current_answers = answers[made]
            else:
                current_answers = answers[next(iter(answers.keys()))]
                
        median_bh = np.median([b[3] for b in blocks]) if blocks else 0
        _, _, first_bw, _ = blocks[0]
        q_radius = max(5, int(first_bw * params.get("radius_ratio", 0.05)))
        questions_per_block = params.get("questions_per_block", 30)
        draw_thick = int(params.get("draw_thick", 3))
        draw_offset = int(params.get("draw_offset", 2))
        
        for block_idx, (bx, by, bw, bh_orig) in enumerate(blocks):
            bh = median_bh if params.get("use_median_height", True) and median_bh > 0 else bh_orig
            rows_y = [int(by + bh * ((params["row_start"] + i * params["row_step"]) / params["row_total"])) for i in range(questions_per_block)]
            cols_x = [int(bx + bw * ((params["col_start"] + j * params["col_step"]) / params["col_total"])) for j in range(4)]

            for i, ry in enumerate(rows_y):
                q_num = str(start_q + block_idx * questions_per_block + i)
                bubbled_list = detected_bubbles.get(q_num, [])
                chosen_answer = ",".join([options[b] for b in bubbled_list]) if bubbled_list else ""
                
                # Vẽ tất cả các vị trí lý thuyết bằng viền đứt nét màu xanh dương nhạt để debug căn lề
                for j, cx in enumerate(cols_x):
                    cv2.circle(image, (cx, ry), q_radius, (255, 200, 100), 1)
                
                method = params.get("method", "inner_core")
                strip_h = max(3, int(q_radius * 0.8))
                
                correct_answer = current_answers.get(q_num)
                if correct_answer:
                    expected_count = len(correct_answer.split(","))
                    if correct_answer == chosen_answer:
                        for b in bubbled_list:
                            if method == "projection":
                                cv2.rectangle(image, (cols_x[b] - q_radius, ry - strip_h), (cols_x[b] + q_radius, ry + strip_h), (0, 255, 0), 2)
                            else:
                                cv2.circle(image, (cols_x[b], ry), q_radius + draw_offset, (0, 255, 0), draw_thick)
                    else:
                        if 0 < len(bubbled_list) <= expected_count:
                            for b in bubbled_list:
                                if method == "projection":
                                    cv2.rectangle(image, (cols_x[b] - q_radius, ry - strip_h), (cols_x[b] + q_radius, ry + strip_h), (0, 0, 255), 2)
                                else:
                                    cv2.circle(image, (cols_x[b], ry), q_radius + draw_offset, (0, 0, 255), draw_thick)
                        elif len(bubbled_list) > expected_count:
                            # TÔ SAI/QUÁ NHIỀU ĐÁP ÁN: Vẽ viền màu cam
                            for b in bubbled_list:
                                if method == "projection":
                                    cv2.rectangle(image, (cols_x[b] - q_radius, ry - strip_h), (cols_x[b] + q_radius, ry + strip_h), (0, 165, 255), 2)
                                else:
                                    cv2.circle(image, (cols_x[b], ry), q_radius + draw_offset, (0, 165, 255), draw_thick)
        return {"image": image}
