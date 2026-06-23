import cv2
import os
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
    def execute(self, thresh, block, image=None, **params):
        if not block: return {"result": ""}
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
        
        circle_area = np.pi * radius_x * radius_y
        max_xor_allowed = circle_area * (1.0 - params.get("fill_threshold", 0.40))
        sbd_mask = create_bubble_mask(radius)
        
        def attempt_read(current_thresh):
            digits = []
            global_dx = 0  # Tổng độ lệch thực tế
            velocity = 0   # Vận tốc trôi (drift per column) do perspective distortion
            
            for j, cx in enumerate(cols_x):
                predicted_offset = global_dx + velocity
                current_cx = cx + predicted_offset
                
                def _get_sbd_bubbled(cx_target, dy_target=0):
                    tots = [evaluate_bubble_xor(current_thresh, cx_target, ry + dy_target, radius, sbd_mask) for ry in rows_y]
                    val_tots = [t for t in tots if t < max_xor_allowed]
                    if not val_tots: return [], tots
                    min_xor = min(val_tots)
                    r_thresh = 0.15 * circle_area + 0.85 * min_xor
                    return [i for i, t in enumerate(tots) if t <= r_thresh and t < max_xor_allowed], tots
                    
                valid_indices, tots = _get_sbd_bubbled(current_cx)
                
                # Khắc phục 1: Tránh nhiễu dọc
                if len(valid_indices) != 1:
                    for test_dx in [-4, 4, -6, 6, -8, 8, -2, 2]:
                        test_valid, test_tots = _get_sbd_bubbled(current_cx + test_dx)
                        if len(test_valid) == 1:
                            valid_indices = test_valid
                            tots = test_tots
                            current_cx += test_dx
                            break
                            
                # Khắc phục 2: Dò tìm mỏ neo bằng ô đậm nhất
                if len(valid_indices) != 1:
                    best_i = min(range(len(rows_y)), key=lambda i: tots[i])
                    if tots[best_i] < circle_area * 0.95:
                        dx, dy = get_bubble_centroid_vector(current_thresh, current_cx, rows_y[best_i], radius)
                        if abs(dx) > 0 or abs(dy) > 0:
                            s_valid, _ = _get_sbd_bubbled(current_cx + dx, dy)
                            if len(s_valid) == 1:
                                valid_indices = s_valid
                                current_cx += dx
                                
                if len(valid_indices) == 1:
                    best_i = valid_indices[0]
                    dx, dy = get_bubble_centroid_vector(current_thresh, current_cx, rows_y[best_i], radius)
                    total_shift = (current_cx + dx) - (cx + predicted_offset)
                    velocity = total_shift
                    global_dx += total_shift
                    digits.append(str(valid_indices[0]))
                elif len(valid_indices) > 1:
                    # Khắc phục 3: Nhiều ứng viên cùng lọt → chọn ô có XOR thấp nhất (tô đậm nhất)
                    best_i = min(valid_indices, key=lambda i: tots[i])
                    dx, dy = get_bubble_centroid_vector(current_thresh, current_cx, rows_y[best_i], radius)
                    total_shift = (current_cx + dx) - (cx + predicted_offset)
                    velocity = total_shift
                    global_dx += total_shift
                    digits.append(str(best_i))
                else:
                    digits.append("?")
                    global_dx = predicted_offset
                    
            return digits

        # Thử đọc bằng global thresh trước! Nhanh và chính xác với ngưỡng tiêu chuẩn.
        debug_dir = params.get("debug_dir")
        debug_prefix = params.get("debug_prefix", "")
        if debug_dir and os.path.exists(debug_dir):
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}{params.get('cols')}_cols_base_thresh.jpg"), thresh[by:by+bh, bx:bx+bw])
            
        base_digits = attempt_read(thresh)
        base_q_count = base_digits.count("?")
        if base_q_count == 0:
            return {"result": "".join(base_digits)}

        # Nếu bị mất nét (có dấu ?) và có ảnh gốc, kích hoạt cứu hộ Dynamic Threshold Iteration!
        if image is not None:
            best_digits = base_digits
            best_q_count = base_q_count
            
            # Khởi tạo temp_thresh an toàn theo giới hạn ảnh
            img_h, img_w = image.shape[:2]
            y1, y2 = max(0, by), min(img_h, by + bh)
            x1, x2 = max(0, bx), min(img_w, bx + bw)
            
            from image_processing.preprocessing.apply_adaptive_threshold import process as apply_thresh
            # Bắt đầu từ 15 (tối ưu nhất) và giảm dần để vớt nét chì mờ
            for C_val in [15, 12, 9, 6, 4]:
                temp_thresh = thresh.copy()
                roi = image[y1:y2, x1:x2]
                if roi.size > 0:
                    local_thresh = apply_thresh(roi, block_size=91, C=C_val, sharpen=False, blur=False)
                    local_thresh = cv2.medianBlur(local_thresh, 3)
                    
                    if debug_dir and os.path.exists(debug_dir):
                        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}{params.get('cols')}_cols_dynamic_C{C_val}.jpg"), local_thresh)
                        
                    temp_thresh[y1:y2, x1:x2] = local_thresh
                    
                digits = attempt_read(temp_thresh)
                q_count = digits.count("?")
                
                if q_count == 0:
                    return {"result": "".join(digits)} # Đã đọc trọn vẹn
                
                if q_count < best_q_count:
                    best_q_count = q_count
                    best_digits = digits
                    
            return {"result": "".join(best_digits)}
            
        else:
            return {"result": "".join(attempt_read(thresh))}

class SBDVisualizerNode:
    def execute(self, image, thresh, block, **params):
        if not block: return {"image": image}
        bx, by, bw, bh = block
        rows = params.get("rows", 10)
        cols = params.get("cols", 6)
        
        # Gọi lại SBDReaderNode để lấy chính xác mảng digits đã xuất ra Excel
        reader = SBDReaderNode()
        res = reader.execute(thresh, block, image, **params)
        digits = res.get("result", "")
        
        rows_y = [int(by + bh * ((params["row_start"] + i * params["row_step"]) / params["row_total"])) for i in range(rows)]
        cols_x = [int(bx + bw * ((params["col_start"] + j * params["col_step"]) / params["col_total"])) for j in range(cols)]
        
        aspect_orig = 0.55 if cols > 4 else 0.30
        radius_x = max(5, int(bw * params.get("radius_ratio", 0.035)))
        radius_y = max(5, int(bh * params.get("radius_ratio", 0.035) * aspect_orig))
        
        for j, cx in enumerate(cols_x):
            if j < len(digits):
                d = digits[j]
                if d.isdigit():
                    ry = rows_y[int(d)]
                    cv2.ellipse(image, (cx, ry), (radius_x + 2, radius_y + 2), 0, 0, 360, (0, 255, 0), 3)
                else:
                    # Phân vân
                    for ry in rows_y:
                        cv2.ellipse(image, (cx, ry), (radius_x + 2, radius_y + 2), 0, 0, 360, (0, 165, 255), 2)
                
        return {"image": image}


class BubbleGridDetectorNode:
    def execute(self, thresh, blocks, start_q=1, **params):
        """Trích xuất bong bóng được tô. Output: mapping q_num -> list indices"""
        if not blocks: return {"detected_bubbles": {}}
        
        results = {}
        median_bh = np.median([b[3] for b in blocks]) if blocks else 0
        _, _, first_bw, _ = blocks[0]
        q_radius = max(5, int(first_bw * params.get("radius_ratio", 0.05)))
        median_bw = np.median([b[2] for b in blocks]) if blocks else 0
        method = params.get("method", "inner_core")
        
        if method == "projection":
            strip_h = max(3, int(q_radius * 0.8))
            min_pixels = strip_h * 2 * params.get("fill_threshold", 0.4)
            questions_per_block = params.get("questions_per_block", 30)
            for block_idx, (bx, by, bw_orig, bh_orig) in enumerate(blocks):
                bh = median_bh if params.get("use_median_height", True) and median_bh > 0 else bh_orig
                bw = median_bw if params.get("use_median_width", True) and median_bw > 0 else bw_orig
                rows_y = [int(by + bh * ((params["row_start"] + i * params["row_step"]) / params["row_total"])) for i in range(questions_per_block)]
                cols_x = [int(bx + bw * ((params["col_start"] + j * params["col_step"]) / params["col_total"])) for j in range(4)]
                for i, ry in enumerate(rows_y):
                    q_num = start_q + block_idx * questions_per_block + i
                    bubbled_list = []
                    for j, cx in enumerate(cols_x):
                        total = count_projection_peak(thresh, cx, ry, max(5, int(bw * params.get("radius_ratio", 0.05))))
                        if total > min_pixels:
                            bubbled_list.append(j)
                    results[str(q_num)] = bubbled_list
        else:
            # Khắc phục lỗi "bành ra" bằng Ellipse Mask
            aspect_orig = 220 / 930 # MCQ chuẩn
            q_radius_x = max(5, int(first_bw * params.get("radius_ratio", 0.05)))
            q_radius_y = max(5, int(median_bh * params.get("radius_ratio", 0.05) * aspect_orig))
            q_radius_ellipse = (q_radius_x, q_radius_y)
            circle_area = np.pi * q_radius_x * q_radius_y
            q_mask = create_bubble_mask(q_radius_ellipse)
            
            # Sử dụng "Lõi 70%" (Inner Core) để tính độ đậm (XOR).
            # Điều này giúp bỏ qua viền in sẵn (Outline) vốn có thể bị in đậm trên ảnh tối.
            # Ô không tô luôn có lõi rỗng, ô tô luôn có lõi đặc.
            core_rx = max(3, int(q_radius_x * 0.70))
            core_ry = max(3, int(q_radius_y * 0.70))
            core_radius = (core_rx, core_ry)
            core_area = np.pi * core_rx * core_ry
            
            # max_xor_allowed là ngưỡng để coi là "có mực" (ví dụ 60% lõi bị tô)
            max_xor_allowed = core_area * (1.0 - params.get("fill_threshold", 0.40))
            core_mask = create_bubble_mask(core_radius)
            questions_per_block = params.get("questions_per_block", 30)
            
            for block_idx, (bx, by, bw_orig, bh_orig) in enumerate(blocks):
                bh = median_bh if params.get("use_median_height", True) and median_bh > 0 else bh_orig
                bw = median_bw if params.get("use_median_width", True) and median_bw > 0 else bw_orig
                rows_y = [int(by + bh * ((params["row_start"] + i * params["row_step"]) / params["row_total"])) for i in range(questions_per_block)]
                cols_x = [int(bx + bw * ((params["col_start"] + j * params["col_step"]) / params["col_total"])) for j in range(4)]
    
                for i, ry in enumerate(rows_y):
                    q_num = start_q + block_idx * questions_per_block + i
                    
                    # Hàm đánh giá chung
                    def _get_bubbled(ry_target, radius, mask, area, dx_target=0):
                        max_allowed = area * (1.0 - params.get("fill_threshold", 0.40))
                        tots = [evaluate_bubble_xor(thresh, cx + dx_target, ry_target, radius, mask) for cx in cols_x]
                        val_tots = [t for t in tots if t < max_allowed]
                        if not val_tots: return [], tots
                        min_xor = min(val_tots)
                        r_thresh = 0.15 * area + 0.85 * min_xor
                        return [j for j, t in enumerate(tots) if t <= r_thresh and t < max_allowed], tots

                    # Lần 1: Dùng full 100% radius (Tốt cho ảnh nhiễu nhẹ)
                    bubbled_list, tots = _get_bubbled(ry, q_radius_ellipse, q_mask, circle_area)
                    
                    # Lần 2: Nếu phân vân (0 hoặc >1 đáp án), dùng lõi 70% (Tốt để khử tẩy xóa viền)
                    if len(bubbled_list) != 1:
                        bubbled_list_core, tots_core = _get_bubbled(ry, core_radius, core_mask, core_area)
                        if len(bubbled_list_core) == 1:
                            bubbled_list = bubbled_list_core
                            tots = tots_core
                            
                    # Lần 3: Thử tịnh tiến tâm nếu vẫn không tìm được 1 đáp án duy nhất
                    if len(bubbled_list) != 1:
                        best_j = min(range(len(cols_x)), key=lambda j: tots[j])
                        if tots[best_j] < circle_area * 0.90:
                            dx, dy = get_bubble_centroid_vector(thresh, cols_x[best_j], ry, q_radius_ellipse)
                            if dx != 0 or dy != 0:
                                s_bubbled, _ = _get_bubbled(ry + dy, q_radius_ellipse, q_mask, circle_area, dx_target=dx)
                                if len(s_bubbled) == 1:
                                    bubbled_list = s_bubbled
                                else:
                                    # Fallback tịnh tiến với lõi 70%
                                    s_bubbled_core, _ = _get_bubbled(ry + dy, core_radius, core_mask, core_area, dx_target=dx)
                                    if len(s_bubbled_core) == 1:
                                        bubbled_list = s_bubbled_core
                                        
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
                error_msg = f"Lỗi: Không tìm thấy mã đề '{made}' trong đáp án."
                current_answers = answers[next(iter(answers.keys()))] # Fallback to grade it anyway, but it has an error
                
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
