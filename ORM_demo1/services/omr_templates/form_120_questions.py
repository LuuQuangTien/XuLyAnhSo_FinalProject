import cv2
import numpy as np
from services.omr_templates.base_template import BaseOMRTemplate

def create_bubble_mask(radius):
    """Tạo trước một mask hình tròn tĩnh để tái sử dụng."""
    size = radius * 2
    mask = np.zeros((size, size), dtype="uint8")
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    return mask

def count_bubble_pixels(thresh, cx, cy, radius, precomputed_mask=None):
    """Đếm số pixel đen (đã threshold thành trắng) trong một vùng tròn ROI nhỏ."""
    sy, ey = max(0, cy - radius), min(thresh.shape[0], cy + radius)
    sx, ex = max(0, cx - radius), min(thresh.shape[1], cx + radius)
    roi = thresh[sy:ey, sx:ex]
    if roi.size == 0: return 0
    
    # Ưu tiên dùng mask đã tạo sẵn để tăng tốc (tránh cấp phát bộ nhớ liên tục)
    if precomputed_mask is not None and roi.shape == precomputed_mask.shape:
        return cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=precomputed_mask))
    
    # Fallback nếu bubble bị dính ra lề ảnh (hiếm khi xảy ra)
    mask = np.zeros(roi.shape, dtype="uint8")
    mcx, mcy = cx - sx, cy - sy
    cv2.circle(mask, (mcx, mcy), radius, 255, -1)
    
    return cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=mask))

class Form120QuestionsTemplate(BaseOMRTemplate):
    name = "Phiếu 120 Câu hỏi"
    description = "Mẫu phiếu thi gồm 120 câu hỏi, đọc Số báo danh và Mã đề thi."
    preview_image_path = "resources/images/phieu_trac_nghiem_120cau_OMR_optimized (1).png"
    requires_alignment = True

    def _analyze_blocks(self, blocks: list, image_shape: tuple) -> tuple[float, tuple, tuple, list]:
        if not blocks or not image_shape:
            return 0.0, None, None, []
            
        img_h, img_w = image_shape[:2]
        
        # 1. Filter overlapping blocks (keep smaller valid blocks, drop fat merged ones)
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
                
        # 2. Separate Top (cy < H/2) and Bottom (cy > H/4)
        top_blocks = []
        bottom_blocks = []
        for b in filtered_blocks:
            bx, by, bw, bh = b
            cy = by + bh/2
            if cy < img_h * 0.45:
                top_blocks.append(b)
            if cy > img_h * 0.2: 
                bottom_blocks.append(b)
                
        # 3. Filter Top blocks: must be on the right side (cx > W/2)
        top_right_blocks = [b for b in top_blocks if (b[0] + b[2]/2) > img_w * 0.45]
        top_right_blocks = sorted(top_right_blocks, key=lambda b: b[0]) # Left to right
        
        # Filter Bottom blocks: keep the 4 largest
        bottom_blocks = sorted(bottom_blocks, key=lambda b: b[2]*b[3], reverse=True)[:4]
        bottom_blocks = sorted(bottom_blocks, key=lambda b: b[0]) # Left to right
        
        sbd_block = None
        made_block = None
        question_blocks = bottom_blocks
        confidence = 1.0
        
        # Assign Top Blocks
        if len(top_right_blocks) >= 2:
            sbd_block = top_right_blocks[0] # SBD xa góc phải hơn (cx nhỏ hơn)
            made_block = top_right_blocks[1]
        elif len(top_right_blocks) == 1:
            bx, by, bw, bh = top_right_blocks[0]
            # Bổ đôi khối dính chùm
            sbd_block = (bx, by, int(bw * 0.65), bh)
            made_block = (bx + int(bw * 0.65), by, bw - int(bw * 0.65), bh)
            confidence -= 0.3
        else:
            confidence -= 0.6
            
        if len(question_blocks) < 4:
            confidence -= 0.3 * (4 - len(question_blocks)) # Tăng hình phạt thiếu block
            
        # 4. Geometric Heuristics Penalties
        if sbd_block and made_block:
            sbd_cy = sbd_block[1] + sbd_block[3]/2
            made_cy = made_block[1] + made_block[3]/2
            if abs(sbd_cy - made_cy) > img_h * 0.05:
                confidence -= 0.1
                
        if len(question_blocks) > 0:
            # So sánh độ tương đồng về chiều rộng (Bắt lỗi khối mập mạp do dính nét)
            widths = [b[2] for b in question_blocks]
            max_w, min_w = max(widths), min(widths)
            if max_w > 0 and (max_w - min_w) / max_w > 0.15:
                confidence -= 0.2
                
            # So sánh độ tương đồng về chiều cao
            heights = [b[3] for b in question_blocks]
            max_h, min_h = max(heights), min(heights)
            if max_h > 0 and (max_h - min_h) / max_h > 0.15:
                confidence -= 0.1
                
            # Kiểm tra khoảng cách X giữa các cột
            if len(question_blocks) >= 3:
                centers_x = [b[0] + b[2]/2 for b in question_blocks]
                gaps = [centers_x[i+1] - centers_x[i] for i in range(len(question_blocks)-1)]
                max_gap, min_gap = max(gaps), min(gaps)
                if max_gap > 0 and (max_gap - min_gap) / max_gap > 0.2:
                    confidence -= 0.15
                    
            # Dóng hàng viền phải
            if made_block and len(question_blocks) == 4:
                made_right = made_block[0] + made_block[2]
                last_q_right = question_blocks[-1][0] + question_blocks[-1][2]
                if abs(made_right - last_q_right) > img_w * 0.05:
                    confidence -= 0.15
                    
        return max(0.0, confidence), sbd_block, made_block, question_blocks

    def calculate_confidence(self, blocks: list, image_shape: tuple = None) -> float:
        if not image_shape:
            # Fallback nếu không truyền image_shape
            if len(blocks) == 6: return 1.0
            elif 4 <= len(blocks) <= 5: return 0.7
            else: return 0.0
            
        conf, _, _, _ = self._analyze_blocks(blocks, image_shape)
        return conf

    def grade(self, image: np.ndarray, thresh: np.ndarray, blocks: list, answers: dict) -> tuple[np.ndarray, str, dict]:
        output_image = image.copy()
        height, width = image.shape[:2]
        
        # Dùng hàm analyze_blocks để lấy blocks chuẩn
        conf, sbd_block, made_block, question_blocks = self._analyze_blocks(blocks, image.shape)
        
        # Fallback nếu điểm tự tin quá thấp (không thể trích xuất đủ block)
        if conf < 0.4 or len(question_blocks) < 4:
            print("[GEOMETRY] Using Mathematical Grid Fallback for Blocks")
            # SBD: x=0.535, y=0.018, w=0.270, h=0.232
            sbd_block = (int(width * 0.535), int(height * 0.018), int(width * 0.270), int(height * 0.232))
            # Mã đề: x=0.820, y=0.018, w=0.155, h=0.232
            made_block = (int(width * 0.820), int(height * 0.018), int(width * 0.155), int(height * 0.232))
            # 4 Khối câu hỏi (y=0.270, w=0.218, h=0.708)
            question_blocks = [
                (int(width * 0.024), int(height * 0.270), int(width * 0.218), int(height * 0.708)),
                (int(width * 0.268), int(height * 0.270), int(width * 0.218), int(height * 0.708)),
                (int(width * 0.512), int(height * 0.270), int(width * 0.218), int(height * 0.708)),
                (int(width * 0.755), int(height * 0.270), int(width * 0.218), int(height * 0.708))
            ]
        
        # Cấu hình Options và Điểm
        options = ["A", "B", "C", "D"]
        results = {}
        score = 0
        total_questions_has_answer = 0
        
        wrong_list = []
        missing_list = []
        ruined_list = []
        
        extracted_sbd = ""
        extracted_made = ""
        
        # Tạo Threshold chuyên dụng cho việc đọc bong bóng: Block size 91 giúp không bị rỗng ruột khi tô chì mờ
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bubble_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 10)
        
        # --- 1. ĐỌC SỐ BÁO DANH ---
        if sbd_block:
            bx, by, bw, bh = sbd_block
            sbd_rows_y = [int(by + bh * ((9.75 + i * 5.5) / 62.0)) for i in range(10)]
            sbd_cols_x = [int(bx + bw * ((9.1 + j * 7.2) / 48.7)) for j in range(6)]
            radius = max(5, int(bw * 0.035))
            min_pixels = int(np.pi * (radius ** 2) * 0.55) # Tăng lên 55% để chống vệt gạch ngang
            
            # Tối ưu hóa: Tạo mask 1 lần duy nhất cho toàn bộ SBD block
            sbd_mask = create_bubble_mask(radius)
            
            sbd_digits = []
            for j, cx in enumerate(sbd_cols_x):
                col_digit = "?"
                max_pixels = 0
                best_i = -1
                for i, ry in enumerate(sbd_rows_y):
                    total = count_bubble_pixels(bubble_thresh, cx, ry, radius, sbd_mask)
                    if total > max_pixels and total > min_pixels:
                        max_pixels = total
                        col_digit = str(i)
                        best_i = i
                        
                sbd_digits.append(col_digit)
                
                # Tô xanh lá vòng tròn SBD được nhận diện thành công
                if best_i != -1:
                    cv2.circle(output_image, (cx, sbd_rows_y[best_i]), radius + 2, (0, 255, 0), 3)
                    
            extracted_sbd = "".join(sbd_digits)

        # --- 2. ĐỌC MĐ ---
        if made_block:
            bx, by, bw, bh = made_block
            made_rows_y = [int(by + bh * ((9.75 + i * 5.5) / 62.0)) for i in range(10)]
            made_cols_x = [int(bx + bw * ((9.1 + j * 7.2) / 27.1)) for j in range(3)]
            radius = max(5, int(bw * 0.06))
            min_pixels = int(np.pi * (radius ** 2) * 0.55) # Tăng lên 55% để chống vệt gạch ngang
            
            # Tối ưu hóa: Tạo mask 1 lần duy nhất cho toàn bộ MĐ block
            made_mask = create_bubble_mask(radius)
            
            made_digits = []
            for j, cx in enumerate(made_cols_x):
                col_digit = "?"
                max_pixels = 0
                best_i = -1
                for i, ry in enumerate(made_rows_y):
                    total = count_bubble_pixels(bubble_thresh, cx, ry, radius, made_mask)
                    if total > max_pixels and total > min_pixels:
                        max_pixels = total
                        col_digit = str(i)
                        best_i = i
                        
                made_digits.append(col_digit)
                
                # Tô xanh lá vòng tròn MĐ được nhận diện thành công
                if best_i != -1:
                    cv2.circle(output_image, (cx, made_rows_y[best_i]), radius + 2, (0, 255, 0), 3)
                    
            extracted_made = "".join(made_digits)

        # --- 2.5 CHỌN ĐÁP ÁN THEO MÃ ĐỀ ---
        current_answers = answers
        made_warning = ""
        if answers and isinstance(next(iter(answers.values()), None), dict):
            # Đây là dictionary đa mã đề từ file Excel
            if not extracted_made or extracted_made == "Không đọc được":
                raise ValueError("Không thể đọc được Mã đề trên bài thi!")
            elif extracted_made in answers:
                current_answers = answers[extracted_made]
            else:
                # Chặn luôn, không cho chấm đại nữa
                raise ValueError(f"Mã đề '{extracted_made}' không tồn tại trong file đáp án Excel!")
                
        # Lấy số câu hỏi lớn nhất từ file đáp án (để bỏ qua các vùng giấy thừa)
        max_q = 120
        if current_answers:
            try:
                max_q = max([int(k) for k in current_answers.keys() if k.isdigit()])
            except ValueError:
                pass

        # Tính Median Height của các khối câu hỏi để KHỬ NHIỄU BÓNG RÂM ở đáy tờ giấy
        median_bh = 0
        if len(question_blocks) > 0:
            median_bh = np.median([b[3] for b in question_blocks])

        # Tính toán trước Radius và Mask cho các câu hỏi vì chúng giống nhau ở tất cả các blocks
        q_radius = 5
        q_mask = None
        if len(question_blocks) > 0:
            _, _, first_bw, _ = question_blocks[0]
            q_radius = max(5, int(first_bw * 0.05))
            q_mask = create_bubble_mask(q_radius)
            
        min_pixels_threshold = int(np.pi * (q_radius ** 2) * 0.55) # Tăng lên 55% để chống vệt gạch ngang

        # --- 3. CHẤM CÂU HỎI ---
        for block_idx, (bx, by, bw, bh_orig) in enumerate(question_blocks):
            # Sử dụng median_bh thay vì bh_orig để chiều cao các khối luôn đồng nhất, bất chấp bóng râm
            bh = median_bh if median_bh > 0 else bh_orig
            
            rows_y = [int(by + bh * ((3.2 + i * 6.4) / 192.0)) for i in range(30)]
            cols_x = [int(bx + bw * ((11.4375 + j * 7.875) / 39.0)) for j in range(4)]

            for i, ry in enumerate(rows_y):
                q_num = block_idx * 30 + i + 1
                if q_num > max_q: break
                
                bubbled_list = []
                
                for j, cx in enumerate(cols_x):
                    total = count_bubble_pixels(bubble_thresh, cx, ry, q_radius, q_mask)
                    
                    if total > min_pixels_threshold:
                        bubbled_list.append(j)
                        
                if bubbled_list:
                    chosen_answer = ",".join([options[b] for b in bubbled_list])
                else:
                    chosen_answer = "Chưa chọn"
                results[q_num] = chosen_answer
                
                correct_answer = current_answers.get(str(q_num))
                if correct_answer:
                    total_questions_has_answer += 1
                    if correct_answer == chosen_answer:
                        score += 1
                        color = (0, 255, 0) # Xanh lá
                        for b in bubbled_list:
                            cv2.circle(output_image, (cols_x[b], ry), q_radius + 2, color, 3)
                    else:
                        color = (0, 0, 255) # Đỏ
                        expected_count = len(correct_answer.split(","))
                        
                        # Ghi nhận trạng thái lỗi
                        if len(bubbled_list) == 0:
                            missing_list.append(str(q_num))
                        elif len(bubbled_list) > expected_count:
                            ruined_list.append(str(q_num))
                        else:
                            wrong_list.append(str(q_num))
                            
                        # Không vẽ nếu bị phá (tô nhiều hơn đáp án quy định)
                        if 0 < len(bubbled_list) <= expected_count:
                            for b in bubbled_list:
                                cv2.circle(output_image, (cols_x[b], ry), q_radius + 2, color, 3)

        # Cập nhật kết quả text
        if total_questions_has_answer > 0:
            final_score = (score / total_questions_has_answer) * 10
        else:
            final_score = 0.0
        sbd_str = extracted_sbd if extracted_sbd else "Không đọc được"
        made_str = (extracted_made if extracted_made else "Không đọc được") + made_warning
        
        result_text = (
            f"--- KẾT QUẢ CHẤM ĐIỂM ---\n"
            f"Số báo danh: {sbd_str}\n"
            f"Mã đề thi: {made_str}\n"
            f"Số câu đúng: {score}/{total_questions_has_answer}\n"
            f"Điểm số: {final_score:.2f} / 10\n"
        )
        
        notes_parts = []
        if wrong_list:
            notes_parts.append(f"Sai: {','.join(wrong_list)}")
        if missing_list:
            notes_parts.append(f"Bỏ trống: {','.join(missing_list)}")
        if ruined_list:
            notes_parts.append(f"Bị phá: {','.join(ruined_list)}")
            
        score_dict = {
            'correct': score,
            'total': total_questions_has_answer,
            'final_score': final_score,
            'sbd': extracted_sbd,
            'made': extracted_made,
            'notes': " | ".join(notes_parts),
            'raw_answers': results
        }
        
        return output_image, result_text, score_dict
