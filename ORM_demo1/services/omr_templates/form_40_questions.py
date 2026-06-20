import cv2
import numpy as np
from services.omr_templates.base_template import BaseOMRTemplate

def count_bubble_pixels(thresh, cx, cy, radius):
    """Đếm số pixel đen (đã threshold thành trắng) trong một vùng tròn ROI nhỏ."""
    sy, ey = max(0, cy - radius), min(thresh.shape[0], cy + radius)
    sx, ex = max(0, cx - radius), min(thresh.shape[1], cx + radius)
    roi = thresh[sy:ey, sx:ex]
    if roi.size == 0: return 0
    
    mask = np.zeros(roi.shape, dtype="uint8")
    mcx, mcy = cx - sx, cy - sy
    cv2.circle(mask, (mcx, mcy), radius, 255, -1)
    
    return cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=mask))

class Form40QuestionsTemplate(BaseOMRTemplate):
    name = "Phiếu 40 Câu hỏi"
    description = "Mẫu phiếu thi chuẩn gồm 4 cột, mỗi cột 10 câu. Tổng cộng 40 câu hỏi trắc nghiệm."
    preview_image_path = "resources/images/image.jpg" # Đường dẫn tới ảnh mẫu
    requires_alignment = False

    def calculate_confidence(self, blocks: list, image_shape: tuple = None) -> float:
        # Lý tưởng là 4 khối
        if len(blocks) == 4:
            return 1.0
        elif len(blocks) == 3:
            return 0.6
        elif len(blocks) == 2:
            return 0.3
        else:
            return 0.0

    def grade(self, image: np.ndarray, thresh: np.ndarray, blocks: list, answers: dict) -> tuple[np.ndarray, str, dict]:
        output_image = image.copy()
        
        options = ["A", "B", "C", "D"]
        results = {}
        score = 0
        total_questions_has_answer = 0
        
        wrong_list = []
        missing_list = []
        ruined_list = []

        # --- 2.5 CHỌN ĐÁP ÁN (DÙNG ĐỀ MẶC ĐỊNH VÌ MẪU NÀY KHÔNG CÓ MÃ ĐỀ) ---
        current_answers = answers
        if answers and isinstance(next(iter(answers.values()), None), dict):
            # Lấy mã đề đầu tiên làm mặc định vì mẫu này không đọc mã đề
            first_key = next(iter(answers.keys()))
            current_answers = answers[first_key]

        # Lấy số câu hỏi lớn nhất từ file đáp án
        max_q = 40
        if current_answers:
            try:
                max_q = max([int(k) for k in current_answers.keys() if k.isdigit()])
            except ValueError:
                pass

        # Duyệt qua từng khối (1-10, 11-20, 21-30, 31-40)
        for block_idx, (bx, by, bw, bh) in enumerate(blocks):
            # Thiết lập 10 hàng bằng toán học
            rows_y = [int(by + bh * 0.142 + i * (bh * 0.087)) for i in range(10)]
            
            # Thiết lập 4 cột bằng toán học
            cols_x = [int(bx + bw * 0.266 + j * (bw * 0.200)) for j in range(4)]
            
            # Tính toán bán kính và ngưỡng nhận diện động (Scale-Invariant)
            # Dựa theo kích thước thực tế của khối (block)
            radius = int(bw * 0.05) # Bán kính ô khoảng 5% chiều rộng khối
            if radius < 5: radius = 5
            
            circle_area = int(np.pi * (radius ** 2))
            min_pixels_threshold = int(circle_area * 0.4) # Phải tô đen ít nhất 40% diện tích ô

            # Quét từng câu hỏi
            for i, ry in enumerate(rows_y):
                q_num = block_idx * 10 + i + 1
                if q_num > max_q: break
                
                bubbled_list = []
                
                # Quét 4 đáp án A, B, C, D
                for j, cx in enumerate(cols_x):
                    # Đếm số pixel trắng trong vùng đó bằng hàm tối ưu
                    total = count_bubble_pixels(thresh, cx, ry, radius)
                    
                    if total > min_pixels_threshold:
                        bubbled_list.append(j)
                        
                if bubbled_list:
                    chosen_answer = ",".join([options[b] for b in bubbled_list])
                else:
                    chosen_answer = "Chưa chọn"
                    
                results[q_num] = chosen_answer
                
                # Chấm điểm (Tính trên tổng số câu có đáp án)
                correct_answer = current_answers.get(str(q_num))
                if correct_answer:
                    total_questions_has_answer += 1 # Luôn cộng vào tổng số câu dù có tô hay không
                    
                    if correct_answer == chosen_answer:
                        score += 1
                        color = (0, 255, 0) # Đúng -> Xanh lá
                        # Vẽ đáp án của thí sinh
                        for b in bubbled_list:
                            cv2.circle(output_image, (cols_x[b], ry), 28, color, 4)
                    else:
                        color = (0, 0, 255) # Sai -> Đỏ
                        expected_count = len(correct_answer.split(","))
                        
                        # Ghi nhận trạng thái lỗi
                        if len(bubbled_list) == 0:
                            missing_list.append(str(q_num))
                        elif len(bubbled_list) > expected_count:
                            ruined_list.append(str(q_num))
                        else:
                            wrong_list.append(str(q_num))
                            
                        # Chỉ vẽ vòng tròn đỏ nếu số ô tô hợp lệ (không bị phá/tô quá nhiều)
                        if 0 < len(bubbled_list) <= expected_count:
                            for b in bubbled_list:
                                cv2.circle(output_image, (cols_x[b], ry), 28, color, 4)

        # Tính điểm
        if total_questions_has_answer > 0:
            final_score = (score / total_questions_has_answer) * 10
        else:
            final_score = 0.0
        
        result_text = (
            f"--- KẾT QUẢ CHẤM ĐIỂM ---\n"
            f"Mẫu nhận diện: Form 40 Câu hỏi\n"
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
            'notes': " | ".join(notes_parts),
            'raw_answers': results
        }
        
        return output_image, result_text, score_dict
