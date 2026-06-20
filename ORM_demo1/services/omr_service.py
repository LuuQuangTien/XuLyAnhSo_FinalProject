import cv2
import numpy as np
import os
from services.omr_templates.form_40_questions import Form40QuestionsTemplate
from services.omr_templates.form_120_questions import Form120QuestionsTemplate
import services.omr_preprocessor as preprocessor

class OMRService:
    # Đăng ký các template khả dụng
    _registered_templates = [
        Form40QuestionsTemplate(),
        Form120QuestionsTemplate()
    ]

    @staticmethod
    def get_all_templates() -> list:
        return OMRService._registered_templates

    @staticmethod
    def detect_templates(image: np.ndarray) -> list[tuple[any, float]]:
        """
        Tính toán độ tự tin cho tất cả các templates đã đăng ký.
        Trả về danh sách các tuple (template, confidence_score), sắp xếp giảm dần.
        """
        if image is None:
            return []
            
        # Tính toán trên ảnh gốc (không nắn)
        gray_unaligned = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_unaligned = preprocessor.apply_adaptive_threshold(gray_unaligned)
        blocks_unaligned = preprocessor.find_blocks(thresh_unaligned)
        
        # Tính toán trên ảnh đã nắn (chỉ dành cho mẫu yêu cầu)
        aligned_img = preprocessor.align_document(image)
        gray_aligned = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        thresh_aligned = preprocessor.apply_adaptive_threshold(gray_aligned)
        blocks_aligned = preprocessor.find_blocks(thresh_aligned)

        results = []
        for template in OMRService._registered_templates:
            if getattr(template, 'requires_alignment', False):
                conf = template.calculate_confidence(blocks_aligned, aligned_img.shape)
            else:
                conf = template.calculate_confidence(blocks_unaligned, image.shape)
            results.append((template, conf))
            
        # Sắp xếp giảm dần theo độ tự tin
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def grade_image(image: np.ndarray, answers: dict, template=None, debug_dir=None, debug_prefix="") -> tuple[np.ndarray, str, dict]:
        """
        Chấm điểm OMR sử dụng một template cụ thể hoặc template tự động nhận diện.
        """
        if image is None:
            return image, "Không có ảnh hợp lệ.", {'correct': 0, 'total': 0, 'final_score': 0}
            
        # Phân luồng xử lý Template
        selected_template = template
        if selected_template is None:
            # Tự động chọn template tốt nhất nếu không chỉ định
            detected = OMRService.detect_templates(image)
            if detected and detected[0][1] >= 0.8:
                selected_template = detected[0][0]

        if selected_template is None:
            return image, f"Lỗi: Không nhận diện được mẫu phiếu thi hợp lệ.", {'correct': 0, 'total': 0, 'final_score': 0}

        # Áp dụng nắn thẳng NẾU template yêu cầu
        if getattr(selected_template, 'requires_alignment', False):
            proc_img = preprocessor.align_document(image, debug_dir, debug_prefix)
        else:
            proc_img = image.copy()
            
        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        thresh = preprocessor.apply_adaptive_threshold(gray)
        blocks = preprocessor.find_blocks(thresh, debug_dir, debug_prefix, proc_img)

        # Giao việc chấm điểm cho Template xử lý (Truyền ảnh đã tiền xử lý)
        return selected_template.grade(proc_img, thresh, blocks, answers)

    @staticmethod
    def pre_scan_image(image: np.ndarray, template=None) -> dict:
        """
        Quét nhanh để kiểm tra tính hợp lệ và lấy metadata (SBD, Mã đề)
        Trả về dict: {'is_valid': bool, 'error': str, 'template': obj, 'sbd': str, 'made': str}
        """
        if image is None:
            return {'is_valid': False, 'error': 'Ảnh không tồn tại hoặc bị hỏng', 'template': None}
            
        selected_template = template
        if selected_template is None or selected_template == "AUTO":
            detected = OMRService.detect_templates(image)
            # Ngưỡng tự tin 0.6 là tạm đủ để xếp loại trong lúc pre-scan
            if not detected or detected[0][1] < 0.6: 
                return {'is_valid': False, 'error': 'Không nhận diện được mẫu phiếu hoặc ảnh quá mờ/mất góc', 'template': None}
            selected_template = detected[0][0]
            
        try:
            if getattr(selected_template, 'requires_alignment', False):
                # Nắn thẳng nhẹ không cần debug
                proc_img = preprocessor.align_document(image)
            else:
                proc_img = image.copy()
                
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            thresh = preprocessor.apply_adaptive_threshold(gray)
            blocks = preprocessor.find_blocks(thresh)
            
            # Kiểm tra cơ bản
            if len(blocks) < 2:
                return {'is_valid': False, 'error': 'Không tìm thấy đủ các khối câu hỏi / định vị', 'template': selected_template}
                
            # Chạy thử grade với đáp án rỗng để lấy dữ liệu metadata
            _, _, score = selected_template.grade(proc_img, thresh, blocks, {})
            
            made = score.get('made', '')
            sbd = score.get('sbd', '')
            
            # Làm sạch chuỗi
            if made == 'Không đọc được': made = ''
            if sbd == 'Không đọc được': sbd = ''
            
            return {
                'is_valid': True, 
                'error': '', 
                'template': selected_template,
                'sbd': sbd,
                'made': made
            }
        except Exception as e:
            return {'is_valid': False, 'error': f'Lỗi hệ thống: {str(e)[:50]}', 'template': selected_template}

