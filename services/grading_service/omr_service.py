"""
OMR Service — Lớp Facade mỏng điều phối toàn bộ quy trình chấm thi.
Thực thi như một Pipeline Executor dựa trên Data-Driven Configuration (JSON) theo chuẩn DAG Node Engine.
"""
import numpy as np
import cv2
from services.grading_service.template_service import TemplateService
from services.grading_service.nodes.engine import NodeEngine
from services.grading_service.nodes import NODE_CLASS_MAPPINGS
from services.grading_service.nodes.cv_nodes import ONNXRestorationNode

class OMRService:
    
    _engine = NodeEngine(NODE_CLASS_MAPPINGS)

    @staticmethod
    def get_all_templates() -> list:
        return TemplateService.get_all_templates()

    @staticmethod
    def grade_image(image: np.ndarray, answers: dict, template=None, debug_dir=None, debug_prefix="", ai_level=0, cached_context=None, ai_device='CPU') -> tuple[np.ndarray, str, dict]:
        if image is None:
            return image, "Không có ảnh hợp lệ.", {'correct': 0, 'total': 0, 'final_score': 0}
        
        # TỰ ĐỘNG THU NHỎ ẢNH ĐỂ TĂNG TỐC ĐỘ CHẤM (MAX HEIGHT = 1600)
        h, w = image.shape[:2]
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        if not cached_context:
            if ai_level == 2:
                restore_out = ONNXRestorationNode().execute(image, model_path="assets/models/realesr-general-x4v3.onnx", max_width_before_inference=800, ai_device=ai_device)
                image = restore_out.get("image", image)
            
        if template is None or isinstance(template, str):
            templates = OMRService.get_all_templates()
            if not templates:
                return image, "Chưa có mẫu nào trong hệ thống.", {'correct': 0, 'total': 0, 'final_score': 0}
            template = templates[0]

        nodes_config = template.get("nodes", [])
        
        # Thiết lập bối cảnh ban đầu (Global inputs)
        initial_context = cached_context.copy() if cached_context else {}
        initial_context["input"] = {
            "image": image,
            "answers": answers,
            "debug_dir": debug_dir,
            "debug_prefix": debug_prefix,
            "use_ai": ai_level > 0
        }
        
        try:
            # Chạy toàn bộ đồ thị (DAG) tuần tự
            final_context = OMRService._engine.execute_pipeline(nodes_config, initial_context)
            
            # Trích xuất kết quả từ output của các Node
            # Do thiết kế ta biết các node cuối có tên là score_q, vis_q, read_sbd, v.v.
            # Tuy nhiên, để linh hoạt, ta rà soát context
            
            score_data = final_context.get("score_q", {})
            sbd_data = final_context.get("read_sbd", {})
            made_data = final_context.get("read_made", {})
            vis_data = final_context.get("vis_q", {})
            
            score = score_data.get("score", 0)
            total_questions = score_data.get("total_q", 0)
            final_score = (score / total_questions) * 10 if total_questions > 0 else 0.0
            
            extracted_sbd = sbd_data.get("result", "")
            extracted_made = made_data.get("result", "")
            
            all_errors = []
            for node_id, node_out in final_context.items():
                if isinstance(node_out, dict) and node_out.get("error"):
                    all_errors.append(node_out["error"])
            
            all_notes = []
            if score_data.get("wrong_list"): all_notes.append(f"Sai: {','.join(score_data['wrong_list'])}")
            if score_data.get("missing_list"): all_notes.append(f"Trống: {','.join(score_data['missing_list'])}")
            if score_data.get("ruined_list"): all_notes.append(f"Phá: {','.join(score_data['ruined_list'])}")
            
            all_notes.extend(all_errors)
            
            sbd_str = extracted_sbd if extracted_sbd else "Không đọc được"
            made_str = extracted_made if extracted_made else "Không đọc được"
            
            result_text = (
                f"--- KẾT QUẢ CHẤM ĐIỂM ---\n"
                f"Mẫu: {template.get('name', 'Unknown')}\n"
                f"Số báo danh: {sbd_str}\n"
                f"Mã đề thi: {made_str}\n"
                f"Số câu đúng: {score}/{total_questions}\n"
                f"Điểm số: {final_score:.2f} / 10\n"
            )
            
            score_dict = {
                'correct': score,
                'total': total_questions,
                'final_score': final_score,
                'sbd': extracted_sbd,
                'made': extracted_made,
                'notes': " | ".join(all_notes),
                'raw_answers': score_data.get("results", {})
            }
            
            # Ảnh đã được vẽ bởi node cuối cùng trong chuỗi
            proc_img = vis_data.get("image", image)
            
            return proc_img, result_text, score_dict
            
        except Exception as e:
            return image, f"Lỗi thực thi Pipeline: {str(e)}", {'correct': 0, 'total': 0, 'final_score': 0}

    @staticmethod
    def pre_scan_image(image: np.ndarray, template=None, ai_level=0, debug_prefix="", debug_dir=None, ai_device='CPU') -> dict:
        """Quét nhanh bằng Pipeline. Rút gọn các Node không cần thiết để chạy cực nhanh."""
        if image is None:
            return {'is_valid': False, 'error': 'Ảnh không tồn tại', 'template': None}
            
        # Reject horizontal images to fail fast
        h, w = image.shape[:2]
        if w > h:
            return {'is_valid': False, 'error': 'Lỗi: Ảnh nằm ngang. Yêu cầu chụp dọc.', 'template': None}
            
        # TỰ ĐỘNG THU NHỎ ẢNH ĐỂ TĂNG TỐC ĐỘ CHẤM (MAX HEIGHT = 1600)
        # Giúp đồng bộ kích thước với hàm grade_image để các hệ số (kernel size, roi size) hoạt động chuẩn xác
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        ai_error = ""
        if ai_level == 2:
            restore_out = ONNXRestorationNode().execute(image, model_path="assets/models/realesr-general-x4v3.onnx", max_width_before_inference=800, ai_device=ai_device)
            image = restore_out.get("image", image)
            ai_error = restore_out.get("error", "")

        selected_template = template
        if selected_template is None or isinstance(selected_template, str):
            templates = OMRService.get_all_templates()
            if not templates: return {'is_valid': False, 'error': 'Chưa có template'}
            selected_template = templates[0]
            
        try:
            # Chỉ lọc ra các Node phục vụ trích xuất ảnh và SBD/Made (bỏ qua chấm điểm, vẽ vời)
            full_nodes = selected_template.get("nodes", [])
            prescan_nodes = []
            for n in full_nodes:
                if n.get("type") in ["ImageStandardizer", "DocumentAligner", "AdaptiveThreshold", "BlockExtractor", "Heuristic120", "Heuristic40", "SBDReader"]:
                    prescan_nodes.append(n)
                    
            initial_context = {
                "input": { "image": image, "answers": {}, "debug_dir": debug_dir, "debug_prefix": debug_prefix, "use_ai": ai_level > 0 }
            }
            
            final_context = OMRService._engine.execute_pipeline(prescan_nodes, initial_context)
            
            sbd = final_context.get("read_sbd", {}).get("result", "")
            made = final_context.get("read_made", {}).get("result", "")
            
            all_errors = []
            for node_id, node_out in final_context.items():
                if isinstance(node_out, dict) and node_out.get("error"):
                    all_errors.append(node_out["error"])
            
            # Luôn bắt lỗi SBD/Mã đề ở bất kỳ level nào (để lên báo cáo)
            if "?" in sbd or not sbd:
                all_errors.append("Lỗi đọc Số báo danh")
            if "?" in made or not made:
                all_errors.append("Lỗi đọc Mã đề thi")
                
            # Kiểm tra mã đề có trong file Excel không
            answers = initial_context["input"].get("answers", {})
            if made and "?" not in made and answers:
                if isinstance(next(iter(answers.values()), None), dict): # Dạng nhiều mã đề
                    if made not in answers:
                        all_errors.append(f"Không tìm thấy mã đề '{made}' trong đáp án")
            
            total_pixels = image.shape[0] * image.shape[1]
                
            if all_errors and ai_level == 0 and total_pixels < 1000000:
                all_errors.append("Gợi ý dùng AI")
            
            error = " | ".join(all_errors)
            if ai_error:
                error = f"AI Error: {ai_error} | {error}"
            
            if all_errors or ai_error:
                return {'is_valid': False, 'error': error, 'template': selected_template, 'sbd': sbd, 'made': made}
            
            return {
                'is_valid': True,
                'error': error, 
                'template': selected_template,
                'sbd': sbd,
                'made': made,
                'context_cache': final_context
            }
        except Exception as e:
            return {'is_valid': False, 'error': f'Lỗi hệ thống: {str(e)[:50]}', 'template': selected_template}
