import os
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from services.grading_service.omr_service import OMRService

class OMRSingleWorker(QThread):
    finished = pyqtSignal(object, str, dict)
    error = pyqtSignal(str)

    def __init__(self, image_data, answers, template, debug_dir, debug_prefix, ai_device='CPU'):
        super().__init__()
        self.image_data = image_data
        self.answers = answers
        self.template = template
        self.debug_dir = debug_dir
        self.debug_prefix = debug_prefix
        self.ai_device = ai_device

    def run(self):
        try:
            graded_image, result_text, score = OMRService.grade_image(
                self.image_data, self.answers, template=self.template,
                debug_dir=self.debug_dir, debug_prefix=self.debug_prefix, ai_device=self.ai_device
            )
            self.finished.emit(graded_image, result_text, score)
        except Exception as e:
            self.error.emit(str(e))


class OMRBatchWorker(QThread):
    progress = pyqtSignal(int, str)
    prescan_finished = pyqtSignal(dict, list) # report dict, valid_files list
    grading_finished = pyqtSignal(int, int, list, list) # success, error, excel_data, errors_log
    error = pyqtSignal(str)

    def __init__(self, mode, input_dir, files, template, answers=None, output_dir=None, logs_dir=None, ai_files=None, auto_ai=False, ai_device='CPU'):
        super().__init__()
        self.mode = mode # 'prescan' or 'grade'
        self.input_dir = input_dir
        self.files = files # Cho prescan là list tên file, cho grade là list tuple (file, template, use_ai)
        self.template = template
        self.answers = answers
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.ai_files = ai_files or []
        self.auto_ai = auto_ai
        self.ai_device = ai_device
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if self.mode == 'prescan':
                self._run_prescan()
            elif self.mode == 'grade':
                self._run_grade()
        except Exception as e:
            self.error.emit(str(e))

    def _run_prescan(self):
        report = {'total': len(self.files), 'valid': 0, 'errors': [], 'mades': {}}
        valid_files = []

        for i, file in enumerate(self.files):
            if self._is_cancelled:
                break
            
            self.progress.emit(i, f"Đang phân tích ảnh {i+1}/{len(self.files)}:\n{file}")
            
            input_path = os.path.join(self.input_dir, file)
            image = cv2.imread(input_path)
            
            base_name, _ = os.path.splitext(file)
            use_ai = file in self.ai_files
            ai_level_used = 0
            
            # Tự động dùng AI nếu được check và có gợi ý
            scan_res = OMRService.pre_scan_image(image, template=self.template, ai_level=0, debug_prefix=f"{base_name}_", debug_dir=self.logs_dir, ai_device=self.ai_device)
            
            if not scan_res['is_valid'] and self.auto_ai:
                ai_level_used = 2
                self.progress.emit(i, f"Ảnh {i+1}/{len(self.files)} mờ/lệch.\nĐang dùng Khôi phục nét AI ({self.ai_device}):\n{file}")
                scan_res = OMRService.pre_scan_image(image, template=self.template, ai_level=2, debug_prefix=f"{base_name}_", debug_dir=self.logs_dir, ai_device=self.ai_device)

            if scan_res['is_valid']:
                report['valid'] += 1
                made = scan_res['made']
                if made:
                    report['mades'][made] = report['mades'].get(made, 0) + 1
                
                temp_path = None
                if 'context_cache' in scan_res:
                    import tempfile
                    import pickle
                    import uuid
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, f"omr_cache_{uuid.uuid4().hex}.pkl")
                    try:
                        with open(temp_path, "wb") as f:
                            pickle.dump(scan_res['context_cache'], f)
                    except Exception:
                        temp_path = None

                valid_files.append((file, scan_res['template'], ai_level_used, temp_path))
            else:
                report['errors'].append({'file': file, 'reason': scan_res['error'], 'sbd': scan_res.get('sbd', ''), 'made': scan_res.get('made', '')})
                
        if not self._is_cancelled:
            self.progress.emit(len(self.files), "Hoàn tất tiền xử lý.")
            self.prescan_finished.emit(report, valid_files)

    def _run_grade(self):
        success_count = 0
        error_count = 0
        excel_data = []
        errors_log = []

        for i, item in enumerate(self.files):
            file = item[0]
            current_template = item[1]
            ai_level_used = item[2]
            temp_path = item[3] if len(item) > 3 else None

            if self._is_cancelled:
                break
                
            self.progress.emit(i, f"Đang chấm bài {i+1}/{len(self.files)}:\n{file}")
            
            input_path = os.path.join(self.input_dir, file)
            image = cv2.imread(input_path)
            
            base_name, ext = os.path.splitext(file)
            
            cached_context = None
            if temp_path and os.path.exists(temp_path):
                import pickle
                try:
                    with open(temp_path, "rb") as f:
                        cached_context = pickle.load(f)
                except Exception:
                    pass
                finally:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
            
            try:
                graded_image, _, score = OMRService.grade_image(
                    image, self.answers, template=current_template,
                    debug_dir=self.logs_dir, debug_prefix=f"{base_name}_", ai_level=ai_level_used,
                    cached_context=cached_context, ai_device=self.ai_device
                )
                
                output_name = f"graded_{base_name}{ext}"
                output_path = os.path.join(self.output_dir, output_name)
                cv2.imwrite(output_path, graded_image)
                
                sbd = score.get('sbd', 'N/A')
                made = score.get('made', 'N/A')
                
                excel_data.append({
                    "Tên file": file,
                    "Số báo danh": sbd,
                    "Mã đề thi": made,
                    "Số câu đúng": f"{score['correct']}/{score['total']}",
                    "Điểm": round(score['final_score'], 2),
                    "Ghi chú": score.get('notes', '')
                })
                success_count += 1
            except Exception as e:
                error_count += 1
                error_msg = str(e)
                errors_log.append(f"• {file}: {error_msg}")
                # Đưa bài bị lỗi vào danh sách xuất Excel để báo cáo cho người dùng
                excel_data.append({
                    "Tên file": file,
                    "Số báo danh": "LỖI",
                    "Mã đề thi": "LỖI",
                    "Số câu đúng": "0/0",
                    "Điểm": 0,
                    "Ghi chú": f"LỖI CHẤM BÀI: {error_msg}"
                })
                
        # Dọn dẹp cache còn sót lại nếu tiến trình bị hủy
        if self._is_cancelled:
            for item in self.files:
                if len(item) > 3 and item[3] and os.path.exists(item[3]):
                    try:
                        os.remove(item[3])
                    except Exception:
                        pass
                
        if not self._is_cancelled:
            self.progress.emit(len(self.files), "Hoàn tất chấm thi.")
            self.grading_finished.emit(success_count, error_count, excel_data, errors_log)
