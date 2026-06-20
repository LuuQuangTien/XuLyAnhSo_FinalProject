import os
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from services.omr_service import OMRService

class OMRSingleWorker(QThread):
    finished = pyqtSignal(object, str, dict)
    error = pyqtSignal(str)

    def __init__(self, image_data, answers, template, debug_dir, debug_prefix):
        super().__init__()
        self.image_data = image_data
        self.answers = answers
        self.template = template
        self.debug_dir = debug_dir
        self.debug_prefix = debug_prefix

    def run(self):
        try:
            graded_image, result_text, score = OMRService.grade_image(
                self.image_data, self.answers, template=self.template,
                debug_dir=self.debug_dir, debug_prefix=self.debug_prefix
            )
            self.finished.emit(graded_image, result_text, score)
        except Exception as e:
            self.error.emit(str(e))


class OMRBatchWorker(QThread):
    progress = pyqtSignal(int, str)
    prescan_finished = pyqtSignal(dict, list) # report dict, valid_files list
    grading_finished = pyqtSignal(int, int, list, list) # success, error, excel_data, errors_log
    error = pyqtSignal(str)

    def __init__(self, mode, input_dir, files, template, answers=None, output_dir=None, logs_dir=None):
        super().__init__()
        self.mode = mode # 'prescan' or 'grade'
        self.input_dir = input_dir
        self.files = files # Cho prescan là list tên file, cho grade là list tuple (file, template)
        self.template = template
        self.answers = answers
        self.output_dir = output_dir
        self.logs_dir = logs_dir
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
            
            scan_res = OMRService.pre_scan_image(image, template=self.template)
            if scan_res['is_valid']:
                report['valid'] += 1
                made = scan_res['made']
                if made:
                    report['mades'][made] = report['mades'].get(made, 0) + 1
                valid_files.append((file, scan_res['template']))
            else:
                report['errors'].append({'file': file, 'reason': scan_res['error']})
                
        if not self._is_cancelled:
            self.progress.emit(len(self.files), "Hoàn tất tiền xử lý.")
            self.prescan_finished.emit(report, valid_files)

    def _run_grade(self):
        success_count = 0
        error_count = 0
        excel_data = []
        errors_log = []

        for i, (file, current_template) in enumerate(self.files):
            if self._is_cancelled:
                break
                
            self.progress.emit(i, f"Đang chấm bài {i+1}/{len(self.files)}:\n{file}")
            
            input_path = os.path.join(self.input_dir, file)
            image = cv2.imread(input_path)
            
            base_name, ext = os.path.splitext(file)
            
            try:
                graded_image, _, score = OMRService.grade_image(
                    image, self.answers, template=current_template,
                    debug_dir=self.logs_dir, debug_prefix=f"{base_name}_"
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
                
        if not self._is_cancelled:
            self.progress.emit(len(self.files), "Hoàn tất chấm thi.")
            self.grading_finished.emit(success_count, error_count, excel_data, errors_log)
