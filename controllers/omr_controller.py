from PyQt6.QtWidgets import QMessageBox, QFileDialog, QDialog, QProgressDialog, QApplication
from PyQt6.QtCore import Qt
from services.grading_service.omr_service import OMRService
from services.grading_service.answer_key_service import AnswerKeyService
from services.grading_service.report_export_service import ReportExportService
from ui.dialogs.template_selection_dialog import TemplateSelectionDialog
from ui.dialogs.answer_key_dialog import AnswerKeyDialog
from ui.dialogs.prescan_report_dialog import PreScanReportDialog
from ui.dialogs.batch_complete_dialog import BatchCompleteDialog
from controllers.workers.omr_worker import OMRBatchWorker
from image_processing.alignment.document_aligner import clear_ai_cache
import cv2
import os

class OMRController:
    def __init__(self, main_window, image_state):
        self.view = main_window
        self.image_state = image_state
        self.active_worker = None

    def _set_ui_locked(self, locked: bool):
        """Khóa/Mở khóa giao diện khi đang chạy luồng ngầm."""
        self.view.toolbar_component.setEnabled(not locked)
        self.view.left_panel.setEnabled(not locked)

    def open_answer_key_dialog(self):
        """Mở hộp thoại thiết lập đáp án OMR."""
        from PyQt6.QtWidgets import QInputDialog
        q_count, ok = QInputDialog.getInt(
            self.view, "Khởi tạo bảng đáp án", 
            "Nhập số lượng câu hỏi cho đề thi:", 
            40, 1, 200
        )
        if not ok:
            return # Người dùng nhấn Cancel
            
        dialog = AnswerKeyDialog(self.view, initial_q_count=q_count)
        dialog.exec()

    def _select_answer_key_file(self):
        """Mở hộp thoại chọn file Excel đáp án. Trả về đường dẫn file hoặc None nếu hủy."""
        default_dir = os.path.join(os.getcwd(), "assets")
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()
            
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Chọn file Excel Đáp án", default_dir, "Excel Files (*.xlsx *.xls)"
        )
        return path if path else None

    def _select_template(self, sample_image=None):
        """
        Hiển thị hộp thoại chọn mẫu. 
        Nếu sample_image có dữ liệu và người dùng chọn Auto, nó sẽ tính điểm tự tin.
        Nếu điểm < 0.8, nó bắt chọn thủ công.
        Trả về template_obj hoặc None.
        """
        templates = OMRService.get_all_templates()
        dialog = TemplateSelectionDialog(templates, self.view)
        
        while True:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
                
            if dialog.is_auto_mode and sample_image is not None:
                # Chạy detect
                detected = OMRService.detect_templates(sample_image)
                best_conf = detected[0][1] if detected else 0.0
                
                if best_conf >= 0.8:
                    return detected[0][0]
                else:
                    QMessageBox.warning(self.view, "Độ tự tin thấp", 
                        f"Độ tự tin cao nhất chỉ đạt {int(best_conf*100)}%.\nVui lòng chọn mẫu thủ công từ danh sách.")
                    # Buộc chuyển sang manual mode và hiện danh sách
                    dialog.force_manual_mode(detected)
                    continue # Bắt người dùng chọn lại
            else:
                if dialog.selected_template == "AUTO":
                    # Không có sample_image (ví dụ batch chưa tải ảnh), bỏ qua kiểm tra tự tin tạm thời,
                    # hoặc báo lỗi. Trong batch ta có thể kiểm tra ở ảnh đầu tiên, nhưng để đơn giản ta bắt 
                    # nó return "AUTO" để xử lý bên ngoài.
                    return "AUTO"
                if dialog.selected_template is None:
                    QMessageBox.warning(self.view, "Lỗi", "Vui lòng chọn một mẫu!")
                    continue
                return dialog.selected_template

    def handle_omr_batch(self):
        from ui.dialogs.batch_setup_dialog import BatchSetupDialog
        
        dialog = BatchSetupDialog(self.view)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        input_dir = dialog.input_dir
        output_dir = dialog.output_dir
        answer_key_path = dialog.answer_key_path
        template = dialog.selected_template
        
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
        
        if not files:
            QMessageBox.information(self.view, "Thông báo", "Không tìm thấy ảnh nào trong thư mục đầu vào.")
            return

        answers = AnswerKeyService.load_answers(answer_key_path)
        if not answers:
            QMessageBox.warning(self.view, "Cảnh báo", "Không tìm thấy dữ liệu đáp án hoặc file rỗng.")
            return

        auto_ai = dialog.chk_use_ai.isChecked()
        ai_device = 'GPU' if dialog.combo_device.currentIndex() == 1 else 'CPU'

        # --- BƯỚC 1: TIỀN XỬ LÝ (PRE-SCAN) ---
        self._set_ui_locked(True)
        clear_ai_cache()
        progress = QProgressDialog("Đang xử lý phân tích ảnh...", "Hủy", 0, len(files), self.view)
        progress.setWindowTitle("Tiền xử lý")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        self.active_worker = OMRBatchWorker('prescan', input_dir, files, template, output_dir=output_dir, logs_dir=logs_dir, auto_ai=auto_ai, ai_device=ai_device)

        def on_prescan_progress(val, text):
            progress.setValue(val)
            progress.setLabelText(text)

        def on_prescan_error(err_str):
            progress.close()
            self._set_ui_locked(False)
            QMessageBox.warning(self.view, "Lỗi", f"Lỗi tiền xử lý: {err_str}")

        def on_prescan_cancel():
            if self.active_worker and self.active_worker.isRunning():
                self.active_worker.cancel()
            clear_ai_cache()
            self._set_ui_locked(False)

        progress.canceled.connect(on_prescan_cancel)

        def on_prescan_finished(report, valid_files):
            progress.close()
            self._set_ui_locked(False)
            
            if self.active_worker and getattr(self.active_worker, '_is_cancelled', False):
                return
                
            needs_ai_files = [err['file'] for err in report['errors'] if "Gợi ý dùng AI" in err['reason']]
            
            if needs_ai_files:
                reply = QMessageBox.question(
                    self.view, "Phát hiện ảnh lỗi", 
                    f"Có {len(needs_ai_files)} ảnh bị lỗi nhận diện và có độ phân giải thấp.\n"
                    "Bạn có muốn dùng AI (Super Resolution) để quét lại các ảnh này không?\n"
                    "(Quá trình này sẽ tốn thêm một chút thời gian)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._start_ai_prescan(input_dir, needs_ai_files, template, report, valid_files, output_dir, answers)
                    return
            
            # Khối lệnh này chạy nếu chọn No HOẶC không có ảnh nào cần AI
            # Cứu các file chỉ bị lỗi SBD/Mã đề (không bị lỗi rách góc)
            saved_errors = []
            for err in report['errors']:
                if "Không đọc được SBD/Mã đề" in err['reason'] and not any(k in err['reason'] for k in ["bắt góc", "Không tìm thấy", "Khối"]):
                    valid_files.append((err['file'], template, False))
                    report['valid'] += 1
                else:
                    saved_errors.append(err)
            report['errors'] = saved_errors
            
            self._finish_prescan_and_show_report(report, valid_files, input_dir, output_dir, answers)

        self.active_worker.progress.connect(on_prescan_progress)
        self.active_worker.error.connect(on_prescan_error)
        self.active_worker.prescan_finished.connect(on_prescan_finished)
        self.active_worker.start()

    def _start_ai_prescan(self, input_dir, failed_files, template, prev_report, prev_valid_files, output_dir, answers):
        self._set_ui_locked(True)
        progress = QProgressDialog("Đang dùng AI quét lại ảnh lỗi...", "Hủy", 0, len(failed_files), self.view)
        progress.setWindowTitle("Tiền xử lý AI")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.active_worker = OMRBatchWorker('prescan', input_dir, failed_files, template, ai_files=failed_files, output_dir=output_dir, logs_dir=logs_dir)

        def on_prescan_progress(val, text):
            progress.setValue(val)
            progress.setLabelText(text)

        def on_prescan_error(err_str):
            progress.close()
            self._set_ui_locked(False)
            QMessageBox.warning(self.view, "Lỗi", f"Lỗi tiền xử lý AI: {err_str}")

        def on_prescan_cancel():
            if self.active_worker and self.active_worker.isRunning():
                self.active_worker.cancel()
            clear_ai_cache()
            self._set_ui_locked(False)

        progress.canceled.connect(on_prescan_cancel)

        def on_ai_prescan_finished(ai_report, ai_valid_files):
            progress.close()
            self._set_ui_locked(False)
            
            if self.active_worker and getattr(self.active_worker, '_is_cancelled', False):
                return
                
            # Cứu các file chỉ bị lỗi SBD ở vòng 1 nhưng vòng AI vẫn thất bại (hoặc AI làm hỏng ảnh)
            final_errors = []
            for ai_err in ai_report['errors']:
                file = ai_err['file']
                # Tìm lỗi gốc ở vòng 1
                orig_err = next((e for e in prev_report['errors'] if e['file'] == file), None)
                if orig_err and "Không đọc được SBD/Mã đề" in orig_err['reason'] and not any(k in orig_err['reason'] for k in ["bắt góc", "Không tìm thấy", "Khối"]):
                    # File này bản chất vẫn chấm được (chỉ sai SBD). Bỏ qua kết quả AI, dùng ảnh gốc!
                    ai_valid_files.append((file, template, False))
                    ai_report['valid'] += 1
                else:
                    final_errors.append(ai_err)
                    
            prev_report['valid'] += ai_report['valid']
            for k, v in ai_report['mades'].items():
                prev_report['mades'][k] = prev_report['mades'].get(k, 0) + v
            
            prev_report['errors'] = final_errors
            prev_valid_files.extend(ai_valid_files)
            
            self._finish_prescan_and_show_report(prev_report, prev_valid_files, input_dir, output_dir, answers)

        self.active_worker.progress.connect(on_prescan_progress)
        self.active_worker.error.connect(on_prescan_error)
        self.active_worker.prescan_finished.connect(on_ai_prescan_finished)
        self.active_worker.start()

    def _finish_prescan_and_show_report(self, report, valid_files, input_dir, output_dir, answers):
            
        from ui.dialogs.prescan_report_dialog import PreScanReportDialog
        report_dialog = PreScanReportDialog(self.view, report)
        if report_dialog.exec() == QDialog.DialogCode.Accepted:
            self._start_batch_grading(input_dir, valid_files, report['errors'], output_dir, answers)

    def _start_batch_grading(self, input_dir, valid_files, ignored_errors, output_dir, answers):
        # --- BƯỚC 2: CHẤM THI CHÍNH THỨC ---
        logs_dir = os.path.join(output_dir, "logs")
        if not os.path.exists(logs_dir): os.makedirs(logs_dir)
            
        images_dir = os.path.join(output_dir, "anh_da_cham")
        if not os.path.exists(images_dir): os.makedirs(images_dir)
            
        excel_path = os.path.join(output_dir, "tong_hop_diem.xlsx")

        self._set_ui_locked(True)
        progress = QProgressDialog("Đang chấm thi...", "Dừng", 0, len(valid_files), self.view)
        progress.setWindowTitle("Tiến trình chấm thi")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Dashboard init
        if hasattr(self.view, 'right_panel') and hasattr(self.view.right_panel, 'update_dashboard'):
            self.view.right_panel.update_dashboard({'total': 0, 'avg': 0, 'gioi': 0, 'kha': 0, 'tb': 0, 'yeu': 0})

        self.active_worker = OMRBatchWorker('grade', input_dir, valid_files, None, answers, images_dir, logs_dir)

        def on_grade_progress(val, text):
            progress.setValue(val)
            progress.setLabelText(text)

        def on_grade_error(err_str):
            progress.close()
            self._set_ui_locked(False)
            QMessageBox.warning(self.view, "Lỗi", f"Lỗi khi chấm: {err_str}")

        def on_grading_cancel():
            if self.active_worker and self.active_worker.isRunning():
                self.active_worker.cancel()
            clear_ai_cache()
            self._set_ui_locked(False)

        progress.canceled.connect(on_grading_cancel)

        def on_grading_finished(success_count, error_count, excel_data, errors_log):
            progress.close()
            clear_ai_cache()
            self._set_ui_locked(False)
            
            if self.active_worker and getattr(self.active_worker, '_is_cancelled', False):
                QMessageBox.information(self.view, "Đã hủy", "Tiến trình đã dừng giữa chừng. Một phần kết quả đã được lưu.")

            # Update final dashboard
            if excel_data is not None:
                # Include prescan errors in the final errors log and excel data
                for err in ignored_errors:
                    errors_log.append(f"{err['file']}: Lỗi tiền xử lý - {err['reason']}")
                    excel_data.append({
                        "Tên file": err['file'],
                        "Số báo danh": "N/A",
                        "Mã đề thi": "N/A",
                        "Số câu đúng": "0/0",
                        "Điểm": 0,
                        "Ghi chú": f"Lỗi tiền xử lý: {err['reason']}"
                    })
                
                from services.grading_service.report_export_service import ReportExportService
                ReportExportService.export_batch_results(excel_data, excel_path)
                
                # Tự động nạp thư mục Output vào Left Panel
                if hasattr(self.view, 'controller') and hasattr(self.view.controller, 'explorer'):
                    self.view.controller.explorer.import_folder_by_path(output_dir)
                
                gioi = kha = tb = yeu = 0
                total_score = 0
                for row in excel_data:
                    sc = row.get("Điểm", 0)
                    total_score += sc
                    if sc >= 8: gioi += 1
                    elif sc >= 6.5: kha += 1
                    elif sc >= 5: tb += 1
                    else: yeu += 1
                
                avg_score = total_score / len(excel_data) if len(excel_data) > 0 else 0
                stats = {'total': success_count + error_count, 'avg': avg_score, 'gioi': gioi, 'kha': kha, 'tb': tb, 'yeu': yeu}
                if hasattr(self.view, 'right_panel') and hasattr(self.view.right_panel, 'update_dashboard'):
                    self.view.right_panel.update_dashboard(stats)

            summary_data = {
                'success': success_count,
                'error': error_count,
                'output_dir': output_dir,
                'excel_path': excel_path,
                'excel_data': excel_data,
                'errors_log': errors_log
            }
            
            dialog = BatchCompleteDialog(self.view, summary_data)
            dialog.exec()

        self.active_worker.progress.connect(on_grade_progress)
        self.active_worker.error.connect(on_grade_error)
        self.active_worker.grading_finished.connect(on_grading_finished)
        self.active_worker.start()
