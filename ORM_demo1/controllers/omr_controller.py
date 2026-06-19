from PyQt6.QtWidgets import QMessageBox, QFileDialog, QDialog, QProgressDialog, QApplication
from PyQt6.QtCore import Qt
from services.omr_service import OMRService
from services.answer_key_service import AnswerKeyService
from services.report_export_service import ReportExportService
from ui.dialogs.template_selection_dialog import TemplateSelectionDialog
from ui.dialogs.answer_key_dialog import AnswerKeyDialog
from ui.dialogs.prescan_report_dialog import PreScanReportDialog
from ui.dialogs.batch_complete_dialog import BatchCompleteDialog
from controllers.workers.omr_worker import OMRSingleWorker, OMRBatchWorker
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

    def handle_omr_processing(self):
        state = self.image_state.get_active_state()
        if not state or not state.has_image():
            QMessageBox.warning(self.view, "Cảnh báo", "Vui lòng mở một ảnh để chấm thi OMR.")
            return

        image_data = state.current_processed_image
        
        # Chọn mẫu
        template = self._select_template(sample_image=image_data)
        if template is None:
            return # User cancelled

        # Chọn file đáp án
        answer_key_path = self._select_answer_key_file()
        if not answer_key_path:
            return # User cancelled

        # Gọi OMRService
        answers = AnswerKeyService.load_answers(answer_key_path)
        if not answers:
            QMessageBox.warning(self.view, "Lỗi", "File đáp án rỗng hoặc không hợp lệ.")
            return

        reply = QMessageBox.question(self.view, "Lưu kết quả", "Bạn có muốn chọn thư mục để xuất báo cáo Excel và lưu ảnh không?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        output_dir = None
        logs_dir = None
        base_name = "current_image"
        if state.current_image_path:
            base_name = os.path.splitext(os.path.basename(state.current_image_path))[0]
        
        if reply == QMessageBox.StandardButton.Yes:
            default_out = os.path.join(os.getcwd(), "resources", "ketqua")
            os.makedirs(default_out, exist_ok=True)
            output_dir = QFileDialog.getExistingDirectory(self.view, "Chọn thư mục lưu ảnh kết quả và Excel", default_out)
            if output_dir:
                logs_dir = os.path.join(output_dir, "logs")
                os.makedirs(logs_dir, exist_ok=True)
        else:
            # Vẫn lưu log mặc định để debug
            default_log = os.path.join(os.getcwd(), "resources", "ketqua", "logs")
            os.makedirs(default_log, exist_ok=True)
            logs_dir = default_log

        # Khóa giao diện và tạo tiến trình
        self._set_ui_locked(True)
        
        progress = QProgressDialog("Đang xử lý ảnh, vui lòng chờ...", None, 0, 0, self.view)
        progress.setWindowTitle("Chấm OMR")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Khởi chạy luồng ngầm
        self.active_worker = OMRSingleWorker(image_data, answers, template, logs_dir, f"{base_name}_")
        
        def on_finished(graded_image, result_text, score):
            progress.close()
            self._set_ui_locked(False)
            
            if self.active_worker and getattr(self.active_worker, '_is_cancelled', False):
                return
                
            if graded_image is not None:
                new_tab_id = f"graded_{base_name}"
                self.view.center_panel.add_or_switch_tab(new_tab_id, f"Kết quả: {base_name}")
                self.view.center_panel.display_cv_image(new_tab_id, graded_image)
                
                # Cập nhật thông tin lên Right Panel thay vì dùng Popup
                if hasattr(self.view, 'right_panel') and hasattr(self.view.right_panel, 'update_single_result'):
                    self.view.right_panel.update_single_result(score)
                
                if output_dir:
                    try:
                        # Xuất Excel
                        ReportExportService.export_single_result(score, base_name, output_dir)
                        # Lưu ảnh đã chấm
                        img_path = os.path.join(output_dir, f"graded_{base_name}.jpg")
                        cv2.imwrite(img_path, graded_image)
                    except Exception as e:
                        QMessageBox.warning(self.view, "Lỗi lưu file", f"Lỗi khi lưu kết quả: {e}")
            else:
                QMessageBox.critical(self.view, "Lỗi", "Không thể cập nhật ảnh sau khi chấm OMR.")
                
        def on_error(err_str):
            progress.close()
            self._set_ui_locked(False)
            QMessageBox.warning(self.view, "Lỗi khi chấm bài", err_str)
            
        self.active_worker.finished.connect(on_finished)
        self.active_worker.error.connect(on_error)
        self.active_worker.start()

    def handle_omr_batch(self):
        # Chọn thư mục đầu vào
        input_dir = QFileDialog.getExistingDirectory(self.view, "Chọn thư mục chứa ảnh bài thi", "")
        if not input_dir: return

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
        
        if not files:
            QMessageBox.information(self.view, "Thông báo", "Không tìm thấy ảnh nào trong thư mục.")
            return

        # Chọn mẫu
        template = self._select_template()
        if template is None: return

        # --- BƯỚC 1: TIỀN XỬ LÝ (PRE-SCAN) MỚI VỚI QTHREAD ---
        self._set_ui_locked(True)
        progress = QProgressDialog("Đang quét tiền xử lý (Pre-scan)...", "Hủy", 0, len(files), self.view)
        progress.setWindowTitle("Tiền xử lý")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self.active_worker = OMRBatchWorker('prescan', input_dir, files, template)

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
            self._set_ui_locked(False)

        progress.canceled.connect(on_prescan_cancel)

        def on_prescan_finished(report, valid_files):
            progress.close()
            self._set_ui_locked(False)
            
            if self.active_worker and getattr(self.active_worker, '_is_cancelled', False):
                return
                
            from ui.dialogs.prescan_report_dialog import PreScanReportDialog
            dialog = PreScanReportDialog(self.view, report)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            if report['valid'] == 0:
                return
                
            self._start_batch_grading(input_dir, valid_files)

        self.active_worker.progress.connect(on_prescan_progress)
        self.active_worker.error.connect(on_prescan_error)
        self.active_worker.prescan_finished.connect(on_prescan_finished)
        self.active_worker.start()

    def _start_batch_grading(self, input_dir, valid_files):
        # --- BƯỚC 2: CHUẨN BỊ XUẤT ---
        output_dir = QFileDialog.getExistingDirectory(self.view, "Chọn thư mục lưu ảnh kết quả và Excel", input_dir)
        if not output_dir: return
            
        logs_dir = os.path.join(output_dir, "logs")
        if not os.path.exists(logs_dir): os.makedirs(logs_dir)
            
        excel_path = os.path.join(output_dir, "tong_hop_diem.xlsx")

        answer_key_path = self._select_answer_key_file()
        if not answer_key_path: return

        answers = AnswerKeyService.load_answers(answer_key_path)
        if not answers:
            QMessageBox.warning(self.view, "Cảnh báo", "Không tìm thấy dữ liệu đáp án hoặc file rỗng.")
            return

        # --- BƯỚC 3: CHẤM THI CHÍNH THỨC VỚI QTHREAD ---
        self._set_ui_locked(True)
        progress = QProgressDialog("Đang chấm thi...", "Dừng", 0, len(valid_files), self.view)
        progress.setWindowTitle("Tiến trình chấm thi")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Dashboard init
        if hasattr(self.view, 'right_panel') and hasattr(self.view.right_panel, 'update_dashboard'):
            self.view.right_panel.update_dashboard({'total': 0, 'avg': 0, 'gioi': 0, 'kha': 0, 'tb': 0, 'yeu': 0})

        self.active_worker = OMRBatchWorker('grade', input_dir, valid_files, None, answers, output_dir, logs_dir)

        def on_grade_progress(val, text):
            progress.setValue(val)
            progress.setLabelText(text)

        def on_grade_error(err_str):
            progress.close()
            self._set_ui_locked(False)
            QMessageBox.warning(self.view, "Lỗi", f"Lỗi khi chấm: {err_str}")

        def on_grade_cancel():
            if self.active_worker and self.active_worker.isRunning():
                self.active_worker.cancel()
            self._set_ui_locked(False)

        progress.canceled.connect(on_grade_cancel)

        def on_grade_finished(success_count, error_count, excel_data, errors_log):
            progress.close()
            self._set_ui_locked(False)
            
            if self.active_worker and getattr(self.active_worker, '_is_cancelled', False):
                QMessageBox.information(self.view, "Đã hủy", "Tiến trình đã dừng giữa chừng. Một phần kết quả đã được lưu.")
                # We can still show results up to the cancel point

            # Update final dashboard
            if excel_data:
                ReportExportService.export_batch_results(excel_data, excel_path)
                
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
        self.active_worker.grading_finished.connect(on_grade_finished)
        self.active_worker.start()
