# Quản lý các luồng công việc liên quan đến file (Mở file, Lưu ảnh, Lưu ảnh mới).
import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from ui import strings
from services.center_panel_service import validate_image
from services.notification_service import info_save_success, warn_save_error
from services.toolbar_buttons_service.utils import load_image, save_image
from services.folder_tree_service import process_imported_files

class FileIOController:
    def __init__(self, main_window, image_state):
        self.view = main_window
        self.image_state = image_state

    def _refresh_history_buttons(self) -> None:
        ctrl = getattr(self.view, "controller", None)
        if ctrl and hasattr(ctrl, "ui_state"):
            ctrl.ui_state.refresh_history_buttons()

    def handle_import_file(self):
        files, _ = QFileDialog.getOpenFileNames(
            self.view,
            strings.MENU_IMPORT_FILE,
            "",
            "Image Files (*.jpg *.jpeg *.png *.webp *.bmp *.gif);;All Files (*)",
        )
        if files:
            processed_data = process_imported_files.execute(files)
            if processed_data:
                self.view.left_panel.add_files_to_tree(processed_data)

    def handle_save(self):
        state = self.image_state.get_active_state()
        if not state or not state.has_image():
            return

        output_path = state.current_output_path
        if not output_path:
            self.handle_save_as()
            return

        reply = QMessageBox.question(
            self.view,
            strings.MSG_SAVE_CONFIRM_TITLE,
            strings.MSG_SAVE_CONFIRM_TEXT,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Ok:
            success = save_image.execute(state.current_processed_image, output_path)
            if success:
                state.set_output_path(output_path)
                info_save_success(self.view)
                
                # Cập nhật tên tab (xóa dấu *)
                filename = os.path.basename(output_path)
                self.view.center_panel.update_tab_title(self.image_state.active_tab_id, filename, False)
                self._refresh_history_buttons()
            else:
                warn_save_error(self.view, "Failed to encode or write image data.")

    def handle_save_as(self):
        state = self.image_state.get_active_state()
        if not state or not state.has_image():
            return

        base_path = state.current_output_path or state.current_image_path
        default_name = os.path.basename(base_path) if base_path else ""
        
        save_path, _ = QFileDialog.getSaveFileName(
            self.view,
            strings.MSG_SAVE_AS_TITLE,
            default_name,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WEBP (*.webp);;BMP (*.bmp);;All Files (*)",
        )
        
        if not save_path:
            return

        success = save_image.execute(state.current_processed_image, save_path)
        if success:
            state.set_output_path(save_path)
            info_save_success(self.view, save_path)
            
            # Cập nhật lại tab với đường dẫn mới (giả sử tab_id vẫn không đổi hoặc tạo mới, 
            # để đơn giản ta chỉ đổi tên hiển thị và xóa dấu *)
            filename = os.path.basename(save_path)
            self.view.center_panel.update_tab_title(self.image_state.active_tab_id, filename, False)
            self._refresh_history_buttons()
            # Cần chú ý: nếu tab_id (file path) thay đổi thì phải update key trong map, 
            # nhưng tạm thời tab_id cứ giữ là original path lúc mở file.
        else:
            warn_save_error(self.view, "Failed to encode or write image data.")

    def handle_save_all(self):
        unsaved_states = self.image_state.get_all_unsaved_states()
        if not unsaved_states:
            return
            
        success_count = 0
        for tab_id, state in unsaved_states.items():
            if state.current_output_path:
                success = save_image.execute(state.current_processed_image, state.current_output_path)
                if success:
                    state.set_output_path(state.current_output_path)
                    filename = os.path.basename(state.current_output_path)
                    self.view.center_panel.update_tab_title(tab_id, filename, False)
                    success_count += 1
            else:
                # Nếu file chưa có tên (thường không xảy ra vì đã mở từ cây thư mục)
                pass
                
        if success_count > 0:
            QMessageBox.information(self.view, strings.APP_NAME, f"Successfully saved {success_count} file(s).")
            self._refresh_history_buttons()
