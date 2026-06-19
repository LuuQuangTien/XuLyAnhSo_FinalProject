# Quản lý các sự kiện tương tác trên cây thư mục (Import folder, Xóa folder, Chọn ảnh).
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from ui import strings
from services.folder_tree_service import build_folder_tree
from services.toolbar_buttons_service.utils import load_image

class ExplorerController:
    def __init__(self, main_window, ui_state_controller, image_state):
        self.view = main_window
        self.image_state = image_state
        self.ui_state = ui_state_controller
        from services.folder_tree_service import ImportedFoldersTracker
        self.imported_folders = ImportedFoldersTracker()

    def handle_import_folder(self):
        folder = QFileDialog.getExistingDirectory(self.view, strings.MENU_IMPORT_FOLDER)
        if folder:
            self.import_folder_by_path(folder)

    def import_folder_by_path(self, folder_path: str):
        """Nhập một thư mục vào cây thư mục theo đường dẫn cụ thể."""
        tracker = self.imported_folders
        if not tracker.can_import(folder_path):
            return
            
        tracker.add(folder_path)
        tree_data = build_folder_tree(folder_path)
        self.view.left_panel.add_folder_by_data(tree_data)

    def handle_remove_folder(self, folder_path: str):
        from services.folder_tree_service import remove_folder
        remove_folder.execute(self.imported_folders, folder_path)
        # Note: UI removal is already handled by LeftPanel after emitting the signal,
        # but ideally the controller should tell the view to remove it.
        # Since LeftPanel.remove_root_item removes the row before emitting, we just update state here.

    def on_image_selected(self, file_path):
        """Handles image selection from the file explorer."""
        # Check if already open
        state = self.image_state.create_or_get_state(file_path)
        
        # Add or switch tab in UI
        import os
        filename = os.path.basename(file_path)
        self.view.center_panel.add_or_switch_tab(file_path, filename)
        self.image_state.set_active_tab(file_path)
        
        # Load image only if it hasn't been loaded
        if not state.has_image():
            loaded = load_image.execute(file_path)
            if loaded is not None:
                if not state.set_loaded_image(file_path, loaded):
                    QMessageBox.warning(
                        self.view,
                        strings.APP_NAME,
                        strings.MSG_CACHE_WRITE_FAILED,
                    )
        
        if state.has_image():
            self.view.center_panel.display_cv_image(file_path, state.current_processed_image)
            self.ui_state.on_image_loaded()
