# Bộ điều phối trung tâm. Khởi tạo và kết nối các bộ điều phối con với giao diện.
from PyQt6.QtWidgets import QMessageBox
from controllers.file_io_controller import FileIOController
from controllers.explorer_controller import ExplorerController
from controllers.image_processing_controller import ImageProcessingController
from controllers.ui_state_controller import UIStateController
from controllers.theme_controller import ThemeController
from services.notification_service import ask_save_changes, ask_save_all_exit
from services.center_panel_service.cache_manager import cache_manager
from services.center_panel_service.image_state import ImageStateManager

class MainController:
    """
    The main entry point for all controllers.
    Initializes sub-controllers and binds UI signals to their respective handlers.
    """
    def __init__(self, main_window):
        self.view = main_window
        self.view.controller = self # For closeEvent interception
        cache_manager.init_cache()
        self.image_state = ImageStateManager()
        
        # Initialize Sub-Controllers
        self.ui_state = UIStateController(main_window, self.image_state)
        self.theme = ThemeController(main_window)
        self.file_io = FileIOController(main_window, self.image_state)
        self.explorer = ExplorerController(main_window, self.ui_state, self.image_state)
        self.image_processing = ImageProcessingController(
            main_window, self.image_state, self.ui_state
        )
        
        # OMR
        from controllers.omr_controller import OMRController
        self.omr_controller = OMRController(main_window, self.image_state)
        
        self._bind_signals()

    def _bind_signals(self):
        """Connects UI signals to controller methods."""
        # Toolbar signals (File I/O and Explorer)
        toolbar = self.view.toolbar_component
        toolbar.action_import_file.triggered.connect(self.file_io.handle_import_file)
        toolbar.action_save.triggered.connect(self.file_io.handle_save)
        toolbar.action_save_as.triggered.connect(self.file_io.handle_save_as)
        toolbar.action_save_all.triggered.connect(self.file_io.handle_save_all)
        toolbar.action_import_folder.triggered.connect(self.explorer.handle_import_folder)
        toolbar.action_exit.triggered.connect(self.view.close)
        
        # OMR
        toolbar.action_omr.triggered.connect(self.omr_controller.handle_omr_processing)
        toolbar.action_omr_batch.triggered.connect(self.omr_controller.handle_omr_batch)
        toolbar.action_edit_answers.triggered.connect(self.omr_controller.open_answer_key_dialog)

        # Toolbar signals (Theme)
        toolbar.action_theme_light.triggered.connect(lambda: self.theme.handle_theme_change("light"))
        toolbar.action_theme_dark.triggered.connect(lambda: self.theme.handle_theme_change("dark"))
        toolbar.action_theme_warm.triggered.connect(lambda: self.theme.handle_theme_change("warm"))
        toolbar.action_theme_cold.triggered.connect(lambda: self.theme.handle_theme_change("cold"))

        # Left Panel signals (Explorer)
        self.view.left_panel.image_selected.connect(self.explorer.on_image_selected)
        self.view.left_panel.folder_removed_requested.connect(self.explorer.handle_remove_folder)
        
        # Center Panel signals (Tabs)
        self.view.center_panel.tab_closed.connect(self.handle_tab_close)
        self.view.center_panel.tab_switched.connect(self.handle_tab_switch)

    def handle_tab_close(self, tab_id: str):
        state = self.image_state.states.get(tab_id)
        if state and state.is_modified:
            import os
            filename = os.path.basename(tab_id)
            reply = ask_save_changes(self.view, filename)
            
            if reply == QMessageBox.StandardButton.Yes:
                # Cần switch sang tab đó trước khi save để logic lấy active state hoạt động đúng
                self.view.center_panel.add_or_switch_tab(tab_id, filename)
                self.image_state.set_active_tab(tab_id)
                self.file_io.handle_save()
                # Sau khi save thành công thì xóa tab
                if not state.is_modified: 
                    self._remove_tab_and_state(tab_id)
            elif reply == QMessageBox.StandardButton.No:
                self._remove_tab_and_state(tab_id)
            # Cancel: do nothing
        else:
            self._remove_tab_and_state(tab_id)

    def _remove_tab_and_state(self, tab_id: str):
        self.view.center_panel.remove_tab(tab_id)
        self.image_state.remove_state(tab_id)
        self.ui_state.on_image_loaded() # Re-evaluate UI state

    def handle_tab_switch(self, tab_id: str):
        self.image_state.set_active_tab(tab_id)
        # Khi chuyển tab, cần gọi on_image_loaded để UIStateController đồng bộ lại slider
        self.ui_state.on_image_loaded()

    def handle_app_close(self) -> bool:
        """Called by MainWindow.closeEvent. Returns True if safe to close."""
        unsaved_count = self.image_state.get_unsaved_count()
        if unsaved_count == 0:
            cache_manager.clear_all()
            return True

        reply = ask_save_all_exit(self.view, unsaved_count)

        if reply == QMessageBox.StandardButton.SaveAll:
            self.file_io.handle_save_all()
            if self.image_state.get_unsaved_count() == 0:
                cache_manager.clear_all()
                return True
            return False
        if reply == QMessageBox.StandardButton.Discard:
            cache_manager.clear_all()
            return True
        return False
