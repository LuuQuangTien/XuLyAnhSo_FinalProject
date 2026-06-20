import os
from PyQt6.QtWidgets import QMessageBox
from services.notification_service import ask_reset_all
from ui import strings

class ImageProcessingController:
    def __init__(self, main_window, image_state, ui_state):
        self.view = main_window
        self.image_state = image_state
        self.ui_state = ui_state

    def _refresh_view_for_active_tab(self, state) -> None:
        tab_id = self.image_state.active_tab_id
        if not tab_id:
            return
        self.view.center_panel.display_cv_image(tab_id, state.current_processed_image)
        path = state.current_image_path or tab_id
        filename = os.path.basename(path) if path else ""
        self.view.center_panel.update_tab_title(tab_id, filename, state.is_modified)
        self.ui_state.refresh_history_buttons()


    def on_undo_request(self) -> None:
        state = self.image_state.get_active_state()
        if not state or not state.has_image() or not state.undo():
            return
        self._refresh_view_for_active_tab(state)

    def on_redo_request(self) -> None:
        state = self.image_state.get_active_state()
        if not state or not state.has_image() or not state.redo():
            return
        self._refresh_view_for_active_tab(state)

    def on_reset_request(self) -> None:
        state = self.image_state.get_active_state()
        if not state or not state.has_image():
            return
        if not ask_reset_all(self.view):
            return
        if not state.reset():
            return
        self._refresh_view_for_active_tab(state)
