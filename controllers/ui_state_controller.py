# Quản lý trạng thái giao diện (Bật/tắt các nút bấm tùy thuộc vào việc có ảnh hay chưa).
from ui import strings

class UIStateController:
    """
    Manages the enabled/disabled state of UI components based on the application state.
    """

    def __init__(self, main_window, image_state):
        self.view = main_window
        self.image_state = image_state

        # Disable processing and I/O buttons by default on startup
        self._update_availability(False)

    def on_image_loaded(self):
        """Called when any image is successfully loaded and displayed."""
        state = self.image_state.get_active_state()
        if state and state.has_image():
            self.view.set_io_enabled(True)
            self.view.right_panel.set_processing_enabled(True)
        else:
            self.view.set_io_enabled(False)
            self.view.right_panel.set_processing_enabled(False)

    def on_image_cleared(self):
        """Called if the image is removed from the application."""
        self._update_availability(False)

    def _update_availability(self, has_image: bool):
        """Toggles the availability of actions requiring an image."""
        self.view.right_panel.set_processing_enabled(has_image)
        self.view.set_io_enabled(has_image)

