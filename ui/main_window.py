# Cửa sổ chính của ứng dụng. Chỉ chứa bố cục (layout) và khởi tạo các thành phần giao diện.
from PyQt6.QtWidgets import QMainWindow, QSplitter, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
import os
from ui.components.left_panel import LeftPanel
from ui.components.center_panel import CenterPanel
from ui.components.right_panel import RightPanel
from ui.components.toolbar import AppToolBar
from services.center_panel_service.image_state import ImageStateManager
from PyQt6.QtGui import QCloseEvent

class MainWindow(QMainWindow):
    """
    Pure UI class for the Main Window.
    Contains layouts and component instantiation only.
    Business logic and signal wiring are handled by MainController.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lily")

        icon_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'icons', 'AppIcon.ico')
        self.setWindowIcon(QIcon(icon_path))
        
        self.resize(1200, 800)
        
        # Reference to main_controller for closing events
        self.controller = None
        
        # Build UI
        self._setup_ui()
        self._create_tool_bar()

    def _setup_ui(self):
        """Initializes the layout and main components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.left_panel = LeftPanel()
        self.center_panel = CenterPanel()
        self.right_panel = RightPanel()
        
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.center_panel)
        self.splitter.addWidget(self.right_panel)
        
        # Set stretch factors (L: 1, C: 4, R: 1)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setStretchFactor(2, 1)
        
        layout.addWidget(self.splitter)

    def _create_tool_bar(self):
        """Initializes the toolbar component."""
        self.toolbar_component = AppToolBar()
        self.addToolBar(self.toolbar_component)

    def set_io_enabled(self, enabled: bool):
        """Enables or disables file I/O actions like Save and Save As."""
        pass

    def closeEvent(self, event: QCloseEvent):
        """Handles application close event."""
        if self.controller and hasattr(self.controller, 'handle_app_close'):
            if not self.controller.handle_app_close():
                event.ignore()
                return
        event.accept()
