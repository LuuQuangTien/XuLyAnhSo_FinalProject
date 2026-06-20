# Giao diện vùng hiển thị ảnh ở giữa màn hình (Đa Tab).
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QWidget, QSizePolicy, QSpinBox, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
import os
from ui import strings
from services.center_panel_service.image_adapter import cv_image_to_qpixmap

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QWheelEvent

class ZoomControlWidget(QWidget):
    zoom_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        self.btn_minus = QPushButton("-")
        self.btn_minus.setFixedSize(30, 30)
        self.btn_minus.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_minus.clicked.connect(self._on_minus)
        
        self.spinbox = QSpinBox()
        self.spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinbox.setRange(10, 800)
        self.spinbox.setSuffix(" %")
        self.spinbox.setFixedSize(65, 30)
        self.spinbox.setKeyboardTracking(False)
        self.spinbox.valueChanged.connect(lambda v: self.zoom_changed.emit(float(v)))
        
        self.btn_plus = QPushButton("+")
        self.btn_plus.setFixedSize(30, 30)
        self.btn_plus.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_plus.clicked.connect(self._on_plus)
        
        layout.addWidget(self.btn_minus)
        layout.addWidget(self.spinbox)
        layout.addWidget(self.btn_plus)
        
        self.setObjectName("ZoomControlWidget")

    def setValue(self, val):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(int(val))
        self.spinbox.blockSignals(False)
        
    def _on_minus(self):
        val = self.spinbox.value()
        self.spinbox.setValue(max(10, val - 10))
        
    def _on_plus(self):
        val = self.spinbox.value()
        self.spinbox.setValue(min(800, val + 10))

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        self.zoom_factor = 1.15
        self._pixmap = None
        self._current_zoom = 100.0
        self.min_zoom = 10.0
        self.max_zoom = 800.0
        
        # Lớp phủ Zoom Control
        self.zoom_control = ZoomControlWidget(self)
        self.zoom_control.zoom_changed.connect(self._update_zoom)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Đặt vị trí widget ở góc trên bên phải
        margin = 15
        zc_width = 135
        zc_height = 32
        self.zoom_control.setGeometry(self.width() - zc_width - margin, margin, zc_width, zc_height)

    def _update_zoom(self, new_zoom):
        new_zoom = max(self.min_zoom, min(new_zoom, self.max_zoom))
        if abs(self._current_zoom - new_zoom) < 0.01: return
        
        factor = new_zoom / self._current_zoom
        self._current_zoom = new_zoom
        
        self.zoom_control.setValue(self._current_zoom)
        self.scale(factor, factor)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        
        # Reset scale và tính toán zoom sau khi Fit In View
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        
        current_scale = self.transform().m11()
        self._current_zoom = current_scale * 100.0
        
        self.zoom_control.setValue(self._current_zoom)

    def wheelEvent(self, event: QWheelEvent):
        if not self._pixmap: return
        
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier or True:
            if event.angleDelta().y() > 0:
                self._update_zoom(self._current_zoom * self.zoom_factor)
            else:
                self._update_zoom(self._current_zoom / self.zoom_factor)



class CenterPanel(QFrame):

    tab_closed = pyqtSignal(str) # emits tab_id (file_path)
    tab_switched = pyqtSignal(str) # emits tab_id (file_path)

    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)
        
        self.tab_map = {} # tab_id -> { "widget": QWidget, "label": QLabel }
        self.placeholder = QLabel(strings.LBL_NO_IMAGE)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setProperty("class", "placeholder-text")
        layout.addWidget(self.placeholder)
        
        self.update_placeholder_visibility()

    def update_placeholder_visibility(self):
        if self.tabs.count() == 0:
            self.tabs.hide()
            self.placeholder.show()
        else:
            self.placeholder.hide()
            self.tabs.show()

    def _on_tab_close_requested(self, index):
        widget = self.tabs.widget(index)
        tab_id = widget.property("tab_id")
        self.tab_closed.emit(tab_id)

    def _on_tab_changed(self, index):
        if index >= 0:
            widget = self.tabs.widget(index)
            tab_id = widget.property("tab_id")
            self.tab_switched.emit(tab_id)
        else:
            self.tab_switched.emit("")

    def add_or_switch_tab(self, tab_id: str, title: str):
        if tab_id in self.tab_map:
            # Switch to existing
            widget = self.tab_map[tab_id]["widget"]
            self.tabs.setCurrentWidget(widget)
            return

        # Create new tab
        container = QWidget()
        container.setProperty("tab_id", tab_id)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        viewer = ImageViewer()
        layout.addWidget(viewer)
        
        self.tab_map[tab_id] = {
            "widget": container,
            "viewer": viewer
        }
        
        self.tabs.addTab(container, title)
        self.tabs.setCurrentWidget(container)
        self.update_placeholder_visibility()

    def remove_tab(self, tab_id: str):
        if tab_id in self.tab_map:
            widget = self.tab_map[tab_id]["widget"]
            index = self.tabs.indexOf(widget)
            self.tabs.removeTab(index)
            del self.tab_map[tab_id]
            self.update_placeholder_visibility()



    def update_tab_title(self, tab_id: str, title: str, is_modified: bool = False):
        if tab_id in self.tab_map:
            widget = self.tab_map[tab_id]["widget"]
            index = self.tabs.indexOf(widget)
            display_title = f"{title} *" if is_modified else title
            self.tabs.setTabText(index, display_title)

    def display_cv_image(self, tab_id: str, cv_img):
        """Hiển thị ảnh từ mảng NumPy (OpenCV) trên tab chỉ định."""
        if cv_img is None or tab_id not in self.tab_map:
            return

        pixmap = cv_image_to_qpixmap(cv_img)
        if pixmap is None:
            return

        viewer = self.tab_map[tab_id]["viewer"]
        viewer.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Optional: auto fitInView if user resizes window.
        # Currently leaving as is so user's zoom level is preserved during minor resizes.
