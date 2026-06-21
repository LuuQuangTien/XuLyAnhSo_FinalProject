import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, 
                             QListWidget, QListWidgetItem, QPushButton, QLabel, 
                             QWidget, QButtonGroup)
from PyQt6.QtGui import QPixmap, QIcon, QPainter
from PyQt6.QtCore import Qt, QSize

class TemplateSelectionDialog(QDialog):
    def __init__(self, templates, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chọn Mẫu Chấm Thi OMR")
        self.resize(500, 600)
        self.templates = templates
        self.selected_template = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Mode Selection ---
        # Removed Auto/Manual mode selection buttons
        
        # --- List Widget ---
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(100, 100))
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.list_widget)
        
        # --- Bottom Buttons ---
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("Tiếp tục")
        self.btn_cancel = QPushButton("Hủy")
        
        self.btn_ok.clicked.connect(self.on_ok_clicked)
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)
        
        self.populate_list(self.templates)

    def populate_list(self, templates_with_conf=None):
        """
        templates_with_conf là mảng tuple: (template_obj, confidence_score) hoặc chỉ là danh sách template.
        """
        self.list_widget.clear()
        
        for item in templates_with_conf:
            if isinstance(item, tuple):
                template, conf = item
                conf_text = f" - Độ tự tin: {int(conf * 100)}%"
            else:
                template = item
                conf_text = ""
                
            list_item = QListWidgetItem()
            
            # Load icon
            preview_path = None
            if isinstance(template, dict):
                preview_path = template.get('preview_image_path')
            elif hasattr(template, 'preview_image_path'):
                preview_path = template.preview_image_path
                
            if preview_path and os.path.exists(preview_path):
                pixmap = QPixmap(preview_path)
                scaled_pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                
                final_pixmap = QPixmap(100, 100)
                final_pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(final_pixmap)
                x = (100 - scaled_pixmap.width()) // 2
                y = (100 - scaled_pixmap.height()) // 2
                painter.drawPixmap(x, y, scaled_pixmap)
                painter.end()
                icon = QIcon(final_pixmap)
            else:
                icon = QIcon()
                
            list_item.setIcon(icon)
            
            # Cập nhật để hỗ trợ đọc từ Dictionary JSON
            if isinstance(template, dict):
                name = template.get('name', 'Unknown JSON Template')
                desc = template.get('description', '')
            else:
                # Giữ nguyên logic cũ phòng trường hợp còn object rác
                name = getattr(template, 'name', 'Unknown Template')
                desc = getattr(template, 'description', '')
            
            display_text = f"{name}{conf_text}\n{desc}"
            list_item.setText(display_text)
            
            # Lưu template object vào Data
            list_item.setData(Qt.ItemDataRole.UserRole, template)
            self.list_widget.addItem(list_item)

    def on_item_double_clicked(self, item):
        template = item.data(Qt.ItemDataRole.UserRole)
        if not template:
            return
            
        preview_path = None
        title = "Preview"
        if isinstance(template, dict):
            preview_path = template.get('preview_image_path')
            title = f"Preview: {template.get('name', '')}"
        elif hasattr(template, 'preview_image_path'):
            preview_path = template.preview_image_path
            title = f"Preview: {getattr(template, 'name', '')}"
            
        if not preview_path or not os.path.exists(preview_path):
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(800, 800)
        layout = QVBoxLayout(dialog)
        
        pixmap = QPixmap(preview_path)
        if pixmap.width() > 800 or pixmap.height() > 800:
            pixmap = pixmap.scaled(800, 800, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        dialog.exec()

    def on_ok_clicked(self):
        current_item = self.list_widget.currentItem()
        if current_item:
            self.selected_template = current_item.data(Qt.ItemDataRole.UserRole)
        else:
            self.selected_template = None
            
        self.accept()
