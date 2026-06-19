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
        self.is_auto_mode = True
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Mode Selection ---
        mode_layout = QHBoxLayout()
        self.btn_auto = QRadioButton("Tự động nhận diện (Khuyến nghị)")
        self.btn_manual = QRadioButton("Chọn thủ công")
        
        self.btn_auto.setChecked(True)
        
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.btn_auto)
        self.mode_group.addButton(self.btn_manual)
        
        mode_layout.addWidget(self.btn_auto)
        mode_layout.addWidget(self.btn_manual)
        layout.addLayout(mode_layout)
        
        self.btn_auto.toggled.connect(self.on_mode_changed)
        
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
        self.on_mode_changed()

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
            if hasattr(template, 'preview_image_path') and os.path.exists(template.preview_image_path):
                pixmap = QPixmap(template.preview_image_path)
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
            
            # Text
            name = getattr(template, 'name', 'Unknown Template')
            desc = getattr(template, 'description', '')
            
            display_text = f"{name}{conf_text}\n{desc}"
            list_item.setText(display_text)
            
            # Lưu template object vào Data
            list_item.setData(Qt.ItemDataRole.UserRole, template)
            self.list_widget.addItem(list_item)

    def on_mode_changed(self):
        self.is_auto_mode = self.btn_auto.isChecked()
        self.list_widget.setEnabled(not self.is_auto_mode)

    def force_manual_mode(self, templates_with_conf):
        """Hàm này được gọi khi Auto bị fail, bắt buộc chuyển sang Manual và hiện điểm conf"""
        self.btn_manual.setChecked(True)
        self.populate_list(templates_with_conf)
        self.list_widget.setCurrentRow(0)

    def on_item_double_clicked(self, item):
        template = item.data(Qt.ItemDataRole.UserRole)
        if not template or not hasattr(template, 'preview_image_path') or not os.path.exists(template.preview_image_path):
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Preview: {getattr(template, 'name', '')}")
        dialog.resize(800, 800)
        layout = QVBoxLayout(dialog)
        
        pixmap = QPixmap(template.preview_image_path)
        if pixmap.width() > 800 or pixmap.height() > 800:
            pixmap = pixmap.scaled(800, 800, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        dialog.exec()

    def on_ok_clicked(self):
        if not self.is_auto_mode:
            current_item = self.list_widget.currentItem()
            if current_item:
                self.selected_template = current_item.data(Qt.ItemDataRole.UserRole)
            else:
                self.selected_template = None
        else:
            self.selected_template = "AUTO"
            
        self.accept()
