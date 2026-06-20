import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QLineEdit, QFileDialog, QMessageBox, QComboBox, QWidget, QFrame, QCheckBox)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from services.grading_service.template_service import TemplateService

class BatchSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thiết lập Chấm thi Hàng loạt")
        self.resize(800, 500)
        
        self.input_dir = ""
        self.output_dir = ""
        self.answer_key_path = ""
        self.selected_template = None
        
        self.templates = TemplateService.get_all_templates()
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Left column (Settings)
        left_layout = QVBoxLayout()
        
        # Input Dir
        left_layout.addWidget(QLabel("1. Thư mục chứa ảnh bài thi:"))
        row1 = QHBoxLayout()
        self.txt_input = QLineEdit()
        self.txt_input.setReadOnly(True)
        btn_input = QPushButton("Mở Folder")
        btn_input.clicked.connect(self.browse_input)
        row1.addWidget(self.txt_input)
        row1.addWidget(btn_input)
        left_layout.addLayout(row1)
        
        # Output Dir
        left_layout.addWidget(QLabel("2. Thư mục lưu kết quả (Excel & Ảnh):"))
        row2 = QHBoxLayout()
        self.txt_output = QLineEdit()
        self.txt_output.setReadOnly(True)
        btn_output = QPushButton("Mở Folder")
        btn_output.clicked.connect(self.browse_output)
        row2.addWidget(self.txt_output)
        row2.addWidget(btn_output)
        left_layout.addLayout(row2)
        
        # Answer Key
        left_layout.addWidget(QLabel("3. File Excel Đáp Án:"))
        row3 = QHBoxLayout()
        self.txt_answer = QLineEdit()
        self.txt_answer.setReadOnly(True)
        btn_answer = QPushButton("Chọn File")
        btn_answer.clicked.connect(self.browse_answer)
        row3.addWidget(self.txt_answer)
        row3.addWidget(btn_answer)
        left_layout.addLayout(row3)
        
        # Template
        left_layout.addWidget(QLabel("4. Chọn Mẫu Phiếu Thi:"))
        row4 = QHBoxLayout()
        self.combo_template = QComboBox()
        self.populate_templates()
        self.combo_template.currentIndexChanged.connect(self.on_template_changed)
        
        row4.addWidget(QLabel("Mẫu phiếu:"))
        row4.addWidget(self.combo_template)
        left_layout.addLayout(row4)
        
        # AI checkbox
        ai_layout = QHBoxLayout()
        self.chk_use_ai = QCheckBox("Hỗ trợ AI Khôi Phục (khi quét lỗi)")
        self.chk_use_ai.setChecked(False)
        self.chk_use_ai.setProperty("class", "checkbox-success")
        
        self.combo_device = QComboBox()
        self.combo_device.addItems(["Dùng CPU", "Dùng GPU (NVIDIA/DirectML)"])
        self.combo_device.setEnabled(False)
        
        self.chk_use_ai.stateChanged.connect(lambda state: self.combo_device.setEnabled(state == Qt.CheckState.Checked.value))
        self.combo_device.currentIndexChanged.connect(self.on_device_changed)
        
        ai_layout.addWidget(self.chk_use_ai)
        ai_layout.addWidget(QLabel("Thiết bị:"))
        ai_layout.addWidget(self.combo_device)
        ai_layout.addStretch()
        left_layout.addLayout(ai_layout)
        
        left_layout.addStretch()
        
        # Bottom Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Hủy")
        self.btn_start = QPushButton("BẮT ĐẦU CHẤM")
        self.btn_start.setProperty("variant", "primary")
        self.btn_start.setEnabled(False)
        
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_start.clicked.connect(self.accept)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_start)
        left_layout.addLayout(btn_layout)
        
        main_layout.addLayout(left_layout, 2)
        
        # Right column (Preview)
        right_layout = QVBoxLayout()
        self.lbl_preview = QLabel("Chưa chọn mẫu")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setProperty("class", "preview-box")
        self.lbl_preview.setMinimumSize(300, 400)
        right_layout.addWidget(self.lbl_preview)
        
        main_layout.addLayout(right_layout, 1)
        
        self.check_ready()
        self.on_template_changed()
        
    def populate_templates(self):
        self.combo_template.clear()
        
        for t in self.templates:
            if isinstance(t, dict):
                name = t.get("name", "Unknown")
            else:
                name = getattr(t, "name", "Unknown")
            self.combo_template.addItem(name, t)
            
        self.combo_template.addItem("Tự động nhận diện", "AUTO")
            
    def browse_input(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh")
        if dir_path:
            self.input_dir = dir_path
            self.txt_input.setText(dir_path)
            self.check_ready()
            
    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả")
        if dir_path:
            self.output_dir = dir_path
            self.txt_output.setText(dir_path)
            self.check_ready()
            
    def browse_answer(self):
        default_dir = os.path.join(os.getcwd(), "assets")
        if not os.path.exists(default_dir): default_dir = os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file Excel Đáp án", default_dir, "Excel Files (*.xlsx *.xls)")
        if path:
            self.answer_key_path = path
            self.txt_answer.setText(os.path.basename(path))
            self.check_ready()
            
    def on_device_changed(self, index):
        if index == 1:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' not in providers and 'TensorrtExecutionProvider' not in providers and 'DmlExecutionProvider' not in providers:
                    QMessageBox.warning(self, "Lỗi Phần Cứng", "Không tìm thấy Card đồ họa rời tương thích (NVIDIA/DirectML) hoặc bạn chưa cài đặt driver CUDA.\nHệ thống sẽ tự động chuyển về dùng CPU.")
                    self.combo_device.setCurrentIndex(0)
                else:
                    QMessageBox.information(self, "Thành công", f"Đã nhận diện thiết bị tăng tốc phần cứng!\nProviders: {', '.join([p for p in providers if p != 'CPUExecutionProvider'])}")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi kiểm tra GPU", f"Không thể kết nối với GPU: {str(e)}")
                self.combo_device.setCurrentIndex(0)

    def on_template_changed(self):
        t = self.combo_template.currentData()
        self.selected_template = t
        if t and 'preview_image_path' in t and os.path.exists(t['preview_image_path']):
            pixmap = QPixmap(t['preview_image_path'])
            scaled = pixmap.scaled(self.lbl_preview.width(), self.lbl_preview.height(), 
                                   Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_preview.setPixmap(scaled)
            self.lbl_preview.setProperty("class", "preview-box-active")
            self.lbl_preview.style().unpolish(self.lbl_preview)
            self.lbl_preview.style().polish(self.lbl_preview)
        else:
            self.lbl_preview.clear()
            self.lbl_preview.setText("Chưa có ảnh Preview")
            self.lbl_preview.setProperty("class", "preview-box")
            self.lbl_preview.style().unpolish(self.lbl_preview)
            self.lbl_preview.style().polish(self.lbl_preview)
        self.check_ready()
            
    def check_ready(self):
        is_ready = bool(self.input_dir and self.output_dir and self.answer_key_path and self.selected_template)
        self.btn_start.setEnabled(is_ready)
