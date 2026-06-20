import os
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame
from PyQt6.QtCore import Qt

class BatchCompleteDialog(QDialog):
    def __init__(self, parent=None, summary_data=None):
        super().__init__(parent)
        self.setWindowTitle("Báo cáo Kết quả Chấm thi")
        self.resize(600, 500)
        
        self.summary = summary_data or {}
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        lbl_title = QLabel("🎉 HOÀN TẤT CHẤM THI")
        lbl_title.setProperty("class", "dialog-title-success")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)
        
        # Summary Cards
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)
        
        success = self.summary.get('success', 0)
        error = self.summary.get('error', 0)
        total = success + error
        
        stats_layout.addWidget(self._create_card("Tổng cộng", str(total), "#555", "#fff"))
        stats_layout.addWidget(self._create_card("Thành công", str(success), "#2e7d32", "#fff"))
        stats_layout.addWidget(self._create_card("Lỗi", str(error), "#c62828", "#fff"))
        layout.addLayout(stats_layout)
        
        # Score Distribution
        excel_data = self.summary.get('excel_data', [])
        if excel_data:
            dist_layout = QVBoxLayout()
            lbl_dist = QLabel("📊 Phổ điểm (Thống kê nhanh):")
            lbl_dist.setProperty("class", "subtitle")
            dist_layout.addWidget(lbl_dist)
            
            gioi, kha, tb, yeu = 0, 0, 0, 0
            for row in excel_data:
                score = row.get("Điểm", 0)
                if score >= 8: gioi += 1
                elif score >= 6.5: kha += 1
                elif score >= 5: tb += 1
                else: yeu += 1
                
            dist_text = (
                f"Giỏi (>=8): {gioi} bài\n"
                f"Khá (6.5 - 8): {kha} bài\n"
                f"Trung bình (5 - 6.5): {tb} bài\n"
                f"Yếu (<5): {yeu} bài"
            )
            lbl_dist_data = QLabel(dist_text)
            lbl_dist_data.setProperty("class", "data-box")
            dist_layout.addWidget(lbl_dist_data)
            layout.addLayout(dist_layout)
        
        # Errors Log
        errors_log = self.summary.get('errors_log', [])
        if errors_log:
            lbl_err = QLabel("⚠️ Danh sách lỗi:")
            lbl_err.setProperty("class", "error-title")
            layout.addWidget(lbl_err)
            
            txt_errors = QTextEdit()
            txt_errors.setReadOnly(True)
            txt_errors.setProperty("class", "error-console")
            txt_errors.setText("\n".join(errors_log))
            layout.addWidget(txt_errors)
        else:
            layout.addStretch()
            
        # Action Buttons
        btn_layout = QHBoxLayout()
        
        btn_open_folder = QPushButton("📂 Mở thư mục kết quả")
        btn_open_folder.setProperty("variant", "secondary")
        btn_open_folder.clicked.connect(self._open_folder)
        
        btn_open_excel = QPushButton("📊 Mở File Excel Điểm")
        btn_open_excel.setProperty("variant", "primary")
        btn_open_excel.clicked.connect(self._open_excel)
        
        btn_close = QPushButton("Đóng")
        btn_close.setProperty("variant", "secondary")
        btn_close.clicked.connect(self.accept)
        
        btn_layout.addWidget(btn_open_folder)
        btn_layout.addWidget(btn_open_excel)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)

    def _create_card(self, title, value, bg_color, text_color):
        frame = QFrame()
        frame.setProperty("class", "card")
        frame.setStyleSheet(f"background-color: {bg_color};")
        lyt = QVBoxLayout(frame)
        lyt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_t = QLabel(title)
        lbl_t.setProperty("class", "score-label")
        lbl_t.setStyleSheet(f"color: {text_color};")
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_v = QLabel(value)
        lbl_v.setProperty("class", "score-val")
        lbl_v.setStyleSheet(f"color: {text_color};")
        lbl_v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lyt.addWidget(lbl_t)
        lyt.addWidget(lbl_v)
        return frame

    def _open_folder(self):
        folder = self.summary.get('output_dir')
        if folder and os.path.exists(folder):
            os.startfile(folder)
            
    def _open_excel(self):
        excel_path = self.summary.get('excel_path')
        if excel_path and os.path.exists(excel_path):
            os.startfile(excel_path)
