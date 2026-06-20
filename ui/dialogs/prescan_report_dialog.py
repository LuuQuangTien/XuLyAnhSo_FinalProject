from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt

class PreScanReportDialog(QDialog):
    def __init__(self, parent=None, report_data=None):
        super().__init__(parent)
        self.setWindowTitle("Báo cáo Tiền xử lý (Pre-scan)")
        self.resize(550, 450)
        
        self.report_data = report_data or {}
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_title = QLabel("TỔNG QUAN XẤP BÀI THI")
        lbl_title.setProperty("class", "dialog-title")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)
        
        total = self.report_data.get('total', 0)
        valid = self.report_data.get('valid', 0)
        errors = self.report_data.get('errors', [])
        mades = self.report_data.get('mades', {})
        
        # --- Bảng Thống kê (3 cột - Card Design) ---
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)
        
        def create_stat_card(title, value, color_hex):
            frame = QFrame()
            frame.setProperty("class", "card")
            # Dynamic border-bottom
            frame.setStyleSheet(f"border-bottom: 4px solid {color_hex};")
            
            flayout = QVBoxLayout(frame)
            flayout.setContentsMargins(10, 15, 10, 15)
            
            lbl_val = QLabel(str(value))
            lbl_val.setProperty("class", "score-val")
            lbl_val.setStyleSheet(f"color: {color_hex};")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            lbl_txt = QLabel(title)
            lbl_txt.setProperty("class", "score-label")
            lbl_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            flayout.addWidget(lbl_val)
            flayout.addWidget(lbl_txt)
            return frame

        stats_layout.addWidget(create_stat_card("TỔNG SỐ ẢNH", total, "#3bc9db")) # Xanh dương nhạt
        stats_layout.addWidget(create_stat_card("HỢP LỆ", valid, "#40c057"))      # Xanh lá
        stats_layout.addWidget(create_stat_card("BỊ LỖI", len(errors), "#fa5252")) # Đỏ
        
        layout.addLayout(stats_layout)
        
        # --- Chi tiết Mã đề (Badge Design) ---
        if mades:
            made_frame = QFrame()
            made_frame.setProperty("class", "card")
            made_layout = QVBoxLayout(made_frame)
            made_layout.setContentsMargins(15, 15, 15, 15)
            
            lbl_mades_title = QLabel("📌 CÁC MÃ ĐỀ PHÁT HIỆN ĐƯỢC:")
            lbl_mades_title.setProperty("class", "subtitle-small")
            made_layout.addWidget(lbl_mades_title)
            
            badges_layout = QHBoxLayout()
            badges_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            badges_layout.setSpacing(10)
            
            for k, v in mades.items():
                lbl_badge = QLabel(f"Mã {k}: {v} bài")
                lbl_badge.setProperty("class", "badge")
                badges_layout.addWidget(lbl_badge)
            badges_layout.addStretch()
            made_layout.addLayout(badges_layout)
            
            layout.addWidget(made_frame)
            
        # --- Danh sách lỗi ---
        if errors:
            lbl_errors = QLabel("⚠️ DANH SÁCH ẢNH LỖI (SẼ BỊ BỎ QUA):")
            lbl_errors.setProperty("class", "error-title")
            layout.addWidget(lbl_errors)
            
            txt_errors = QTextEdit()
            txt_errors.setReadOnly(True)
            txt_errors.setProperty("class", "error-console")
            err_text = ""
            for err in errors:
                err_text += f"• {err['file']}: {err['reason']}\n"
            txt_errors.setText(err_text)
            layout.addWidget(txt_errors)
            
        layout.addStretch()
        
        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Hủy bỏ")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.setProperty("variant", "secondary")
        btn_cancel.clicked.connect(self.reject)
        
        btn_continue = QPushButton("Bỏ qua lỗi & Tiếp tục Chấm" if errors else "Tiếp tục Chấm")
        btn_continue.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_continue.setProperty("variant", "success")
        btn_continue.clicked.connect(self.accept)
        
        if valid == 0:
            btn_continue.setEnabled(False)
            btn_continue.setText("Không có ảnh hợp lệ")
            
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_continue)
        
        layout.addLayout(btn_layout)
