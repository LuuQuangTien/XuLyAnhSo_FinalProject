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
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4dabf7;")
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
            # Dark mode friendly card style with colored bottom border
            frame.setStyleSheet(f"""
                QFrame {{ 
                    background-color: #2c2f33; 
                    border-radius: 8px; 
                    border-bottom: 4px solid {color_hex}; 
                }}
            """)
            flayout = QVBoxLayout(frame)
            flayout.setContentsMargins(10, 15, 10, 15)
            
            lbl_val = QLabel(str(value))
            lbl_val.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color_hex};")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            lbl_txt = QLabel(title)
            lbl_txt.setStyleSheet("font-size: 13px; font-weight: bold; color: #ced4da;")
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
            made_frame.setStyleSheet("QFrame { background-color: #25262b; border-radius: 8px; }")
            made_layout = QVBoxLayout(made_frame)
            made_layout.setContentsMargins(15, 15, 15, 15)
            
            lbl_mades_title = QLabel("📌 CÁC MÃ ĐỀ PHÁT HIỆN ĐƯỢC:")
            lbl_mades_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #e9ecef;")
            made_layout.addWidget(lbl_mades_title)
            
            badges_layout = QHBoxLayout()
            badges_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            badges_layout.setSpacing(10)
            
            for k, v in mades.items():
                lbl_badge = QLabel(f"Mã {k}: {v} bài")
                lbl_badge.setStyleSheet("""
                    background-color: #4dabf7; 
                    color: white; 
                    padding: 5px 12px; 
                    border-radius: 12px; 
                    font-weight: bold;
                    font-size: 12px;
                """)
                badges_layout.addWidget(lbl_badge)
            badges_layout.addStretch()
            made_layout.addLayout(badges_layout)
            
            layout.addWidget(made_frame)
            
        # --- Danh sách lỗi ---
        if errors:
            lbl_errors = QLabel("⚠️ DANH SÁCH ẢNH LỖI (SẼ BỊ BỎ QUA):")
            lbl_errors.setStyleSheet("font-weight: bold; font-size: 13px; color: #fa5252; margin-top: 10px;")
            layout.addWidget(lbl_errors)
            
            txt_errors = QTextEdit()
            txt_errors.setReadOnly(True)
            txt_errors.setStyleSheet("""
                QTextEdit { 
                    color: #ffc9c9; 
                    background-color: #343a40; 
                    border: 1px solid #fa5252; 
                    border-radius: 6px; 
                    padding: 8px; 
                    font-family: Consolas, monospace;
                    font-size: 13px;
                }
            """)
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
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #495057; color: white; border-radius: 6px; padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #343a40; }
        """)
        btn_cancel.clicked.connect(self.reject)
        
        btn_continue = QPushButton("Bỏ qua lỗi & Tiếp tục Chấm" if errors else "Tiếp tục Chấm")
        btn_continue.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_continue.setStyleSheet("""
            QPushButton {
                background-color: #40c057; color: white; border-radius: 6px; padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2f9e44; }
            QPushButton:disabled { background-color: #2b8a3e; color: #adb5bd; }
        """)
        btn_continue.clicked.connect(self.accept)
        
        if valid == 0:
            btn_continue.setEnabled(False)
            btn_continue.setText("Không có ảnh hợp lệ")
            
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_continue)
        
        layout.addLayout(btn_layout)
