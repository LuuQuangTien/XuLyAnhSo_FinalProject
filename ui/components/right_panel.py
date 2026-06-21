from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QScrollArea, QProgressBar, QLineEdit, QFileDialog, QMessageBox
import pandas as pd
from PyQt6.QtCore import Qt
from ui import strings

class RightPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        self.setMinimumWidth(280)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)



        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        # --- SINGLE RESULT UI ---
        self.single_result_container = QWidget()
        self.single_result_container.hide() # Ẩn mặc định, chỉ hiện khi chấm ảnh đơn
        single_layout = QVBoxLayout(self.single_result_container)
        single_layout.setContentsMargins(10, 15, 10, 15)
        single_layout.setSpacing(8)
        self.single_result_container.setProperty("class", "result-container")

        lbl_single_title = QLabel("📄 KẾT QUẢ BÀI THI")
        lbl_single_title.setProperty("class", "subtitle")
        lbl_single_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        single_layout.addWidget(lbl_single_title)

        self.lbl_score_huge = QLabel("0.0")
        self.lbl_score_huge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_score_huge.setProperty("class", "score-huge")
        single_layout.addWidget(self.lbl_score_huge)

        self.lbl_correct_count = QLabel("Số câu đúng: 0/0")
        self.lbl_correct_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_correct_count.setProperty("class", "score-count")
        single_layout.addWidget(self.lbl_correct_count)
        
        # Đường kẻ ngang
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setProperty("class", "separator-line")
        single_layout.addWidget(line)

        self.lbl_sbd = QLabel("Số báo danh: N/A")
        self.lbl_sbd.setProperty("class", "info-label")
        self.lbl_sbd.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        single_layout.addWidget(self.lbl_sbd)

        self.lbl_made = QLabel("Mã đề thi: N/A")
        self.lbl_made.setProperty("class", "info-label")
        self.lbl_made.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        single_layout.addWidget(self.lbl_made)

        self.lbl_notes = QLabel("")
        self.lbl_notes.setProperty("class", "error-label")
        self.lbl_notes.setWordWrap(True)
        single_layout.addWidget(self.lbl_notes)

        layout.addWidget(self.single_result_container)

        # --- DASHBOARD UI (BATCH) ---
        lbl_dash = QLabel("📊 THỐNG KÊ")
        lbl_dash.setProperty("class", "subtitle")
        lbl_dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_dash)
        
        # Chọn file Excel
        row_excel = QHBoxLayout()
        self.txt_excel_path = QLineEdit()
        self.txt_excel_path.setReadOnly(True)
        self.txt_excel_path.setPlaceholderText("Đường dẫn file Excel...")
        btn_browse_excel = QPushButton("Mở")
        btn_browse_excel.clicked.connect(self._browse_excel)
        row_excel.addWidget(self.txt_excel_path)
        row_excel.addWidget(btn_browse_excel)
        layout.addLayout(row_excel)
        
        self.lbl_total = QLabel("Số bài đã chấm: 0")
        self.lbl_total.setProperty("class", "info-label")
        layout.addWidget(self.lbl_total)
        
        self.lbl_avg = QLabel("Điểm trung bình: 0.0")
        self.lbl_avg.setProperty("class", "info-label")
        layout.addWidget(self.lbl_avg)
        
        # Phổ điểm
        lbl_pho = QLabel("Phân bố phổ điểm:")
        lbl_pho.setProperty("class", "chart-title")
        layout.addWidget(lbl_pho)
        
        self.bars = {}
        categories = [
            ("Giỏi (>=8)", "gioi"),
            ("Khá (6.5-8)", "kha"),
            ("TB (5-6.5)", "tb"),
            ("Yếu (<5)", "yeu")
        ]
        
        for key, level in categories:
            row = QVBoxLayout()
            row.setSpacing(2)
            lbl = QLabel(f"{key}: 0")
            lbl.setProperty("class", "chart-label")
            bar = QProgressBar()
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setProperty("class", "chart-bar")
            bar.setProperty("scoreLevel", level)
            bar.setMaximum(100)
            bar.setValue(0)
            
            row.addWidget(lbl)
            row.addWidget(bar)
            layout.addLayout(row)
            
            self.bars[key] = (lbl, bar)
            
        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _browse_excel(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file Excel kết quả", "", "Excel Files (*.xlsx *.xls)")
        if path:
            self.load_excel_stats(path)
            
    def load_excel_stats(self, excel_path: str):
        self.txt_excel_path.setText(excel_path)
        try:
            df = pd.read_excel(excel_path)
            
            # Kiểm tra xem có cột "Điểm" hay không
            if "Điểm" not in df.columns:
                QMessageBox.warning(self, "Lỗi định dạng", "File Excel không có cột 'Điểm'. Không thể thống kê.")
                return
                
            scores = pd.to_numeric(df["Điểm"], errors='coerce').dropna()
            total = len(df)
            avg = scores.mean() if total > 0 else 0
            
            gioi = len(scores[scores >= 8])
            kha = len(scores[(scores >= 6.5) & (scores < 8)])
            tb = len(scores[(scores >= 5) & (scores < 6.5)])
            yeu = len(scores[scores < 5])
            
            stats = {
                'total': total,
                'avg': avg,
                'gioi': gioi,
                'kha': kha,
                'tb': tb,
                'yeu': yeu
            }
            self.update_dashboard(stats)
            
        except Exception as e:
            QMessageBox.warning(self, "Lỗi đọc file", f"Không thể đọc file Excel: {e}")

    def update_dashboard(self, stats):
        total = stats.get('total', 0)
        self.lbl_total.setText(f"Số bài đã chấm: {total}")
        
        avg = stats.get('avg', 0.0)
        self.lbl_avg.setText(f"Điểm trung bình: {avg:.2f}")
        
        gioi = stats.get('gioi', 0)
        kha = stats.get('kha', 0)
        tb = stats.get('tb', 0)
        yeu = stats.get('yeu', 0)
        
        def update_bar(key, count):
            lbl, bar = self.bars[key]
            lbl.setText(f"{key}: {count}")
            pct = int((count / total * 100)) if total > 0 else 0
            bar.setValue(pct)
            
        update_bar("Giỏi (>=8)", gioi)
        update_bar("Khá (6.5-8)", kha)
        update_bar("TB (5-6.5)", tb)
        update_bar("Yếu (<5)", yeu)

    def update_single_result(self, score: dict):
        self.single_result_container.show()
        
        final_score = score.get('final_score', 0)
        self.lbl_score_huge.setText(f"{final_score:.2f}")
        
        if final_score >= 8:
            level = "gioi"
        elif final_score >= 6.5:
            level = "kha"
        elif final_score >= 5:
            level = "tb"
        else:
            level = "yeu"
            
        self.single_result_container.setProperty("scoreLevel", level)
        self.lbl_score_huge.setProperty("scoreLevel", level)
        
        self.single_result_container.style().unpolish(self.single_result_container)
        self.single_result_container.style().polish(self.single_result_container)
        
        self.lbl_score_huge.style().unpolish(self.lbl_score_huge)
        self.lbl_score_huge.style().polish(self.lbl_score_huge)
        
        correct = score.get('correct', 0)
        total = score.get('total', 0)
        self.lbl_correct_count.setText(f"Số câu đúng: {correct}/{total}")
        
        self.lbl_sbd.setText(f"Số báo danh: {score.get('sbd', 'N/A')}")
        self.lbl_made.setText(f"Mã đề thi: {score.get('made', 'N/A')}")
        
        notes = score.get('notes', '')
        if notes:
            self.lbl_notes.setText(notes)
            self.lbl_notes.show()
        else:
            self.lbl_notes.hide()

    def set_processing_enabled(self, enabled: bool):
        pass

    def set_animation_mode(self, running: bool) -> None:
        pass
