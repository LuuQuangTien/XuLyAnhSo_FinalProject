import os
import pandas as pd
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, 
    QPushButton, QTableWidget, QTableWidgetItem, QWidget, 
    QMessageBox, QHeaderView, QFileDialog, QInputDialog
)
from PyQt6.QtCore import Qt
from services.answer_key_service import AnswerKeyService

class AnswerKeyDialog(QDialog):
    def __init__(self, parent=None, excel_path="assets/sample_answers.xlsx", initial_q_count=40):
        super().__init__(parent)
        self.setWindowTitle("Thiết lập Đáp án OMR (Excel)")
        self.resize(700, 700)
        
        self.excel_path = excel_path
        os.makedirs(os.path.dirname(self.excel_path), exist_ok=True)
        
        self.total_questions = initial_q_count
        self._setup_ui()
        self.generate_table()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # --- Top Controls ---
        top_layout = QHBoxLayout()
        
        btn_add_made = QPushButton("+ Thêm Mã Đề")
        btn_add_made.clicked.connect(self.add_made_column)
        
        top_layout.addWidget(btn_add_made)
        top_layout.addStretch()
        
        main_layout.addLayout(top_layout)
        
        # --- Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Câu hỏi", "Mã Đề 000"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(160)
        self.table.setColumnWidth(0, 80)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self.rename_column)
        
        main_layout.addWidget(self.table)
        
        # --- Bottom Controls ---
        bottom_layout = QHBoxLayout()
        
        btn_load_file = QPushButton("Nạp từ Excel...")
        btn_load_file.clicked.connect(self.prompt_load_excel)
        
        btn_scan_image = QPushButton("Quét từ Ảnh...")
        btn_scan_image.setStyleSheet("background-color: #2196F3; color: white;")
        btn_scan_image.clicked.connect(self.scan_from_image)
        
        btn_save = QPushButton("Lưu Excel & Áp dụng")
        btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.save_and_apply)
        
        btn_cancel = QPushButton("Hủy")
        btn_cancel.clicked.connect(self.reject)
        
        bottom_layout.addWidget(btn_scan_image)
        bottom_layout.addWidget(btn_load_file)
        bottom_layout.addStretch()
        bottom_layout.addWidget(btn_cancel)
        bottom_layout.addWidget(btn_save)
        
        main_layout.addLayout(bottom_layout)

    def generate_table(self):
        """Tạo lại bảng dữ liệu trống."""
        self.table.setRowCount(self.total_questions)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Câu hỏi", "Mã Đề 000"])
        
        for i in range(self.total_questions):
            item_num = QTableWidgetItem(str(i + 1))
            item_num.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item_num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 0, item_num)
            
            from ui.components.bubble_answer_widget import BubbleAnswerWidget
            widget = BubbleAnswerWidget(options="ABCD")
            self.table.setCellWidget(i, 1, widget)
            self.table.setRowHeight(i, 40)

    def rename_column(self, col_idx):
        """Cho phép người dùng đổi tên Mã Đề khi click đúp vào tiêu đề cột."""
        if col_idx == 0: return # Không cho đổi tên cột "Câu hỏi"
        old_name = self.table.horizontalHeaderItem(col_idx).text()
        new_name, ok = QInputDialog.getText(self, "Đổi tên Mã Đề", "Nhập mã đề thi mới (VD: 101, 102):", text=old_name)
        if ok and new_name.strip():
            self.table.horizontalHeaderItem(col_idx).setText(new_name.strip())

    def add_made_column(self):
        """Thêm 1 cột mã đề mới."""
        col_idx = self.table.columnCount()
        
        # Hỏi tên mã đề trước khi tạo
        new_name, ok = QInputDialog.getText(self, "Thêm Mã Đề", "Nhập mã đề (VD: 101, 102):", text=f"Mã Đề {col_idx}")
        if not ok or not new_name.strip():
            return
            
        self.table.insertColumn(col_idx)
        self.table.setHorizontalHeaderItem(col_idx, QTableWidgetItem(new_name.strip()))
        for i in range(self.table.rowCount()):
            from ui.components.bubble_answer_widget import BubbleAnswerWidget
            widget = BubbleAnswerWidget(options="ABCD")
            self.table.setCellWidget(i, col_idx, widget)

    def get_dataframe(self):
        """Chuyển đổi dữ liệu bảng QTableWidget thành pandas DataFrame."""
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        
        headers = [self.table.horizontalHeaderItem(j).text() for j in range(cols)]
        data = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                if j == 0:
                    item = self.table.item(i, j)
                    row_data.append(item.text().strip() if item else "")
                else:
                    widget = self.table.cellWidget(i, j)
                    row_data.append(widget.get_answer() if widget else "")
            data.append(row_data)
            
        return pd.DataFrame(data, columns=headers)

    def validate_and_format_table(self):
        """Kiểm tra số lượng ô trống (không cần validate ký tự nữa vì đã dùng Widget)."""
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        missing_count = 0
        
        for i in range(rows):
            for j in range(1, cols):
                widget = self.table.cellWidget(i, j)
                if widget and not widget.get_answer():
                    missing_count += 1
                
        return missing_count

    def save_and_apply(self):
        missing_count = self.validate_and_format_table()
            
        if missing_count > 0:
            reply = QMessageBox.question(
                self, "Cảnh báo thiếu đáp án", 
                f"Phát hiện {missing_count} ô trống chưa nhập đáp án.\nBạn có muốn tiếp tục lưu không?", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        df = self.get_dataframe()
        
        default_dir = os.path.dirname(self.excel_path)
        path, _ = QFileDialog.getSaveFileName(self, "Lưu file đáp án", default_dir, "Excel Files (*.xlsx)")
        
        if not path: return
        if not path.lower().endswith('.xlsx'): path += '.xlsx'

        try:
            AnswerKeyService.save_dataframe(df, path)
            QMessageBox.information(self, "Thành công", f"Đã lưu đáp án thành công vào:\n{path}")
            self.excel_path = path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu file Excel:\n{e}")

    def prompt_load_excel(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file Excel đáp án", "", "Excel Files (*.xlsx *.xls)")
        if path:
            self.load_from_excel(path, default_if_missing=False)

    def scan_from_image(self):
        import cv2
        from services.omr_service import OMRService
        
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh phiếu đáp án", "", "Images (*.png *.jpg *.jpeg)")
        if not path: return
            
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("Không thể đọc file ảnh.")
                
            pre_scan = OMRService.pre_scan_image(image)
            if not pre_scan['is_valid']:
                QMessageBox.warning(self, "Lỗi quét ảnh", pre_scan['error'])
                return
                
            template = pre_scan['template']
            made = pre_scan['made']
            if not made:
                made, ok = QInputDialog.getText(self, "Nhập Mã Đề", "Không đọc được mã đề trên phiếu. Vui lòng nhập tay:")
                if not ok or not made.strip(): return
                made = made.strip()
                
            # Grade with empty answers to extract raw bubbles
            _, _, score_dict = OMRService.grade_image(image, answers={}, template=template)
            raw_answers = score_dict.get('raw_answers', {})
            
            clean_answers = {}
            for q, ans in raw_answers.items():
                if ans and ans != "Chưa chọn":
                    clean_answers[int(q)] = ans
                    
            if not clean_answers:
                QMessageBox.warning(self, "Lỗi", "Không tìm thấy đáp án nào được tô trên phiếu!")
                return
                
            cols = self.table.columnCount()
            target_col = -1
            for j in range(1, cols):
                if self.table.horizontalHeaderItem(j).text() == made:
                    target_col = j
                    break
                    
            if target_col == -1:
                target_col = cols
                self.table.insertColumn(target_col)
                self.table.setHorizontalHeaderItem(target_col, QTableWidgetItem(made))
                for i in range(self.table.rowCount()):
                    from ui.components.bubble_answer_widget import BubbleAnswerWidget
                    widget = BubbleAnswerWidget(options="ABCD")
                    self.table.setCellWidget(i, target_col, widget)
                    
            # Fill data
            for i in range(self.table.rowCount()):
                q_num = i + 1
                ans = clean_answers.get(q_num, "")
                widget = self.table.cellWidget(i, target_col)
                if widget:
                    widget.set_answer(ans)
                    
            QMessageBox.information(self, "Thành công", f"Đã tạo đáp án cho Mã đề: {made}")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi xử lý ảnh", str(e))

    def load_from_excel(self, path, default_if_missing=False):
        if not os.path.exists(path):
            if default_if_missing: self.generate_table()
            return

        try:
            df = AnswerKeyService.get_dataframe(path)
            if df.empty:
                if default_if_missing: self.generate_table()
                return

            df = df.astype(str) # Convert all to string
            self.total_questions = len(df)
            
            headers = df.columns.tolist()
            self.table.setColumnCount(len(headers))
            self.table.setRowCount(self.total_questions)
            self.table.setHorizontalHeaderLabels(headers)
            
            for i, row in df.iterrows():
                self.table.setRowHeight(i, 40)
                for j, col_name in enumerate(headers):
                    val = str(row[col_name]).replace('nan', '').strip()
                    if j == 0: # Cột đầu tiên (Câu hỏi)
                        item = QTableWidgetItem(val)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                        self.table.setItem(i, j, item)
                    else:
                        from ui.components.bubble_answer_widget import BubbleAnswerWidget
                        widget = BubbleAnswerWidget(options="ABCD", initial_ans=val)
                        self.table.setCellWidget(i, j, widget)
                    
            self.excel_path = path
        except Exception as e:
            QMessageBox.critical(self, "Lỗi nạp file", f"File Excel không hợp lệ:\n{e}")
            if default_if_missing: self.generate_table()
