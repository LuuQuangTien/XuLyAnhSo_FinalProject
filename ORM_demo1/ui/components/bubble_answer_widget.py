from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt

class BubbleAnswerWidget(QWidget):
    answer_changed = pyqtSignal(str)
    
    def __init__(self, parent=None, options="ABCDEF", initial_ans=""):
        super().__init__(parent)
        self.options = list(options)
        self.selected = set([c for c in initial_ans if c in self.options])
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        
        layout.addStretch() # Đẩy vào giữa từ bên trái
        
        self.buttons = {}
        for opt in self.options:
            btn = QPushButton(opt)
            btn.setFixedSize(28, 28) # Tăng kích thước chút xíu
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Xóa viền chấm khi focus (click)
            
            if opt in self.selected:
                btn.setChecked(True)
                
            # Đổi font Arial để rõ nét, bỏ border-radius nếu không thích, hoặc giữ 14px để tròn
            btn.setStyleSheet("""
                QPushButton {
                    border-radius: 14px;
                    border: 1px solid #777;
                    background-color: transparent;
                    color: #ccc;
                    font-weight: bold;
                    font-family: Arial, sans-serif;
                    font-size: 13px;
                    padding: 0px;
                    margin: 0px;
                }
                QPushButton:checked {
                    background-color: #2196F3;
                    border: 1px solid #2196F3;
                    color: white;
                }
                QPushButton:hover:!checked {
                    background-color: #444;
                    color: white;
                }
            """)
            btn.clicked.connect(self._create_toggle_handler(opt))
            
            self.buttons[opt] = btn
            layout.addWidget(btn)
            
        layout.addStretch() # Đẩy vào giữa từ bên phải

    def _create_toggle_handler(self, opt):
        def handler(checked):
            if checked:
                self.selected.add(opt)
            else:
                self.selected.discard(opt)
            
            ans_str = ",".join(sorted(list(self.selected)))
            self.answer_changed.emit(ans_str)
        return handler

    def get_answer(self):
        return ",".join(sorted(list(self.selected)))
        
    def set_answer(self, ans_str):
        self.selected = set([c for c in ans_str if c in self.options])
        for opt, btn in self.buttons.items():
            btn.setChecked(opt in self.selected)
