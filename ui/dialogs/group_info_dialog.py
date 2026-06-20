from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout)
from PyQt6.QtCore import Qt

class GroupInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thông tin nhóm")
        self.resize(400, 300)
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Title label
        title_label = QLabel("Chương trình chấm phiếu trắc nghiệm")
        title_label.setProperty("class", "title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Group info labels
        group_label = QLabel("Nhóm 17")
        group_label.setProperty("class", "subtitle")
        group_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(group_label)
        
        members = [
            "Nguyễn Thế Tân - 23110152",
            "Nguyễn Đức Tâm - 23110151",
            "Lưu Quang Tiến - 23110157"
        ]
        
        for member in members:
            member_label = QLabel(member)
            member_label.setProperty("class", "member-name")
            member_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(member_label)
        
        main_layout.addStretch()
        
        # Buttons layout
        btn_layout = QHBoxLayout()
        
        # Demo button
        demo_btn = QPushButton("Demo")
        demo_btn.clicked.connect(self.accept)
        demo_btn.setProperty("variant", "success")
        btn_layout.addWidget(demo_btn)
        
        # Close button
        close_btn = QPushButton("Đóng")
        close_btn.clicked.connect(self.reject)
        close_btn.setProperty("variant", "secondary")
        btn_layout.addWidget(close_btn)
        
        main_layout.addLayout(btn_layout)
