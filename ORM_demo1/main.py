# Điểm vào chính của ứng dụng (Entry point). Khởi tạo giao diện và các bộ điều phối (Controllers).
import sys
import os
import ctypes
from PyQt6.QtWidgets import QApplication

# Bây giờ có thể import từ folder ui ở gốc
from ui.main_window import MainWindow

def main():

    myappid = 'mycompany.myproduct.subproduct.version'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    
    app = QApplication(sys.argv)
    
    # Apply global styles
    from ui.styles import GLOBAL_STYLE
    app.setStyleSheet(GLOBAL_STYLE)
    
    window = MainWindow()
    
    # Initialize the Controller to wire signals and business logic
    from controllers.main_controller import MainController
    controller = MainController(window)
    
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
