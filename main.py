# Điểm vào chính của ứng dụng (Entry point). Khởi tạo giao diện và các bộ điều phối (Controllers).
import sys
import os
import ctypes

# Import onnxruntime TRƯỚC PyQt6 để tránh xung đột DLL trên Windows
try:
    import onnxruntime
except Exception:
    pass

from PyQt6.QtWidgets import QApplication

# Bây giờ có thể import từ folder ui ở gốc
from ui.main_window import MainWindow
from ui.dialogs.group_info_dialog import GroupInfoDialog

def main():

    myappid = 'mycompany.myproduct.subproduct.version'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    
    app = QApplication(sys.argv)
    
    # Apply global styles
    from ui.styles import GLOBAL_STYLE
    app.setStyleSheet(GLOBAL_STYLE)
    
    # KIỂM TRA VÀ TẢI MÔ HÌNH AI (Nếu chưa có)
    from utils.model_downloader import download_models_if_needed
    download_models_if_needed()
            
    # Show group info dialog first
    intro_dialog = GroupInfoDialog()
    if intro_dialog.exec() == GroupInfoDialog.DialogCode.Accepted:
        # Only show main window after intro dialog is closed
        window = MainWindow()
        
        # Initialize the Controller to wire signals and business logic
        from controllers.main_controller import MainController
        controller = MainController(window)
        
        window.show()
        
        sys.exit(app.exec())

if __name__ == "__main__":
    main()
