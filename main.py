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
    import urllib.request
    from PyQt6.QtWidgets import QProgressDialog, QMessageBox
    from PyQt6.QtCore import Qt
    
    models_dir = os.path.join(os.getcwd(), 'assets', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    models_to_download = [
        {
            "name": "RealESR-AnimeVideo-v3_x4.onnx",
            "url": "https://drive.google.com/uc?id=1UzASVotKdKCfohxbFsIIRmo7OjJZqxuW&export=download",
            "desc": "AI Model Khôi Phục Ảnh (2.5MB)"
        },
        {
            "name": "silueta.onnx",
            "url": "https://drive.google.com/uc?id=1u_n3GtpMF9AWt71bskxof_uGNvBAN2l8&export=download",
            "desc": "AI Model Xóa Nền (44MB)"
        }
    ]
    
    for model_info in models_to_download:
        model_path = os.path.join(models_dir, model_info["name"])
        if not os.path.exists(model_path):
            url = model_info["url"]
            progress = QProgressDialog(f"Đang tải {model_info['desc']}... Vui lòng chờ", "Hủy", 0, 100)
            progress.setWindowTitle("Tải Dữ Liệu AI")
            progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            
            def reporthook(blocknum, blocksize, totalsize):
                if progress.wasCanceled():
                    raise Exception("Người dùng đã hủy tải.")
                if totalsize > 0:
                    percent = int(blocknum * blocksize * 100 / totalsize)
                    progress.setValue(min(percent, 100))
                else:
                    if progress.maximum() != 0:
                        progress.setMaximum(0)
                QApplication.processEvents()
                
            try:
                urllib.request.urlretrieve(url, model_path, reporthook)
            except Exception as e:
                if os.path.exists(model_path):
                    os.remove(model_path) # Xóa file lỗi đang tải dở
                QMessageBox.warning(None, "Lỗi Tải Model", f"Không thể tải {model_info['name']}: {e}\nỨng dụng vẫn chạy nhưng tính năng AI tương ứng sẽ không khả dụng.")
            
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
