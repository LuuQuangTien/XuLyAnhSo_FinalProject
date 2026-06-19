# Hàm hỗ trợ tải stylesheet cho giao diện ứng dụng.
# Load base.qss (cấu trúc chung) + theme.qss (bảng màu) rồi ghép lại.
import os

_THEMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'themes')

def _read_file(filename):
    path = os.path.join(_THEMES_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def load_stylesheet(theme_name="dark"):
    """Load base.qss + {theme_name}.qss và ghép lại thành stylesheet hoàn chỉnh."""
    base = _read_file("base.qss")
    theme = _read_file(f"{theme_name}.qss")
    return base + "\n" + theme

GLOBAL_STYLE = load_stylesheet("dark")
