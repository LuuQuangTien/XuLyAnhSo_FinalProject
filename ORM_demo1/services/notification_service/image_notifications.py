# Các hàm hỗ trợ hiển thị hộp thoại thông báo (lỗi, cảnh báo, thành công).
from PyQt6.QtWidgets import QMessageBox
from ui import strings

def warn_no_image_selected(parent=None):
    """Shows a warning when no image is selected for processing or saving."""
    QMessageBox.warning(parent, strings.APP_NAME, strings.MSG_NO_IMAGE_WARNING)

def info_save_success(parent=None, file_path=""):
    """Shows a success message after saving an image."""
    msg = strings.MSG_SAVE_SUCCESS
    if file_path:
        msg = strings.MSG_SAVE_AS_SUCCESS.format(file_path)
    QMessageBox.information(parent, strings.APP_NAME, msg)

def warn_save_error(parent=None, error_msg=""):
    """Shows an error message if saving fails."""
    title = strings.MSG_SAVE_ERROR_TITLE
    QMessageBox.critical(parent, title, error_msg if error_msg else "Unknown error occurred while saving.")

def ask_save_changes(parent=None, file_name=""):
    """Asks user to save changes before closing a specific tab."""
    msg = strings.MSG_UNSAVED_TAB.format(file_name)
    reply = QMessageBox.question(
        parent,
        strings.APP_NAME,
        msg,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Yes
    )
    return reply

def ask_save_all_exit(parent=None, unsaved_count=0):
    """Asks user to save all changes before exiting the app."""
    msg = strings.MSG_UNSAVED_APP_EXIT.format(unsaved_count)
    reply = QMessageBox.question(
        parent,
        strings.APP_NAME,
        msg,
        QMessageBox.StandardButton.SaveAll | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.SaveAll
    )
    return reply


def ask_reset_all(parent=None) -> bool:
    """Xác nhận đưa ảnh về bước gốc (xóa các bước lịch sử sau đó)."""
    reply = QMessageBox.question(
        parent,
        strings.APP_NAME,
        strings.MSG_RESET_CONFIRM,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return reply == QMessageBox.StandardButton.Yes
