# Chuyển đổi định dạng ảnh từ OpenCV (NumPy) sang PyQt (QPixmap/QImage).
from PyQt6.QtGui import QImage, QPixmap
import cv2


def cv_image_to_qimage(cv_img):
    if cv_img is None:
        return None

    if len(cv_img.shape) == 2:  # Grayscale
        height, width = cv_img.shape
        bytes_per_line = width
        return QImage(
            cv_img.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8,
        ).copy()

    height, width, _ = cv_img.shape
    bytes_per_line = 3 * width
    cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return QImage(
        cv_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()


def cv_image_to_qpixmap(cv_img):
    q_img = cv_image_to_qimage(cv_img)
    if q_img is None:
        return None
    return QPixmap.fromImage(q_img)
