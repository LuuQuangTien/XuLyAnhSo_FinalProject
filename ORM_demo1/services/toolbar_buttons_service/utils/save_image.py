# Hàm hỗ trợ lưu ảnh xuống ổ cứng (Hỗ trợ đường dẫn Unicode).
import os
import cv2

def execute(image, output_path):
    """
    Saves an image to the specified path with Unicode-safe handling.
    """
    if image is None or not output_path:
        return False

    ext = os.path.splitext(output_path)[1].lower() or ".png"
    if ext == ".jpg":
        ext = ".jpeg"

    try:
        success, encoded = cv2.imencode(ext, image)
        if not success:
            return False

        encoded.tofile(output_path)
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False
