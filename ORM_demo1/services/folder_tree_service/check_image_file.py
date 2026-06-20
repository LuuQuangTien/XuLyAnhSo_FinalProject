# Kiểm tra định dạng file có phải là hình ảnh hợp lệ hay không.
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def is_image_file(filename):
    return filename.lower().endswith(VALID_IMAGE_EXTENSIONS)
