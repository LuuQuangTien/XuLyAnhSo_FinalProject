# Lọc và trích xuất thông tin các file ảnh hợp lệ từ danh sách file được chọn.
import os
from .check_image_file import is_image_file

def execute(files):
    """
    Processes a list of file paths and returns a list of valid image data.
    
    Args:
        files: List of absolute file paths.
        
    Returns:
        List of dictionaries containing name and path of valid images.
    """
    valid_images = []
    for f in files:
        filename = os.path.basename(f)
        if is_image_file(filename):
            valid_images.append({
                "name": filename,
                "path": f
            })
    return valid_images
