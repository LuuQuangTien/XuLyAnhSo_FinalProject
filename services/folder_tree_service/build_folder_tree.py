# Quét và xây dựng cấu trúc cây thư mục từ đường dẫn trên ổ cứng.
import os
from services.folder_tree_service.check_image_file import is_image_file


def build_folder_tree(folder_path):
    root_name = os.path.basename(folder_path)
    return {
        "name": root_name,
        "path": folder_path,
        "type": "directory",
        "children": _scan_directory(folder_path),
    }


def _scan_directory(dir_path):
    nodes = []
    try:
        items = sorted(os.listdir(dir_path))
        for item_name in items:
            if item_name.startswith("."):
                continue

            item_path = os.path.join(dir_path, item_name)
            if os.path.isdir(item_path):
                nodes.append(
                    {
                        "name": item_name,
                        "path": item_path,
                        "type": "directory",
                        "children": _scan_directory(item_path),
                    }
                )
            elif os.path.isfile(item_path) and is_image_file(item_name):
                nodes.append(
                    {
                        "name": item_name,
                        "path": item_path,
                        "type": "file",
                        "children": [],
                    }
                )
    except Exception as e:
        print(f"Error reading directory: {e}")
    return nodes
