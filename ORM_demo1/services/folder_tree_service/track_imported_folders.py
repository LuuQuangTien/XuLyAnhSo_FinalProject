# Theo dõi và quản lý danh sách các thư mục đã được nhập vào ứng dụng.
class ImportedFoldersTracker:
    def __init__(self):
        self._folders = set()

    def can_import(self, folder_path):
        return folder_path not in self._folders

    def add(self, folder_path):
        self._folders.add(folder_path)

    def remove(self, folder_path):
        self._folders.discard(folder_path)
