# Giao diện cây thư mục bên trái màn hình.
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QTreeView, QAbstractItemView, QLabel, QMenu
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
import os

from ui import strings
from services.folder_tree_service import (
    ImportedFoldersTracker, 
    remove_folder
)

class LeftPanel(QFrame):
    image_selected = pyqtSignal(str)
    folder_removed_requested = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        self.setMinimumWidth(250)
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)
        
        # Tiêu đề
        title = QLabel(strings.LBL_EXPLORER)
        title.setProperty("class", "explorer-title")
        self.main_layout.addWidget(title)
        
        # Cấu trúc cây thư mục
        self.setup_tree_view()

    def setup_tree_view(self):
        self.tree_model = QStandardItemModel()
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree_view.setIndentation(20)
        self.tree_view.setObjectName("fileTree")
        
        # Bật menu ngữ cảnh (chuột phải)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.show_context_menu)
        
        self.tree_view.clicked.connect(self.on_item_clicked)
        self.main_layout.addWidget(self.tree_view)

    def show_context_menu(self, position: QPoint):
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            return
            
        item = self.tree_model.itemFromIndex(index)
        if not item:
            return
            
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)
        
        # Chỉ hiện menu "Remove" cho thư mục gốc hoặc thư mục được import
        if item.parent() is None:
            menu = QMenu()
            remove_action = QAction(strings.MENU_REMOVE_FOLDER, self)
            remove_action.triggered.connect(lambda: self.remove_root_item(item))
            menu.addAction(remove_action)
            menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def remove_root_item(self, item):
        folder_path = item.data(Qt.ItemDataRole.UserRole)
        # Emit signal to Controller instead of executing business logic directly
        self.folder_removed_requested.emit(folder_path)
        
        row = item.row()
        self.tree_model.removeRow(row)

    def add_folder_by_data(self, tree_data):
        """Pure UI method to add a folder node to the tree."""
        root_item = self._build_item_from_node(tree_data)
        self.tree_model.appendRow(root_item)
        self.tree_view.expand(root_item.index())

    def _build_item_from_node(self, node):
        item = QStandardItem(node["name"])
        item.setData(node["path"], Qt.ItemDataRole.UserRole)
        item.setData(node["type"], Qt.ItemDataRole.UserRole + 1)

        for child in node.get("children", []):
            item.appendRow(self._build_item_from_node(child))

        return item

    def add_files_to_tree(self, processed_data):
        """Pure UI method to add files to the 'Imported Files' node."""
        root_item = self._get_or_create_imported_files_node()

        for img in processed_data:
            file_item = QStandardItem(img["name"])
            file_item.setData(img["path"], Qt.ItemDataRole.UserRole)
            file_item.setData("file", Qt.ItemDataRole.UserRole + 1)
            root_item.appendRow(file_item)
        
        self.tree_view.expand(root_item.index())

    def _get_or_create_imported_files_node(self):
        for i in range(self.tree_model.rowCount()):
            item = self.tree_model.item(i)
            if item.text() == strings.LBL_IMPORTED_FILES:
                return item
        
        root_item = QStandardItem(strings.LBL_IMPORTED_FILES)
        root_item.setData("virtual", Qt.ItemDataRole.UserRole + 1)
        self.tree_model.appendRow(root_item)
        return root_item

    def on_item_clicked(self, index):
        item = self.tree_model.itemFromIndex(index)
        if item:
            item_type = item.data(Qt.ItemDataRole.UserRole + 1)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if item_type == "file" and file_path:
                self.image_selected.emit(file_path)
