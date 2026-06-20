# Giao diện thanh công cụ (Toolbar) phía trên màn hình.
import os
from PyQt6.QtWidgets import QToolBar, QToolButton, QMenu, QLabel, QWidget
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt
from ui import strings


class AppToolBar(QToolBar):
    """Toolbar UI component. Business actions are wired externally."""

    def __init__(self):
        super().__init__()
        self.setMovable(False)
        
        # File Menu Button
        self.file_button = QToolButton()
        self.file_button.setText(strings.MENU_FILE)

        self.file_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.addWidget(self.file_button)

        # File Actions
        self.action_import_folder = QAction(strings.MENU_IMPORT_FOLDER, self)
        self.action_import_file = QAction(strings.MENU_IMPORT_FILE, self)
        self.action_save = QAction(strings.MENU_SAVE, self)
        self.action_save_as = QAction(strings.MENU_SAVE_AS, self)
        self.action_save_all = QAction(strings.MENU_SAVE_ALL, self)
        self.action_exit = QAction(strings.MENU_EXIT, self)
        
        self.file_menu = QMenu()
        self.file_menu.addAction(self.action_import_folder)
        self.file_menu.addAction(self.action_import_file)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.action_save)
        self.file_menu.addAction(self.action_save_as)
        self.file_menu.addAction(self.action_save_all)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.action_exit)
        self.file_button.setMenu(self.file_menu)

        self.addSeparator()

        # Theme Menu Button
        self.theme_button = QToolButton()
        self.theme_button.setText(strings.MENU_THEME)

        self.theme_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icons_dir = os.path.join(base_dir, '..', '..', 'assets', 'icons')
        
        # Load Icons
        self.icon_light = QIcon(os.path.join(icons_dir, 'appiconlight.png'))
        self.icon_dark = QIcon(os.path.join(icons_dir, 'appicondark.png'))
        self.icon_warm = QIcon(os.path.join(icons_dir, 'appiconwarm.png'))
        self.icon_cold = QIcon(os.path.join(icons_dir, 'appiconcold.png'))

        # Set default icon
        self.theme_button.setIcon(self.icon_dark)
        self.addWidget(self.theme_button)

        # Theme Actions
        self.action_theme_light = QAction(self.icon_light, strings.THEME_LIGHT, self)
        self.action_theme_dark = QAction(self.icon_dark, strings.THEME_DARK, self)
        self.action_theme_warm = QAction(self.icon_warm, strings.THEME_WARM, self)
        self.action_theme_cold = QAction(self.icon_cold, strings.THEME_COLD, self)

        self.theme_menu = QMenu()
        self.theme_menu.addAction(self.action_theme_light)
        self.theme_menu.addAction(self.action_theme_dark)
        self.theme_menu.addAction(self.action_theme_warm)
        self.theme_menu.addAction(self.action_theme_cold)
        self.theme_button.setMenu(self.theme_menu)

        self.addSeparator()



        # Lấy Style của GUI để lấy Icon mặc định của Qt
        style = self.style()
        icon_single = style.standardIcon(style.StandardPixmap.SP_FileDialogDetailedView)
        icon_batch = style.standardIcon(style.StandardPixmap.SP_DirIcon)
        icon_settings = style.standardIcon(style.StandardPixmap.SP_FileDialogListView)

        # OMR Processing Button
        self.action_omr = QAction(icon_single, "Chấm thi ảnh hiện tại", self)
        self.addAction(self.action_omr)

        self.action_omr_batch = QAction(icon_batch, "Chấm thi hàng loạt (Thư mục)", self)
        self.addAction(self.action_omr_batch)
        
        self.addSeparator()

        # Answer Key Editor Button
        self.action_edit_answers = QAction(icon_settings, "Thiết lập Đáp án", self)
        self.addAction(self.action_edit_answers)

        # Spacer to push everything else to the left
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().Policy.Expanding, spacer.sizePolicy().Policy.Preferred)
        self.addWidget(spacer)

