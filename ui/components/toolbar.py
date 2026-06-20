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

