# Quản lý giao diện màu sắc của ứng dụng (Theme/Màu áo).
from PyQt6.QtWidgets import QApplication
from ui import styles

class ThemeController:
    def __init__(self, main_window):
        self.view = main_window

    def handle_theme_change(self, theme_name):
        """
        Loads the corresponding QSS and updates the application style and toolbar icon.
        """
        # Load the stylesheet
        qss = styles.load_stylesheet(theme_name)
        if qss:
            # Apply to the entire application
            QApplication.instance().setStyleSheet(qss)
            
            # Update the toolbar icon to reflect the current theme
            toolbar = self.view.toolbar_component
            if theme_name == "light":
                toolbar.theme_button.setIcon(toolbar.icon_light)
            elif theme_name == "dark":
                toolbar.theme_button.setIcon(toolbar.icon_dark)
            elif theme_name == "warm":
                toolbar.theme_button.setIcon(toolbar.icon_warm)
            elif theme_name == "cold":
                toolbar.theme_button.setIcon(toolbar.icon_cold)
