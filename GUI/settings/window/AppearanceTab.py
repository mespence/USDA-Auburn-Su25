from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QButtonGroup
from PyQt6.QtCore import Qt

from settings.window.ThemePreview import UIThemeOption
from settings.window.ThemeRenderer import create_theme_preview
from settings.Settings import Settings

class AppearanceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)

        label = QLabel("Themes")
        label.setStyleSheet("font-family: 'Inter'; font-size: 14px; font-weight: 600;")
        main_layout.addWidget(label)
        main_layout.addSpacing(15)

        theme_group = QButtonGroup(self)
        theme_group.setExclusive(True)

        light_preview = create_theme_preview(Settings.LIGHT)
        dark_preview = create_theme_preview(Settings.DARK)

        light_option = UIThemeOption(Settings.LIGHT, light_preview)
        dark_option = UIThemeOption(Settings.DARK, dark_preview)

        theme_group.addButton(light_option.radio)
        theme_group.addButton(dark_option.radio)

        theme_layout = QHBoxLayout()
        theme_layout.addWidget(light_option, alignment=Qt.AlignmentFlag.AlignLeft)
        theme_layout.addWidget(dark_option, alignment=Qt.AlignmentFlag.AlignLeft)

        main_layout.addLayout(theme_layout)
        main_layout.addStretch()
        self.setLayout(main_layout)