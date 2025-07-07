from PyQt6.QtWidgets import (
   QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QFrame, QLabel, QToolButton, QComboBox, QPushButton, QSizePolicy,
    QSpinBox, QCheckBox, QColorDialog
)
from PyQt6.QtGui import (
    QColor, QBrush, QPen, QPainter,
    QIcon, QMouseEvent,
)
from PyQt6.QtCore import Qt, QSize

from settings.Settings import Settings

class AppearanceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # === Plot Theme ComboBox ===
        theme_label = QLabel("Plot Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentText(Settings.plot_theme["NAME"])
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)

        layout.addWidget(theme_label)
        layout.addWidget(self.theme_combo)

        # === Data Line Color Picker ===
        color_label = QLabel("Data Line Color:")
        self.color_button = QPushButton()
        self.color_button.setText(Settings.data_line_color.name().upper())
        self.color_button.setStyleSheet(f"background-color: {Settings.data_line_color.name()}")
        self.color_button.clicked.connect(self.choose_color)

        layout.addWidget(color_label)
        layout.addWidget(self.color_button)

        # === Data Line Width SpinBox ===
        width_label = QLabel("Data Line Width:")
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 5)
        self.width_spinbox.setValue(Settings.data_line_width)
        self.width_spinbox.valueChanged.connect(self.on_width_changed)

        layout.addWidget(width_label)
        layout.addWidget(self.width_spinbox)

        # === Boolean Toggles ===
        self.checkboxes = {}
        bool_settings = {
            "Show Horizontal Grid": "show_h_grid",
            "Show Vertical Grid": "show_v_grid",
            "Show Labels": "show_labels",
            "Show Durations": "show_durations",
            "Show Comments": "show_comments"
        }

        for label_text, attr in bool_settings.items():
            checkbox = QCheckBox(label_text)
            checkbox.setChecked(getattr(Settings, attr))
            checkbox.toggled.connect(lambda state, a=attr: (print(getattr(Settings, attr)), setattr(Settings, a, state)))
            print(getattr(Settings, attr))
            self.checkboxes[attr] = checkbox
            layout.addWidget(checkbox)

    

    def choose_color(self):
        color = QColorDialog.getColor(initial=self.data_line_color, parent=self)
        if color.isValid():
            Settings.data_line_color = color
            self.color_button.setText(color.name())
            self.color_button.setStyleSheet(f"background-color: {color.name()}")
            print("Data line color set to:", color.name())

    def on_theme_changed(self, value: str):
        Settings.plot_theme = value
        print("Theme set to:", value)
    def on_width_changed(self, value: int):
        Settings.data_line_width = value
        print("Data line width set to:", value)



class EPGSettingsTab(QWidget):
    pass



class SidebarButton(QToolButton):
    def __init__(self, text: str, index: int, icon_path: str = None, parent=None):
        super().__init__(parent)
        self.index = index
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        if icon_path:
            self.setIcon(QIcon(icon_path))
            self.setIconSize(QSize(32, 32))
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        else:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        self.setStyleSheet("""
            QToolButton {
                background-color: #2b2b2b;
                color: white;
                padding: 6px 10px 6px 10px;
                text-align: left;
                border: none;
                font-size: 10pt;
                font-weight: normal;
            }
            QToolButton:hover {
                background-color: rgba(32, 147, 254, 0.15);
            }
            QToolButton:pressed {
                background-color: rgba(32, 147, 254, 0.35);
            }
            QToolButton:checked {
                background-color: rgba(32, 147, 254, 0.25);
                border-left: 4px solid #2093FE;
                padding-left: 6px;
                font-weight: 600;
            }
        """)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setMinimumSize(800, 600)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar_frame = QFrame()
        self.sidebar_frame.setFixedWidth(180)
        self.sidebar_frame.setStyleSheet("background-color: #2b2b2b;")
        self.sidebar_layout = QVBoxLayout(self.sidebar_frame)
        self.sidebar_layout.setContentsMargins(0, 12, 0, 0)
        self.sidebar_layout.setSpacing(0)

        button_info = [
            ("Appearance", None),  # Use valid icon paths or leave empty
            ("Test", None)
        ]

        self.buttons = []
        for index, (label, icon) in enumerate(button_info):
            btn = SidebarButton(label, index, icon)
            btn.clicked.connect(lambda _, i=index: self.switch_tab(i))
            self.sidebar_layout.addWidget(btn)
            self.buttons.append(btn)

        self.sidebar_layout.addStretch()
        self.buttons[0].setChecked(True)

        self.stack = QStackedWidget()
        self.stack.addWidget(AppearanceTab(self.stack))
        self.stack.addWidget(self._create_test_tab())

        main_layout.addWidget(self.sidebar_frame)
        main_layout.addWidget(self.stack)

    def switch_tab(self, index: int):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)

    def _create_test_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel("EPG Settings")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return tab