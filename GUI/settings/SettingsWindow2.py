from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QFrame, QLabel, QToolButton, QSizePolicy
)
from PyQt6.QtGui import (
    QColor, QBrush, QPen, QPainter,
    QIcon, QMouseEvent,
)
from PyQt6.QtCore import Qt, QSize
import sys

# === Dummy AppearanceTab for MWE ===
class AppearanceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("Appearance Settings")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

class SidebarButton(QToolButton):
    def __init__(self, text: str, icon_path: str, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIcon(QIcon(icon_path))
        self.setIconSize(QSize(32, 32))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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

class ColorDot(QToolButton):
    def __init__(self, color: QColor, selected=False, parent=None):
        super().__init__(parent)
        self.color = color
        self.selected = selected
        self.setFixedSize(32, 32)
        self.setCheckable(True)
        self.setStyleSheet("border: none;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.selected or self.isChecked():
            pen = QPen(QColor("white"), 3)
            painter.setPen(pen)
        else:
            painter.setPen(Qt.PenStyle.NoPen)

        brush = QBrush(self.color)
        painter.setBrush(brush)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.drawEllipse(rect)

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
            ("   Appearance", "icons/eye.svg"),  # Use valid icon paths or leave empty
            ("   Test", "icons/info-circle.svg")
        ]

        self.buttons = []
        for index, (label, icon) in enumerate(button_info):
            btn = SidebarButton(label, icon, index)
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
        label = QLabel("Test Settings")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return tab

# === Entry Point ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SettingsWindow()
    window.show()
    sys.exit(app.exec())
