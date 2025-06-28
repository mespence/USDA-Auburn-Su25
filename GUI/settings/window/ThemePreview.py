from PyQt6.QtWidgets import QWidget, QVBoxLayout, QRadioButton
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtCore import Qt, pyqtSignal

class HoverOverlay(QWidget):
    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self.color = color
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setVisible(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.color, 3)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 6, 6)

class UIThemeOption(QWidget):
    selected = pyqtSignal(str)

    def __init__(self, theme: dict, svg_data, parent=None):
        super().__init__(parent)
        self.theme_name = theme["NAME"]
        self.accent_color = QColor(theme["ACCENT_COLOR"])
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)

        self.preview = QSvgWidget()
        self.preview.load(svg_data)
        self.preview.setFixedSize(150, 110)
        self.preview.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.preview.setMouseTracking(True)
        self.preview.enterEvent = self._on_preview_enter
        self.preview.leaveEvent = self._on_preview_leave
        self.preview.mousePressEvent = self._on_click

        self.overlay = HoverOverlay(self.accent_color, self.preview)
        self.overlay.resize(self.preview.size())

        self.radio = QRadioButton(self.theme_name.capitalize())
        self.radio.setStyleSheet("""
            QRadioButton {
                font-family: 'Inter';
                font-size: 10pt;
                font-weight: 500;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.addWidget(self.preview, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.radio, alignment=Qt.AlignmentFlag.AlignLeft)
        self.setLayout(layout)

    def _on_click(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.radio.setChecked(True)
            self.selected.emit(self.theme_name)

    def _on_preview_enter(self, event):
        self.overlay.setVisible(True)
        self.overlay.raise_()

    def _on_preview_leave(self, event):
        self.overlay.setVisible(False)
