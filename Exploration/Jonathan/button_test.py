from PyQt6.QtWidgets import QToolButton, QApplication, QWidget, QVBoxLayout
from PyQt6.QtGui import QPixmap, QPainter, QColor, QIcon
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import QSize, Qt
import sys

def load_svg_colored(svg_path, stroke_color="#FFFFFF", size=QSize(32, 32)):
    svg_renderer = QSvgRenderer(svg_path)
    pixmap = QPixmap(size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(stroke_color))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    svg_renderer.render(painter)
    painter.end()

    return QIcon(pixmap)

app = QApplication(sys.argv)
win = QWidget()
layout = QVBoxLayout()

btn = QToolButton()
icon = load_svg_colored("icons/sliders.svg", "#FFFFFF")  # ← force stroke color to white
btn.setIcon(icon)
btn.setIconSize(QSize(32, 32))
btn.setStyleSheet("background-color: #333333; border-radius: 4px;")

layout.addWidget(btn)
win.setLayout(layout)
win.setStyleSheet("background-color: #111111;")
win.show()
sys.exit(app.exec())
