import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QFont, QFontMetricsF, QPainter, QBrush, QColor

import pyqtgraph as pg
from pyqtgraph import GraphicsObject


class TextWithBackground(GraphicsObject):
    def __init__(self, text, pos: QPointF, font=QFont("Sans", 12),
                 color=QColor("black"), bg_color=QColor(255, 255, 0, 160),
                 anchor=(0.5, 0.5)):
        super().__init__()
        self.text = text
        self.font = font
        self.text_color = color
        self.bg_color = bg_color
        self.anchor = QPointF(*anchor)
        self.setZValue(1000)
        self.setPos(pos)

    def boundingRect(self):
        metrics = QFontMetricsF(self.font)
        rect = metrics.tightBoundingRect(self.text)
        rect.adjust(-2, -2, 2, 2)  # Add a margin
        offset = QPointF(rect.width() * self.anchor.x(),
                         rect.height() * self.anchor.y())
        rect.moveTopLeft(-offset)
        return rect

    def paint(self, painter: QPainter, *args):
        painter.setFont(self.font)

        rect = self.boundingRect()

        # Background
        painter.setBrush(QBrush(self.bg_color))
        painter.setPen(QColor("black"))
        painter.drawRect(rect)

        # Text
        painter.setPen(self.text_color)
        metrics = QFontMetricsF(self.font)
        text_pos = QPointF(-metrics.horizontalAdvance(self.text) * self.anchor.x(),
                   metrics.ascent() - metrics.height() * self.anchor.y())
        painter.drawText(text_pos, self.text)


def main():
    app = QApplication([])

    # Create PlotWidget
    plot = pg.PlotWidget()
    plot.setWindowTitle("TextWithBackground Example")
    plot.setGeometry(100, 100, 800, 600)
    plot.setBackground("white")

    # Dummy sine wave
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    plot.plot(x, y, pen=pg.mkPen("blue"))

    # Add custom label
    label = TextWithBackground("Label", pos=QPointF(5, 0.5))
    plot.getViewBox().addItem(label)

    plot.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
