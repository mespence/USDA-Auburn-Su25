from pyqtgraph import (
    PlotWidget, TextItem, InfiniteLine, mkPen
)
from PyQt6.QtWidgets import QGraphicsRectItem
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QFont, QColor

class CommentMarker():
    """
    A class for rendering user-placed comments with a
    vertical dashed line and rotated text item
    """
    def __init__(self, time: float, text: str, plot_widget: PlotWidget):
        self.time = time
        self.text = text
        self.plot_widget = plot_widget
        self.viewbox = plot_widget.getPlotItem().getViewBox()

        self.marker = InfiniteLine(
            post = self.time,
            angle = 90,
            pen = mkPen(self.color, style = Qt.PenStyle.DashLine),
            movable = False,
        )

        self.viewbox.addItem(self.marker)

        self.text_item = TextItem(text, color='black', anchor=(0,0))
        self.text_item.setRotation(270)

        y_min, y_max = self.viewbox.viewRange()[1]
        comment_y = y_min + 0.1 * (y_max - y_min)
        self.text_item.setPos(self.time, comment_y)

        self.viewbox.addItem(self.text_item)

        self.viewbox.sigTransformChanged.connect(self.update_position) 

    def update_position(self):
        _, (y_min, y_max) = self.viewbox.viewRange()
        comment_y = y_min + 0.1 * (y_max - y_min)
        self.text_item.setPos(self.time, comment_y)

    def set_text(self, new_text: str):
        self.text = new_text
        self.text_item.setText(new_text)

    def set_visible(self, visible: bool):
        self.marker.setVisible(visible)
        self.text_item.setVisible(visible)

    def remove(self):
        self.viewbox.removeItem(self.marker)
        self.viewbox.removeItem(self.text_item)

