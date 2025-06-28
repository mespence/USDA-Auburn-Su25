from PyQt6.QtGui import QIconEngine, QPixmap, QPainter, QColor
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import Qt, QPoint, QRect


class ColoredSvgIconEngine(QIconEngine):
    """
    A custom QIconEngine that renders an SVG file and applies a color tint.

    This engine ignores any internal fill or stroke attributes in the SVG file
    and applies the specified color uniformly using QPainter composition.

    Parameters:
        svg_path (str): Path to the SVG file to be used as the icon source.
        color (str): Color to apply as a hex string (e.g., '#FFFFFF').
    """

    def __init__(self, svg_path: str, color: str):
        super().__init__()
        self.svg_path = svg_path
        self.color = QColor(color)

   
    def paint(self, painter, rect, mode, state):
        """
        Renders the SVG, applies a color tint to the non-transparent parts using
        CompositionMode_SourceIn, and preserves alpha transparency.
        """
        renderer = QSvgRenderer(self.svg_path)
        size = rect.size()

        # Step 1: Render the original SVG into a transparent pixmap
        base_pixmap = QPixmap(size)
        base_pixmap.fill(Qt.GlobalColor.transparent)
        base_painter = QPainter(base_pixmap)
        renderer.render(base_painter)
        base_painter.end()

        # Step 2: Prepare a new transparent pixmap to apply the tint
        color_pixmap = QPixmap(size)
        color_pixmap.fill(Qt.GlobalColor.transparent)

        # Step 3: Draw the base SVG into the tint pixmap and apply color
        tint_painter = QPainter(color_pixmap)
        tint_painter.drawPixmap(0, 0, base_pixmap)  # Copy alpha shape
        tint_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        tint_painter.fillRect(color_pixmap.rect(), self.color)
        tint_painter.end()

        # Step 4: Draw final result into target
        painter.drawPixmap(rect, color_pixmap)



    def clone(self):
        """
        Creates a copy of this icon engine.

        Returns:
            ColoredSvgIconEngine: A new instance with the same SVG path and color.
        """
        return ColoredSvgIconEngine(self.svg_path, self.color.name())
