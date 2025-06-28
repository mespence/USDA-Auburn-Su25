from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt6.QtCore import Qt, QSize, QRectF, QByteArray, QBuffer
from PyQt6.QtSvg import QSvgGenerator

class CustomRoundedRect:
    def __init__(self, rect: QRectF, tlRadius=0, trRadius=0, blRadius=0, brRadius=0):
        self.rect = rect
        self.radii = (tlRadius, trRadius, blRadius, brRadius)

    def build_path(self) -> QPainterPath:
        r = self.rect
        tl, tr, bl, br = self.radii
        path = QPainterPath()
        path.moveTo(r.x() + tl, r.y())
        path.lineTo(r.right() - tr, r.y())
        path.quadTo(r.right(), r.y(), r.right(), r.y() + tr)
        path.lineTo(r.right(), r.bottom() - br)
        path.quadTo(r.right(), r.bottom(), r.right() - br, r.bottom())
        path.lineTo(r.x() + bl, r.bottom())
        path.quadTo(r.x(), r.bottom(), r.x(), r.bottom() - bl)
        path.lineTo(r.x(), r.y() + tl)
        path.quadTo(r.x(), r.y(), r.x() + tl, r.y())
        path.closeSubpath()
        return path

    def draw(self, painter: QPainter):
        painter.drawPath(self.build_path())

def create_theme_preview(theme: dict, size=(150, 110)) -> QByteArray:
    width, height = size
    svg_data = QByteArray()
    buffer = QBuffer(svg_data)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)

    generator = QSvgGenerator()
    generator.setOutputDevice(buffer)
    generator.setSize(QSize(width, height))
    generator.setViewBox(QRectF(0, 0, width, height))
    generator.setResolution(96)

    painter = QPainter(generator)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    border_width = 2
    def adjusted_rect(x, y, w, h) -> QRectF:
        adjust = border_width / 2
        return QRectF(x + adjust, y + adjust, w - border_width, h - border_width)

    painter.setBrush(QColor(theme["BACKGROUND_COLOR_1"]))
    painter.setPen(QPen(QColor(theme["BORDER_COLOR"]), border_width))
    painter.drawRoundedRect(adjusted_rect(0, 0, width, height), 6, 6)

    painter.setBrush(QColor(theme["BACKGROUND_COLOR_2"]))
    body_rect = adjusted_rect(0, 15, width, height - 15)
    CustomRoundedRect(body_rect, blRadius=6, brRadius=6).draw(painter)

    outer_margin = 15
    painter.setBrush(QColor(theme["BACKGROUND_COLOR_1"]))
    popup_rect = adjusted_rect(outer_margin, 25, width - 2 * outer_margin, height - 55)
    CustomRoundedRect(popup_rect, 6, 6, 6, 6).draw(painter)

    painter.setPen(Qt.PenStyle.NoPen)
    button_spacing = 5

    painter.setBrush(QColor(theme["ACCENT_COLOR"]))
    left_button_rect = adjusted_rect(outer_margin, 85, (width - button_spacing) / 2 - outer_margin, 15)
    CustomRoundedRect(left_button_rect, 6, 6, 6, 6).draw(painter)

    painter.setBrush(QColor(theme["BUTTON_COLOR"]))
    right_button_rect = adjusted_rect(width / 2 + 5, 85, (width - button_spacing) / 2 - outer_margin, 15)
    CustomRoundedRect(right_button_rect, 6, 6, 6, 6).draw(painter)

    painter.setBrush(QColor(theme["FONT_COLOR_1"]))
    inner_margin = 25
    for i in range(3):
        top = int(35 + i * 12)
        width_line = (width - 2 * inner_margin) if i < 2 else (width - 2 * inner_margin) // 2
        painter.drawRoundedRect(inner_margin, top, width_line, 6, 4, 4)

    painter.end()
    buffer.close()
    return svg_data