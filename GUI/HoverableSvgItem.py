from PyQt6.QtWidgets import QGraphicsColorizeEffect, QGraphicsTextItem, QGraphicsRectItem
from PyQt6.QtSvgWidgets import QGraphicsSvgItem
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtCore import Qt, QPointF

class HoverableSvgItem(QGraphicsSvgItem):
    def __init__(self, marker = None):
        super().__init__(marker.icon_path)
        self.marker = marker
        self.setAcceptHoverEvents(True)

        # highlight on hover
        self.hl_effect = QGraphicsColorizeEffect()
        self.hl_effect.setColor(Qt.GlobalColor.blue)
        self.hl_effect.setStrength(0) # off by default
        self.setGraphicsEffect(self.hl_effect)

        self.preview = None

    def hoverEnterEvent(self, event):
        if not self.preview:
            self.create_preview()
        else:
            self.preview.setVisible(True)

        self.hl_effect.setStrength(0.8)
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        if self.preview:
            self.preview.setVisible(False)
        
        self.hl_effect.setStrength(0)        
        super().hoverLeaveEvent(event)
    
    def create_preview(self):
        padding = 5
        max_width = 200
        scene = self.scene()

        text_item = QGraphicsTextItem(self.marker.text)
        text_item.setTextWidth(max_width)

        text_bounds = text_item.boundingRect()
        box_width = text_bounds.width() + 2 * padding
        box_height = text_bounds.height() + 2 * padding

        rect = QGraphicsRectItem(0, 0, box_width, box_height)
        rect.setBrush(QColor("#ffffe1"))
        rect.setPen(QPen(QColor("#000000"), 1))
        rect.setZValue(20)

        text_item.setParentItem(rect)
        text_item.setPos(padding, padding)

        icon_bounds = self.mapRectToScene(self.boundingRect())
        spacing = 5

        right_pos = QPointF(icon_bounds.right() + spacing, icon_bounds.top() - text_bounds.height() - spacing)
        left_pos = QPointF(icon_bounds.left() - box_width - spacing, icon_bounds.top() - text_bounds.height() - spacing)
        viewbox_bounds = self.marker.viewbox.sceneBoundingRect()

        if right_pos.x() + box_width < viewbox_bounds.right():
            rect.setPos(right_pos)
        else:
            rect.setPos(left_pos)

        scene.addItem(rect)
        self.preview = rect

        
