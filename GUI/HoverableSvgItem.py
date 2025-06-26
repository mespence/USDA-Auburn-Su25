from PyQt6.QtWidgets import QGraphicsColorizeEffect, QGraphicsTextItem, QGraphicsRectItem
from PyQt6.QtSvgWidgets import QGraphicsSvgItem
from PyQt6.QtGui import QColor, QPen, QFont
from PyQt6.QtCore import Qt, QPointF

def truncate(text: str, limit: int) -> str:
    """
    Truncate a string to the maximum number of characters,
    adds '...' if truncated.

    Parameters:
        text (str): input string to truncate
        limit (int): maximum length including ellipsis

    Returns:
        str: truncated string ending with '...' if exceeding limit
    """
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."

class HoverableSvgItem(QGraphicsSvgItem):
    """
    A custom QGraphicsSvgItem that represents a comment icon with 
    interactive hover behavior.

    Features:
    - Highlights the icon when hovered using a blue glow effect.
    - Displays a text preview box near the icon on hover.
    - Automatically truncates long comment text in the preview with ellipsis.
    - Disables hover preview when the view is being panned or scrolled 
      to prevent stale previews.
    """
    def __init__(self, marker = None):
        """
        Initialize the hoverable SVG icon.

        Parameters:
            marker (CommentMarker): The comment marker associated with this icon.
        """
        super().__init__(marker.icon_path)
        self.marker = marker
        self.setAcceptHoverEvents(True)

        # highlight on hover
        self.hl_effect = QGraphicsColorizeEffect()
        self.hl_effect.setColor(Qt.GlobalColor.blue)
        self.hl_effect.setStrength(0)
        self.setGraphicsEffect(self.hl_effect)

        self.preview = None
        self.text = None
        self.text_limit = 150
        self.offset = QPointF()

        self.marker.viewbox.sigTransformChanged.connect(self.viewbox_change)

    def hoverEnterEvent(self, event) -> None:
        """
        Handles the hover enter event: shows and positions the preview,
        and highlights the icon.
        """
        if not self.preview:
            self.create_preview()

        self.update_offset()
        self.preview.setPos(self.offset)
        self.preview.setVisible(True)

        self.hl_effect.setStrength(0.8)
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event) -> None:
        """
        Handles the hover leave event: hides the preview and 
        removes highlight.
        """
        if self.preview:
            self.preview.setVisible(False)
        
        self.hl_effect.setStrength(0)        
        super().hoverLeaveEvent(event)
    
    def create_preview(self) -> None:
        """
        Creates a reusable preview box that displays a truncated version
        of the comment text.

        The preview is only created once and reused across hovers to avoid
        performance issues or crashes caused by rapid creation and 
        deletion of QGraphics items.
        """
        padding = 5
        max_width = 200
        scene = self.scene()

        text_item = QGraphicsTextItem(truncate(
                                        self.marker.text, 
                                        self.text_limit))
        text_item.setTextWidth(max_width)
        text_item.setDefaultTextColor(Qt.GlobalColor.black)
        text_item.setFont(QFont("Arial", 12))

        text_bounds = text_item.boundingRect()
        box_width = text_bounds.width() + 2 * padding
        box_height = text_bounds.height() + 2 * padding

        rect = QGraphicsRectItem(0, 0, box_width, box_height)
        rect.setBrush(QColor("#ffffe1"))
        rect.setPen(QPen(QColor("#000000"), 1))
        rect.setZValue(20)

        text_item.setParentItem(rect)
        text_item.setPos(padding, padding)

        self.preview = rect
        self.text = text_item
        scene.addItem(self.preview)
        self.preview.setVisible(False)

    def update_offset(self) -> None:
        """
        Calculates and updates the position offset for the preview box 
        based on current view bounds and available space.
        """
        spacing = 5
        preview_width = self.preview.boundingRect().width()
        preview_height = self.preview.boundingRect().height()

        icon_bounds = self.mapRectToScene(self.boundingRect())
        viewbox_bounds = self.marker.viewbox.sceneBoundingRect()

        right_pos = QPointF(icon_bounds.right() + spacing, 
                            icon_bounds.top() - preview_height - spacing)
        left_pos = QPointF(icon_bounds.left() - preview_width - spacing, 
                           icon_bounds.top() - preview_height - spacing)

        if right_pos.x() + preview_width < viewbox_bounds.right():
            self.offset = right_pos
        else:
            self.offset = left_pos

    def refresh_text(self, new_text: str) -> None:
        """
        Updates the text inside the preview box with a new 
        comment (truncated).

        Parameters:
            new_text (str): The updated comment text.
        """
        if self.text:
            self.text.setPlainText(truncate(new_text, self.text_limit))
            # adjust rect width/height if text length changes
            padding = 5
            text_bounds  = self.text.boundingRect()
            self.preview.setRect(0, 0,
                                  text_bounds.width()  + 2*padding,
                                  text_bounds.height() + 2*padding)
            
    def viewbox_change(self) -> None:
        """
        Called when the viewbox transform changes (e.g. pan or zoom).
        Hides the preview and resets highlight to avoid stale overlays.
        """
        if self.preview:
            self.preview.setVisible(False)
        self.hl_effect.setStrength(0)

    def remove(self) -> None:
        """
        Removes the preview box from the scene and disconnects the viewbox
        transform signal to prevent further updates.
        """
        if self.preview:
            # if icon never hovered, then no need to delete anything
            scene = self.scene()
            scene.removeItem(self.preview)
        
        try:
            self.marker.viewbox.sigTransformChanged.disconnect(self.viewbox_change)
        except Exception as e:
            print(f"[ERROR HSVGI] disconnecting viewbox change signal: {e}")