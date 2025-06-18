from pyqtgraph import ViewBox

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QWheelEvent

class PanZoomViewBox(ViewBox):
    """
    Custom ViewBox that overrides default mouse/scroll behavior to support
    pan and zoom using wheel + modifiers.

    Pan/Zoom behavior:
    - Ctrl + Scroll: horizontal/vertical zoom (with Shift)
    - Scroll only: pan (horizontal or vertical based on Shift)
    """

    def __init__(self, datawindow = None) -> None:
        super().__init__()
        self.datawindow = datawindow
        self.zoom_viewbox_limit: float = 0.8

    def wheelEvent(self, event: QWheelEvent, axis=None) -> None:
        """
        Handles wheel input for zooming and panning, based on modifier keys.

        - Ctrl: zoom
        - Shift: vertical zoom or pan
        - No modifiers: horizontal pan
        """
        
        delta = event.angleDelta().y()
        modifiers = event.modifiers()
        live = self.datawindow.live_mode

        ctrl_held = modifiers & Qt.KeyboardModifier.ControlModifier
        shift_held = modifiers & Qt.KeyboardModifier.ShiftModifier

        if ctrl_held:
            zoom_factor = 1.001**delta
            center = self.mapToView(event.position())
    
            if shift_held: 
                # y zoom
                self.scaleBy((1, 1 / zoom_factor), center)
            else:
                # x zoom
                if live:
                    (x_min, x_max), _ = self.viewRange()
                    current_span = x_max - x_min
                    new_span = current_span / zoom_factor
                    self.datawindow.auto_scroll_window = new_span
                else:
                    (x_min, x_max), _ = self.viewRange()
                    width = x_max - x_min
                    new_width = width / zoom_factor

                    center_x = center.x()
                    new_x_min = center_x - (center_x - x_min) / zoom_factor
                    zero_ratio = - new_x_min / new_width

                    if zero_ratio > self.zoom_viewbox_limit:
                        new_x_min = 0 - self.zoom_viewbox_limit * new_width
                        new_x_max = new_x_min + new_width
                        self.setXRange(new_x_min, new_x_max, padding=0)
                    else:
                        self.scaleBy((1 / zoom_factor, 1), center)

        else:
            (x_min, x_max), (y_min, y_max) = self.viewRange()
            width, height = x_max - x_min, y_max - y_min

            if shift_held:
                # y zoom
                v_zoom_factor = 5e-4
                dy = delta * v_zoom_factor * height
                self.translateBy(y=dy)
            else:
                # x zoom
                if not live:
                    h_zoom_factor = 2e-4
                    dx = delta * h_zoom_factor * width

                    new_x_min = x_min + dx
                    zero_ratio = - new_x_min / width

                    # don't pan if it moves x=0 more than 80% across the ViewBox
                    if zero_ratio > self.zoom_viewbox_limit:
                        new_x_min = 0 - self.zoom_viewbox_limit * width
                        new_x_max = new_x_min + width
                        self.setXRange(new_x_min, new_x_max, padding=0)
                        # pass  
                    else:
                        self.translateBy(x=dx)

        event.accept()
        self.datawindow.update_plot()

    def mouseDragEvent(self, event, axis=None) -> None:
        event.ignore()
