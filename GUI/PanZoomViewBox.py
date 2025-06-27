from pyqtgraph import ViewBox

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QWheelEvent, QAction, QKeyEvent
from PyQt6.QtWidgets import QMenu

from pyqtgraph import InfiniteLine

from Settings import Settings


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
        self.datawindow = None
        self.zoom_viewbox_limit: float = 0.8

    def wheelEvent(self, event: QWheelEvent, axis=None) -> None:
        """
        Handles wheel input for zooming and panning, based on modifier keys.

        - Ctrl: zoom
        - Shift: vertical zoom or pan
        - No modifiers: horizontal pan
        """

        if self.datawindow is None:
            self.datawindow = self.parentItem().getViewWidget()
        
        delta = event.angleDelta().y()
        modifiers = event.modifiers()
        live = getattr(self.datawindow, "live_mode", False)

        ctrl_held = modifiers & Qt.KeyboardModifier.ControlModifier
        shift_held = modifiers & Qt.KeyboardModifier.ShiftModifier

        if ctrl_held:
            # zoom
            zoom_factor = 1.001**delta
            center = self.mapToView(event.position())
    
            if shift_held: 
                # y zoom
                self.scaleBy((1, 1 / zoom_factor), center)
            else:
                # x zoom
                self.x_zoom(live, zoom_factor, center)
        else:
            (x_min, x_max), (y_min, y_max) = self.viewRange()
            width, height = x_max - x_min, y_max - y_min

            if shift_held:
                # y pan
                v_zoom_factor = 5e-4
                dy = delta * v_zoom_factor * height
                self.translateBy(y=dy)
            else:
                # x pan
                print("x")
                if not live:
                    print("not live")
                    h_zoom_factor = 2e-4
                    dx = delta * h_zoom_factor * width

                    new_x_min = x_min + dx
                    zero_ratio = - new_x_min / width

                    # don't pan if it moves x=0 more than 80% across the ViewBox
                    if zero_ratio > self.zoom_viewbox_limit:
                        self.snap_within_viewbox(new_x_min, width) 
                    else:
                        self.translateBy(x=dx)

        self.datawindow.update_plot()
        event.accept()

    def snap_within_viewbox(self, x_min, width):
        new_x_min = 0 - self.zoom_viewbox_limit * width
        new_x_max = x_min + width
        self.setXRange(new_x_min, new_x_max, padding=0)

    def x_zoom(self, live, zoom_factor, center):
        (x_min, x_max), _ = self.viewRange()
        current_span = x_max - x_min
        if live:
            print("live")
            new_span = current_span / zoom_factor
            self.datawindow.auto_scroll_window = new_span
            print(new_span)
        else:
            print("not live")
            center_x = center.x()

            # ensure 0 stays within 80% of viewbox limit
            new_width = current_span / zoom_factor
            new_x_min = center_x - (center_x - x_min) / zoom_factor
            zero_ratio = - new_x_min / new_width

            if zero_ratio > self.zoom_viewbox_limit:
                self.snap_within_viewbox(new_x_min, new_width)
            else:
                self.scaleBy((1 / zoom_factor, 1), center)
        pass

    def keyPressEvent(self, event: QKeyEvent) -> None:
        live = getattr(self.datawindow, "live_mode", False)
        viewbox_rect = self.viewRect()
        center = viewbox_rect.center()
        zoom_factor_in = 1.1
        zoom_factor_out = 0.9

        if event.key() == Qt.Key.Key_Up:
            # y zoom in
            self.scaleBy((1, 1 / zoom_factor_in), center)
        elif event.key() == Qt.Key.Key_Down:
            # y zoom out
            self.scaleBy((1, 1 / zoom_factor_out), center)
        elif event.key() == Qt.Key.Key_Left:
            # x zoom out
            self.x_zoom(live, zoom_factor_out, center)
        elif event.key() == Qt.Key.Key_Right:
            # x zoom in
            self.x_zoom(live, zoom_factor_in, center)

        self.datawindow.update_plot()

    def mouseDragEvent(self, event, axis=None) -> None:
        """
        Disables default drag-to-pan behavior (drag is used for selections).
        """
        event.ignore()


    def contextMenuEvent(self, event):
        """
        Displays a context menu for right-clicking on LabelAreas.

        Currently adds a label type submenu to change the label's classification.
        """
        if self.datawindow is None:
            self.datawindow = self.parentItem().getViewWidget()

        live = getattr(self.datawindow, "live_mode", False)

        if live:
            event.ignore()
            return

        item = self.datawindow.selection.hovered_item
        if isinstance(item, InfiniteLine):
            print('Right-clicked InfiniteLine')
            return  # TODO: infinite line context menu not yet implemented

        scene_pos = event.scenePos()
        data_pos = self.mapSceneToView(scene_pos)
        x = data_pos.x()  
        
        menu = QMenu()
        label_type_dropdown = QMenu("Change Label Type", menu)

        label_names = list(Settings.label_to_color.keys())
        label_names.remove("END AREA")
        for label in label_names:            
            action = QAction(label, menu)
            action.setCheckable(True)

            if item.label == label:
                action.setChecked(True)
                
            action.triggered.connect(
                lambda checked, label_area=item, label=label:
                self.datawindow.selection.change_label_type(label_area, label)
            )
        
            label_type_dropdown.addAction(action)

        add_comment = QAction("Add Comment", menu)

        #action3 = QAction("Custom Option 3")
        menu.addMenu(label_type_dropdown)
        menu.addAction(add_comment)
        #menu.addAction(action3)

        selected_action = menu.exec(event.screenPos())           
        if selected_action == label_type_dropdown:
            print("Option 1 selected")
        elif selected_action == add_comment:
            self.datawindow.add_comment_at_click(x)
            print("Option 2 selected")
        else:
            pass
