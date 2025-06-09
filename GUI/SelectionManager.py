
from pyqtgraph import PlotWidget, InfiniteLine, mkPen, mkBrush

from PyQt6.QtGui import QColor, QMouseEvent, QKeyEvent
from PyQt6.QtCore import Qt, QPointF

from LabelArea import LabelArea
from Settings import Settings

# ctrl vs shift click
# werid multi select bugs still

class Selection:
    """
    Manages `DataWindow` selections and the actions performed on them.
    """
    def __init__(self, plot_widget: PlotWidget):
        

        self.selected_items: list = []  # InfiniteLine | LabelArea
        self.highlighted_item = None
        self.hovered_item = None
        self.dragged_line: InfiniteLine = None  # Which InfiniteLine (if any) is being dragged
        
        self.datawindow: PlotWidget = plot_widget  # the parent PlotWidget (i.e. the DataWindow)

        self.default_style = {
            'transition line': mkPen(color='#000000', width=2),
            'baseline': mkPen(color='#808080', width=2),
            'text color': '#000000'
        }

        self.highlighted_style = {
            'transition line' : mkPen(width=4, color='#0D6EFDC0'),
            'baseline' : mkPen(width=4, color='#0D6EFDC0'),
        }

        self.selected_style = {
            'transition line': mkPen(color='#0D6EFD', width=6),
            'baseline': mkPen(color='#0D6EFD', width=6),
            'area': mkBrush(color='#0D6EFD80'),  # label area LinearRegionItems
            'text background': mkBrush(color='#0D6EFD90'),  # label area text backgrounds
            'text color': "#FFFFFF"
        }

        self.moving_mode: bool = False

    def clear(self):
        self.selected_items.clear()
        self.dragged_line = None

    def _sort_key(self, item):
        """
        A sort key used to keep `selected_items` in chronological order.
        """
        if isinstance(item, LabelArea):
            return item.start_time
        if isinstance(item, InfiniteLine):
            return item.value()
        return float('inf')  # fallback


    def select(self, item):
        self.highlighted_item = None
        if isinstance(item, InfiniteLine):
            item.setPen(self.selected_style['transition line']) # same as highlighting currently
        if isinstance(item, LabelArea):
            labels = self.datawindow.labels
            idx = labels.index(item)
            left_line = labels[idx].transition_line
            if labels[idx] == labels[-1]: # trying to the end area
                right_line = left_line  # TODO: fix this to work with whatever we decide for last line
            else:
                right_line = labels[idx+1].transition_line
            left_line.setPen(self.selected_style['transition line'])
            right_line.setPen(self.selected_style['transition line'])
            item.area.setBrush(self.selected_style['area'])
            item.label_text.setColor(self.selected_style['text color'])
            item.duration_text.setColor(self.selected_style['text color'])
            item.label_background.setBrush(self.selected_style['text background'])
            item.duration_background.setBrush(self.selected_style['text background'])
        self.selected_items.append(item)
        self.selected_items.sort(key=self._sort_key)
        self.datawindow.viewbox.update()

    def deselect_item(self, item):
        if isinstance(item ,InfiniteLine):
            if item == self.datawindow.baseline:
                item.setPen(self.default_style['baseline'])
            else:
                item.setPen(self.default_style['transition line'])

        if isinstance(item, LabelArea):
            labels = self.datawindow.labels
            idx = labels.index(item)

            item.area.setBrush(mkBrush(color=Settings.label_to_color[item.label]))
            item.label_background.setBrush(mkBrush(item.get_background_color()))
            item.duration_background.setBrush(mkBrush(item.get_background_color()))

            item.label_text.setColor(self.default_style['text color'])
            item.duration_text.setColor(self.default_style['text color'])

            item.transition_line.setPen(self.default_style['transition line'])

            # deselect right transition line if next item is unselected
            if idx < len(labels) - 1 and not self.is_selected(labels[idx + 1]):
                labels[idx + 1].transition_line.setPen(self.default_style['transition line'])

        self.selected_items.remove(item)

    def deselect_all(self):
        for item in self.selected_items[:]:
            self.deselect_item(item)
        
    
    def is_selected(self, item) -> bool:
        """
        Returns `True` if `item` is part of the current selection.
        """
        if isinstance(item, LabelArea):
            return item in self.selected_items
        if isinstance(item, InfiniteLine):
            return item in self.get_selected_lines()
        return False

    def get_selected_lines(self) -> list[InfiniteLine]:
        """
        Returns all transition lines associated with current selection.
        """
        if not self.selected_items:
            return []
        
        lines = [
            item if isinstance(item, InfiniteLine) else getattr(item, 'transition_line', None)
            for item in self.selected_items
        ]

        if not any(isinstance(item, LabelArea) for item in self.selected_items):  # no label areas
            return lines

        last_label = [item for item in self.selected_items if isinstance(item, LabelArea)][-1]
        labels = self.datawindow.labels

        if last_label != labels[-1]: # not end area
            idx = labels.index(last_label)
            lines.append(labels[idx + 1].transition_line)
        
        return lines
    

    def key_press_event(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete:
            for item in self.selected_items[:]: # shallow copy
                if isinstance(item, LabelArea):
                    self.delete_label_area(item)

            # if all(isinstance(item, LabelArea) for item in self.selected_items):
            #     for label_area in self.selected_items:
            #         self.delete_label_area(label_area)
            #     self.selected_items = None

    def mouse_press_event(self, event: QMouseEvent) -> None:
        point = self.datawindow.window_to_viewbox(event.position())
        x, y = point.x(), point.y()

        (x_min, x_max), (y_min, y_max) = self.datawindow.viewbox.viewRange()
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            self.deselect_all()
            self.datawindow.scene().update()
            return
        
        # ----------- LEFT CLICK -----------
        if event.button() == Qt.MouseButton.LeftButton:
            # Left-clicked InfiniteLine 
            if isinstance(self.hovered_item, InfiniteLine):
                print('Clicked InfiniteLine')

                if not self.is_selected(self.hovered_item):
                    self.deselect_all()
                    self.select(self.hovered_item)  

                self.dragged_line = self.hovered_item
                self.moving_mode = True
                self.datawindow.setCursor(Qt.CursorShape.ClosedHandCursor)

            # Left-clicked LabelArea
            elif isinstance(self.hovered_item, LabelArea):
                print('Clicked LabelArea')
                # normal click
                if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                    self.deselect_all()
                    self.select(self.hovered_item)

                # shift-click
                elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.select(self.hovered_item)
                
    def mouse_move_event(self, event: QMouseEvent) -> None:
        point = self.datawindow.window_to_viewbox(event.position())
        x, y = point.x(), point.y()

        if self.datawindow.edit_mode_enabled and not self.moving_mode:
            self.hover(x,y) 
            self.datawindow.scene().update()
        elif self.moving_mode and self.dragged_line is not None:
            self.apply_drag(x,y)
        return
                
    def mouse_release_event(self, event: QMouseEvent) -> None:
        point = self.datawindow.window_to_viewbox(event.position())
        x, y = point.x(), point.y()

        if self.moving_mode:
            self.deselect_item(self.dragged_line)
            self.hover(x,y)
            #self.update_highlight(self.dragged_line)
        
            self.moving_mode = False
            self.dragged_line = None
            self.datawindow.setCursor(Qt.CursorShape.OpenHandCursor)
        return
    
    def apply_drag(self, x: float, y: float) -> None:
        """
        Applies drag movement to the currently selected line (transition or baseline).
        Called on mouse move while dragging.
        """
        line = self.dragged_line
        if line is None:
            return

        # Handle dragging the baseline
        if line == self.datawindow.baseline:
            self.datawindow.baseline.setPos(y)
            return
    
        # Handle dragging a transition line
        MIN_PIXEL_DISTANCE = 2 * self.datawindow.devicePixelRatioF() # pixels
        labels = self.datawindow.labels
        viewbox = self.datawindow.viewbox

        for i, label in enumerate(labels):
            if label.transition_line == line:
                if i == 0:
                    return  # can't drag the leftmost edge

                left = labels[i - 1]
                right = label

                right_end_time =right.start_time + right.duration

                min_x_scene = viewbox.mapViewToScene(QPointF(left.start_time, 0)).x() + MIN_PIXEL_DISTANCE
                max_x_scene = viewbox.mapViewToScene(QPointF(right_end_time,  0)).x() - MIN_PIXEL_DISTANCE
                min_x = viewbox.mapSceneToView(QPointF(min_x_scene, 0)).x()
                max_x = viewbox.mapSceneToView(QPointF(max_x_scene, 0)).x()

                x = max(min_x, min(x, max_x)) # clamp x to range (min_x, max_x)

                delta = x - right.start_time
                left.duration += delta
                right.start_time = x
                right.duration -= delta
                right.set_transition_line(x)

                left.update_label_area()
                right.update_label_area()

        return

            # # Case 2: dragging right edge of label[i]  (skips rightmost edge)
            # if i < len(labels) - 1 and labels[i + 1].transition_line == line:
            #     left = labels[i]
            #     right = labels[i + 1]

            #     min_x_scene = viewbox.mapViewToScene(QPointF(left.start_time,0)).x() + MIN_PIXEL_DISTANCE
            #     max_x_scene = viewbox.mapViewToScene(QPointF(right.start_time + right.duration,0)).x() - MIN_PIXEL_DISTANCE
            #     min_x = viewbox.mapSceneToView(QPointF(min_x_scene, 0)).x()
            #     max_x = viewbox.mapSceneToView(QPointF(max_x_scene, 0)).x()
            #     x = max(min_x, min(x, max_x))

            #     left.duration = x - left.start_time
            #     right.start_time = x
            #     right.duration -= x - right.start_time
            #     right.set_transition_line(x)

            #     left.update_label_area()
            #     right.update_label_area()
            #     return


    def delete_label_area(self, label_area: LabelArea) -> None:
        """
        Deletes the specified label area and expands the left label area, 
        instead expanding the right label area if the deleted label area 
        is the first label area.
        """ 
        self.deselect_item(label_area)
        labels = self.datawindow.labels
        current_idx = labels.index(label_area)
        before_idx = current_idx - 1
        after_idx = current_idx + 1

        if len(labels) > 1:
            if label_area == labels[0]:  # expand left
                expanded_label_area = labels[after_idx]
                new_start_time = label_area.start_time
                new_range = [new_start_time, expanded_label_area.start_time +  expanded_label_area.duration]

                expanded_label_area.start_time = new_start_time
                expanded_label_area.set_transition_line(new_start_time)

            else:  # expand right
                expanded_label_area = labels[before_idx]
                if labels[before_idx].label == labels[after_idx].label:
                    # if labels next to each other are the same, merge
                    new_range = [expanded_label_area.start_time, label_area.start_time + label_area.duration + labels[after_idx].duration]
                new_range = [expanded_label_area.start_time, label_area.start_time + label_area.duration]

            expanded_label_area.area.setRegion(new_range)
            new_dur = expanded_label_area.duration + label_area.duration 
            expanded_label_area.duration = new_dur
            expanded_label_area.duration_text.setText(str(round(new_dur, 2)))
            expanded_label_area.update_label_area()

        for item in label_area.getItems():
            self.datawindow.viewbox.removeItem(item)   

        # if current_idx != len(self.labels):
        #     self.datawindow.labels[current_idx + 1].transition_line.setPen(mkPen(color='black', width=2))

        del self.datawindow.labels[current_idx]

    def hover(self, x: float, y: float):
        """
        Handles the actions performed when the mouse is at 
        a given (x, y) in ViewBox coordinates.
        """
        TRANSITION_HIGHLIGHT_THRESHOLD = 3 * self.datawindow.devicePixelRatioF() # pixels
        BASELINE_HIGHLIGHT_THRESHOLD = 3 * self.datawindow.devicePixelRatioF()

        if not self.datawindow.labels:
            return  # nothing to highlight
        
        (x_min, x_max), (y_min, y_max) = self.datawindow.viewbox.viewRange()

        if not (x_min <= x <= x_max and y_min <= y <= y_max):  # cursor outside viewbox
            if self.highlighted_item is not None:
                self.unhighlight_item(self.highlighted_item)
            return

        pixel_ratio = self.datawindow.devicePixelRatioF()

        transition_line, transition_distance = self.datawindow.get_closest_transition(x)
        baseline, baseline_distance = self.datawindow.get_baseline_distance(y)
        label_area = self.datawindow.get_closest_label_area(x)

        if transition_distance <= TRANSITION_HIGHLIGHT_THRESHOLD * pixel_ratio:
            if transition_line == self.datawindow.labels[0].transition_line:
                return  # can't drag leftmost transition
            self.hovered_item = transition_line
            self.update_highlight(transition_line, cursor = Qt.CursorShape.OpenHandCursor)
        elif baseline_distance <= BASELINE_HIGHLIGHT_THRESHOLD * pixel_ratio:
            self.hovered_item = baseline
            self.update_highlight(baseline, cursor = Qt.CursorShape.OpenHandCursor)
        else:
            self.hovered_item = label_area
            self.update_highlight(label_area, cursor = None)


    def update_highlight(self, item, cursor = None):
        if self.is_selected(item):
            return  # don't highlight already selected items
        
        # check if highlighted item needs to change:
        if self.highlighted_item != item:
            if self.highlighted_item is not None:
                self.unhighlight_item(self.highlighted_item) # unhighlight previous item
    
            if isinstance(item ,InfiniteLine):
                if item == self.datawindow.baseline:
                    item.setPen(self.highlighted_style['baseline'])
                else:
                    item.setPen(self.highlighted_style['transition line'])

            elif isinstance(item, LabelArea):
                label_color = self.datawindow.composite_on_white(Settings.label_to_color[item.label])
                h, s, l, a = label_color.getHslF()
                selected_color = QColor.fromHslF(h, min(s * 8, 1), l * 0.9, a)  
                selected_color.setAlpha(200)
                item.area.setBrush(mkBrush(color=selected_color))
            
            if cursor:
                self.datawindow.setCursor(cursor)
            else:
                self.datawindow.setCursor(Qt.CursorShape.ArrowCursor)  # default

            self.highlighted_item = item

    def unhighlight_item(self, item):
        if isinstance(item ,InfiniteLine):
            if item == self.datawindow.baseline:
                item.setPen(self.default_style['baseline'])
            else:
                item.setPen(self.default_style['transition line'])
        if isinstance(item, LabelArea):
            item.area.setBrush(mkBrush(color=Settings.label_to_color[item.label]))

        self.highlighted_item = None