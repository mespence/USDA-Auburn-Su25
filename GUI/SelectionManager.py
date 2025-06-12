from pyqtgraph import PlotWidget, InfiniteLine, mkPen, mkBrush

from PyQt6.QtGui import QColor, QMouseEvent, QKeyEvent
from PyQt6.QtCore import Qt, QPointF

from LabelArea import LabelArea
from Settings import Settings


# werid multi select bugs still, multi delete

class Selection:
    """
    Manages `DataWindow` selections and the actions performed on them.

    More than one LabelArea can be selected at a time, but only a single 
    InfiniteLine can be directly selected.
    """
    def __init__(self, plot_widget: PlotWidget):
        

        self.selected_items: list = []  # InfiniteLine | LabelArea, sorted chronologically
        self.selection_parent: LabelArea = None  # which item is the "parent" of a multi-selection
        self.highlighted_item = None  # the item visually highlighted
        self.hovered_item = None  # the item the mouse is currently over
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
            'baseline': mkPen(color="#031A3B", width=6),
            'area': mkBrush(color='#0D6EFD80'),  # label area LinearRegionItems
            'text background': mkBrush(color='#0D6EFD90'),  # label area text backgrounds
            'text color': "#FFFFFF"
        }

        self.moving_mode: bool = False

        self.last_cursor_pos: tuple[float, float] = None  # the last (x,y) of the cursor in viewbox coords

    def _sort_key(self, item):
        """
        A sort key used to keep `selected_items` in chronological order.
        """
        if isinstance(item, LabelArea):
            return item.start_time
        if isinstance(item, InfiniteLine):
            return item.value()
        return float('inf')  # fallback
    

    def select(self, item) -> None:
        """
        Handles styling a selected item and adds it to the selection.

        Input:
            `item`: an object to be selected (e.g. InfiniteLine or LabelArea)
        """
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

            if left_line not in self.selected_items:
                self.selected_items.append(left_line)
            if right_line not in self.selected_items:
                self.selected_items.append(right_line)

        self.selected_items.append(item)
        self.selected_items.sort(key=self._sort_key)
        self.datawindow.viewbox.update()

    
    def multi_select(self, item: LabelArea) -> None:
        """
        Selects all LabelAreas between the selection_parent and the given item.
        """
        if self.selection_parent == None:  # fallback
            self.deselect_all()
            self.selection_parent = item
            self.select(item)

        labels = self.datawindow.labels[:]
        parent = self.selection_parent

        try:
            idx1 = labels.index(parent)
            idx2 = labels.index(item)
        except ValueError:
            return # one of the items isn't in th label list

        start_idx, end_idx = sorted((idx1, idx2))
        self.deselect_all()
        for i in range(start_idx, end_idx + 1):
            if not labels[i].is_end_area:
                self.select(labels[i])
        self.selection_parent = parent

    def deselect_item(self, item):
        """
        Resets styling an item to pre-selection state and remove it from the selection.

        Input:
            `item`: an object to be deselected (e.g. InfiniteLine or LabelArea)
        """
        if isinstance(item ,InfiniteLine) and item != self.highlighted_item:
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
        """
        Deselects all items in the selection.
        """
        for item in self.selected_items[:]:
            self.deselect_item(item)
        self.selection_parent = None
        
    
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
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            label_areas_to_delete = [item for item in self.selected_items if isinstance(item, LabelArea)][::-1]
            self.deselect_all()
            for label_area in label_areas_to_delete:
                print(f"Deleting: {label_area.label} at {label_area.start_time:.2f}s" )
                if label_area == label_areas_to_delete[0] or label_area == label_areas_to_delete[-1]:                    
                    self.delete_label_area(label_area, merge_adjacent=True) # merge on first and last only
                else:
                    self.delete_label_area(label_area, merge_adjacent=False)

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

            #COMMENT TESITNG

            if self.datawindow.comment_editing:
                self.datawindow.comment_editing = False
                return

            # END

            # Left-clicked no item
            if self.hovered_item == None:
                self.deselect_all()

            # Left-clicked InfiniteLine 
            elif isinstance(self.hovered_item, InfiniteLine):
                if not self.is_selected(self.hovered_item):
                    self.deselect_all()
                    self.select(self.hovered_item)  

                self.dragged_line = self.hovered_item
                self.moving_mode = True
                self.datawindow.setCursor(Qt.CursorShape.ClosedHandCursor)

            # Left-clicked LabelArea
            elif isinstance(self.hovered_item, LabelArea):
                if self.hovered_item.is_end_area:
                    return  # can't select end area

                # ---- Normal click -------
                if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                    self.deselect_all()
                    self.select(self.hovered_item)
                    self.selection_parent = self.hovered_item

                # ---- Shift-click --------
                elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    if all(isinstance(item, InfiniteLine) for item in self.selected_items):
                        self.deselect_all()  # only label areas can have multi selection
                    if not self.selected_items:
                        self.select(self.hovered_item)
                        self.selection_parent = self.hovered_item
                    else:
                        self.multi_select(self.hovered_item)                        

                # ---- Ctrl-click ---------
                elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    if all(isinstance(item, InfiniteLine) for item in self.selected_items):
                        self.deselect_all()  # only label areas can have multi selection
                    if not self.is_selected(self.hovered_item):
                        self.select(self.hovered_item)
                        self.selection_parent = self.hovered_item
                    else:
                        self.deselect_item(self.hovered_item)
                
    def mouse_move_event(self, event: QMouseEvent) -> None:
        point = self.datawindow.window_to_viewbox(event.position())
        x, y = point.x(), point.y()

        if self.datawindow.edit_mode_enabled and not self.moving_mode:
            self.hover(x, y) 
            self.datawindow.scene().update()
        elif self.moving_mode and self.dragged_line is not None:
            self.apply_drag(x, y)
        return
                
    def mouse_release_event(self, event: QMouseEvent) -> None:
        point = self.datawindow.window_to_viewbox(event.position())
        x, y = point.x(), point.y()

        if self.moving_mode:
            if self.dragged_line == self.datawindow.baseline:
                # Place baseline
                self.dragged_line.setPen(self.highlighted_style['baseline'])
                self.highlighted_item = self.dragged_line
                self.deselect_item(self.dragged_line)
            else:
                # Place transition line, resetting to previous style
                labels = self.datawindow.labels

                # get the index of the label area the dragged line belongs to
                idx = next(
                    i for i, label_area in enumerate(labels) 
                    if label_area.transition_line == self.dragged_line
                )

                if idx is not None:
                    left_selected = self.is_selected(labels[idx-1]) if idx > 0 else False
                    right_selected = self.is_selected(labels[idx]) if idx < len(labels) else False

                    if left_selected or right_selected:
                        pass  # already selected: don't update pen
                    elif self.dragged_line == self.hovered_item: # standalone selection
                        self.dragged_line.setPen(self.highlighted_style['transition line'])
                        self.highlighted_item = self.dragged_line
                        self.deselect_item(self.dragged_line)
                
            self.moving_mode = False
            self.dragged_line = None
            self.datawindow.setCursor(Qt.CursorShape.OpenHandCursor)

            self.hover(x,y)
        return     
        
    def apply_drag(self, x: float, y: float) -> None:
        """
        Applies drag movement to the currently selected line (transition or baseline).
        Called on mouse move while dragging.
        """
        line = self.dragged_line
        if line is None:
            return

        # Dragging the baseline
        if line == self.datawindow.baseline:
            self.datawindow.baseline.setPos(y)
            return
    
        # Dragging a transition line
        MIN_PIXEL_DISTANCE = 2 * self.datawindow.devicePixelRatioF()
        labels = self.datawindow.labels
        viewbox = self.datawindow.viewbox

        for i, label in enumerate(labels):
            if label.transition_line == line:
                if i == 0:
                    return  # can't drag the leftmost edge

                left = labels[i - 1]  # label area to the left of the line
                right = label  # label area to the right of the line

                # minimum of the clamp
                min_x_scene = viewbox.mapViewToScene(QPointF(left.start_time, 0)).x() + MIN_PIXEL_DISTANCE
                min_x = viewbox.mapSceneToView(QPointF(min_x_scene, 0)).x()     
                   
                if right.is_end_area: # no right clamp if next label is end area
                    x = max(x, min_x)
                else:  # clamp to right
                    right_end_time =right.start_time + right.duration
                    max_x_scene = viewbox.mapViewToScene(QPointF(right_end_time, 0)).x() - MIN_PIXEL_DISTANCE
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



    def delete_label_area(self, label_area: LabelArea, merge_adjacent: bool = True) -> None:
        """
        Deletes the specified label area and expands the left label area, 
        instead expanding the right label area if the deleted label area 
        is the first label area. Handles merging same-type labels after a deletion.
        """ 
        labels = self.datawindow.labels

     
        current_idx = labels.index(label_area)

        
        before_idx = current_idx - 1
        after_idx = current_idx + 1

        if len(labels) > 1:
            if label_area == labels[0] and after_idx < len(labels):  # expand left
                expanded_label_area = labels[after_idx] 
                label_area.label = expanded_label_area.label
                self.merge_adjacent_labels(label_area)



                # new_start_time = label_area.start_time


                # new_range = [new_start_time, expanded_label_area.start_time +  expanded_label_area.duration]
                # new_dur = expanded_label_area.duration + label_area.duration
                # expanded_label_area.start_time = new_start_time
                # expanded_label_area.set_transition_line(new_start_time)

            else:  # expand right
                expanded_label_area = labels[before_idx]
                label_area.label = expanded_label_area.label
                self.merge_adjacent_labels(label_area)


            #     if after_idx < len(labels):
            #         after_label = labels[after_idx]
            #         merging_adjacent = (expanded_label_area.label == after_label.label)

            #         if merging_adjacent and merge_adjacent:
            #             new_range = [expanded_label_area.start_time, after_label.start_time + after_label.duration]
            #             new_dur = expanded_label_area.duration + label_area.duration + after_label.duration

            #             for item in after_label.getItems():
            #                 self.datawindow.viewbox.removeItem(item)
            #             del self.datawindow.labels[after_idx]
            #         else:
            #             new_range = [expanded_label_area.start_time, label_area.start_time + label_area.duration]
            #             new_dur = expanded_label_area.duration + label_area.duration

            # expanded_label_area.area.setRegion(new_range) 
            # expanded_label_area.duration = new_dur
            # expanded_label_area.duration_text.setText(f"{new_dur:.2f}")
            # expanded_label_area.update_label_area()

        # for item in label_area.getItems():
        #     self.datawindow.viewbox.removeItem(item)   

        # del self.datawindow.labels[current_idx]

        # if self.last_cursor_pos:
        #     x, y = self.last_cursor_pos
        #     self.hover(x, y)

   # TODO: CHANGE TO def merge_labels(self, area1: LabelArea, area2: LabelArea, area3: LabelArea = None, label: str = None) -> tuple[float, float]:
    def merge_adjacent_labels(self, label_area: LabelArea) -> None:
        """
        
        """
        labels = self.datawindow.labels
        idx = labels.index(label_area)

        before = labels[idx - 1] if idx > 0 else None
        after = labels[idx + 1] if idx + 1 < len(labels) else None

        merging_left = (before and before.label == label_area.label)
        merging_right = (after and after.label == label_area.label)

        def remove_label_area(area: LabelArea):
            """Helper function to remove label areas from the viewbox and update labels."""
            for item in area.getItems():
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)

            if area in self.datawindow.labels:
                if self.is_selected(area):
                    self.deselect_item(area)
                self.datawindow.labels.remove(area)

        if merging_left and merging_right:
            if self.is_selected(label_area):
                self.select(before)

            remove_label_area(label_area)
            remove_label_area(after)
            before.duration = before.duration + label_area.duration + after.duration
            before.update_label_area()

        elif merging_left:
            if self.is_selected(label_area):
                self.select(before)

            remove_label_area(label_area)
            before.duration = before.duration + label_area.duration
            before.update_label_area()

        elif merging_right:
            remove_label_area(after)
            label_area.duration = label_area.duration + after.duration
            label_area.update_label_area()

        if self.last_cursor_pos:
            x, y = self.last_cursor_pos
            self.hover(x, y)



    def hover(self, x: float, y: float):
        """
        Handles the actions performed when the mouse is at 
        a given (x, y) in ViewBox coordinates.
        """
        (x_min, x_max), (y_min, y_max) = self.datawindow.viewbox.viewRange()

        if not (x_min <= x <= x_max and y_min <= y <= y_max):  # cursor outside viewbox
            if self.highlighted_item is not None:
                self.unhighlight_item(self.highlighted_item)
            self.datawindow.setCursor(Qt.CursorShape.ArrowCursor)
            return

        item_to_highlight = self.get_hovered_item(x ,y)

        cursor = None
        if isinstance(item_to_highlight, InfiniteLine):
            if item_to_highlight == self.datawindow.labels[0].transition_line:
                return
            cursor = Qt.CursorShape.OpenHandCursor

        self.update_highlight(item_to_highlight, cursor=cursor)

    def get_hovered_item(self, x:float, y:float) -> InfiniteLine | LabelArea:
        TRANSITION_HIGHLIGHT_THRESHOLD = 3 * self.datawindow.devicePixelRatioF() # pixels
        BASELINE_HIGHLIGHT_THRESHOLD = 3 * self.datawindow.devicePixelRatioF()

        # Get distances to relevant items
        transition_line, transition_distance = self.datawindow.get_closest_transition(x)
        baseline, baseline_distance = self.datawindow.get_baseline_distance(y)
        label_area = self.datawindow.get_closest_label_area(x)

        if not self.datawindow.baseline.isVisible():
            baseline_distance = float('inf')  # ignore baseline if not visible

        if transition_distance <= TRANSITION_HIGHLIGHT_THRESHOLD:
            if transition_line == self.datawindow.labels[0].transition_line:
                return  # can't drag the leftmost transition
            hovered_item = transition_line
        elif baseline_distance <= BASELINE_HIGHLIGHT_THRESHOLD:
            hovered_item = baseline
        else:
            hovered_item = label_area

        self.hovered_item = hovered_item  # update attribute
        return hovered_item


    def update_highlight(self, item, cursor: Qt.CursorShape = None):
        """
        Handles styling a highlighted item and updating the cursor.
        """
        if cursor:
            self.datawindow.setCursor(cursor)
        else:
            self.datawindow.setCursor(Qt.CursorShape.ArrowCursor)  # default

        if self.is_selected(item):
            self.unhighlight_item(self.highlighted_item) # unhighlight previous item
            return  # don't highlight already selected items

        # Check if highlighted item needs to change:
        if self.highlighted_item != item:
            if self.highlighted_item is not None:
                self.unhighlight_item(self.highlighted_item) # unhighlight previous item
    
            if isinstance(item ,InfiniteLine):
                if item == self.datawindow.baseline:
                    item.setPen(self.highlighted_style['baseline'])
                else:
                    item.setPen(self.highlighted_style['transition line'])

            elif isinstance(item, LabelArea):
                label_area_color = self.datawindow.composite_on_white(Settings.label_to_color[item.label])
                text_background_color = item.label_background.brush().color()

                highlighted_area_color = self.get_highlighted_color(label_area_color)
                highlighted_background_color = self.get_highlighted_color(text_background_color)
                item.area.setBrush(mkBrush(color=highlighted_area_color))
                item.label_background.setBrush(mkBrush(highlighted_background_color))
                item.duration_background.setBrush(mkBrush(highlighted_background_color))

            self.highlighted_item = item

    def get_highlighted_color(self, color: QColor) -> QColor:
        """
        Computes the highlighted color for a given QColor.
        """
        h, s, l, a = color.getHslF()
        #highlighted_color = QColor.fromHslF(h, min(s * 5, 1), l * 0.9, a)
        highlighted_color = QColor.fromHslF(h, max(s, 1 - s**2), l * 0.9, a)    
        highlighted_color.setAlpha(200)
        return highlighted_color

    def unhighlight_item(self, item):
        if isinstance(item ,InfiniteLine):
            if item == self.datawindow.baseline:
                item.setPen(self.default_style['baseline'])
            else:
                item.setPen(self.default_style['transition line'])
        if isinstance(item, LabelArea):
            item.area.setBrush(mkBrush(color=Settings.label_to_color[item.label]))
            item.label_background.setBrush(mkBrush(color=item.get_background_color()))
            item.duration_background.setBrush(mkBrush(color=item.get_background_color()))

        self.highlighted_item = None

    def change_label_type(self, label_area: LabelArea, new_label: str) -> None:
        if self.is_selected(label_area): # label area selected
            selected_label_areas = [label for label in self.selected_items if isinstance(label, LabelArea)]
            for label_area in selected_label_areas: # change without merging
                label_area.label = new_label
                label_area.update_label_area()
                label_area.area.setBrush(self.selected_style['area'])
                label_area.label_background.setBrush(self.selected_style['text background'])
                label_area.duration_background.setBrush(self.selected_style['text background'])
            for label_area in selected_label_areas[:]: # merge all labels if necessary
                if label_area in self.datawindow.labels:
                    self.merge_adjacent_labels(label_area)
        else: # label area highlighted
            label_area.label = new_label
            label_area.update_label_area()
            self.merge_adjacent_labels(label_area)



        dw = self.datawindow
        dw.viewbox.update()
        transitions = [(label_area.start_time, label_area.label) for label_area in dw.labels]
        dw.epgdata.set_transitions(dw.file, transitions, dw.transition_mode)