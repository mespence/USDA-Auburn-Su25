from pyqtgraph import (
    PlotWidget, ViewBox,
    TextItem, InfiniteLine, LinearRegionItem,
    mkPen, mkBrush,
)
from PyQt6.QtWidgets import QGraphicsRectItem
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QFont, QFontMetricsF, QPen, QColor

from Settings import Settings

class LabelArea():
    """
    A class handling everything pertaining to EPG labels.
    Includes the label and duration text, the plot objects 
    for label coloring, and the background/bounding boxes of text.
    """
    def __init__(self, time: float, dur: float, label: str, plot_widget: PlotWidget):
        self.plot_widget: PlotWidget  # the parent PlotWidget
        self.viewbox: ViewBox # the ViewBox object that the LabelArea is being rendered in
        self.text_metrics: QFontMetricsF  # metrics about the font for the text
        self.start_time: float # the time at the start of the label

        self.label: str  # the label string
        self.label_text: TextItem  # the text object holding the label string
        self.label_bbox: QRectF  # the bounding box around the label text
        self.label_background: QGraphicsRectItem  # the background fill for the label text
        self.label_debug_box: QGraphicsRectItem  # the debug bbox for the label

        self.duration: float  # duration of the label in sec
        self.duration_text: TextItem  # the text object holding the duration string
        self.duration_bbox: QRectF  # the bounding box around the duration text
        self.duration_background: QGraphicsRectItem  # the background fill for the dur text
        self.duration_debug_box: QGraphicsRectItem  # the debug bbox for the duration

        self.enable_debug: bool # whether to show debug boxes

        self.transition_line: InfiniteLine  # the vertical line starting the LabelArea
        self.area: LinearRegionItem   # the colored FillBetweenItem for the label
        
        
        # ----- initialize instance variables --------------------------

        self.plot_widget = plot_widget
        self.viewbox = self.plot_widget.getPlotItem().getViewBox()
        self.start_time = time

        _, (y_min, y_max) = self.viewbox.viewRange()

        centered_x = time + dur / 2
        label_y = y_min + 0.05 * (y_max - y_min)
        duration_y = y_max - 0.05 * (y_max - y_min)

        font = QFont('Sans', 10)
        self.text_metrics = QFontMetricsF(font)

        self.label = label   
        self.label_text = TextItem(label, color='black', anchor=(0.5, 0.5))
        self.label_text.setFont(font)
        self.label_text.setPos(centered_x, label_y)
        self.viewbox.addItem(self.label_text)

        self.duration = dur
        self.duration_text = TextItem(str(round(dur, 2)), color='black', anchor=(0.5, 0.5))
        self.duration_text.setFont(font)
        self.duration_text.setPos(centered_x, duration_y)
        self.viewbox.addItem(self.duration_text)

        self.transition_line = InfiniteLine(
            pos=time,
            angle=90,  # vertical
            pen=mkPen(color='black', width=2),
            hoverPen=None,
            movable=False,
        )
        self.viewbox.addItem(self.transition_line)
  
        self.area = LinearRegionItem(
            values = (time, time + dur),
            orientation='vertical',
            brush=mkBrush(color=Settings.label_to_color[self.label]),
            hoverBrush=None,
            movable=False,
        )
        self.area.setZValue(-10)
        self.viewbox.addItem(self.area)

        self.label_bbox = self.bounding_box(self.label_text)
        self.duration_bbox = self.bounding_box(self.duration_text)

        self.label_background = self.background_box(self.label_text)
        self.viewbox.addItem(self.label_background)
        
        self.duration_background = self.background_box(self.duration_text)
        self.viewbox.addItem(self.duration_background)

        self.enable_debug = self.plot_widget.enable_debug

        self.viewbox.sigTransformChanged.connect(self.update_label_area)  # bug here after deleting all areas
    
        if self.enable_debug:
            self.toggle_debug_boxes()   

    def toggle_debug_boxes(self) -> None:
        if self.enable_debug:
            if not hasattr(self, "label_debug_box"):
                self.label_debug_box = QGraphicsRectItem(self.label_bbox)
                self.label_debug_box.setPen(mkPen(color='red'))
            else:   
                self.update_rect(self.label_debug_box, self.label_bbox)
            self.viewbox.addItem(self.label_debug_box)

            if not hasattr(self, "duration_debug_box"):
                self.duration_debug_box = QGraphicsRectItem(self.duration_bbox)
                self.duration_debug_box.setPen(mkPen(color='red'))
            else:
                self.update_rect(self.duration_debug_box, self.duration_bbox)
            self.viewbox.addItem(self.duration_debug_box)    

        else:
            if hasattr(self, "label_debug_box"):
                self.viewbox.removeItem(self.label_debug_box)
            if hasattr(self, "duration_debug_box"):
                self.viewbox.removeItem(self.duration_debug_box)

    def setVisible(self, visible: bool) -> None:
        for item in self.getItems():
            item.setVisible(visible)


    # def set_duration_visibility(self, state: bool) -> None:
    #     """
    #     Toggles the visibility of the duration text and background.
    #     """
    #     self.duration_text.setVisible(state)
    #     self.duration_background.setVisible(state)
    #     self.update_visibility()  # to handle intersections


    def bounding_box(self, text_item: TextItem) -> QRectF:
        """
        Returns the bounding box around a TextItem 
        based on the font metrics. Used in visibility 
        calculations.
        """
        # get bounding box from font metrics
        text = text_item.toPlainText()
        raw_rect = self.text_metrics.tightBoundingRect(text)
        raw_rect.adjust(-2,-2, 2, 2) # padding of 2 to match background box visuals

        # get anchor offset
        anchor_x, anchor_y = text_item.anchor
        offset_x = raw_rect.width() * anchor_x
        offset_y = raw_rect.height() * anchor_y

        # convert to ViewBox coordinates
        local_top_left = QPointF(-offset_x, -offset_y)
        local_bottom_right = local_top_left + QPointF(raw_rect.width(), raw_rect.height())
        view_top_left = text_item.mapToView(local_top_left)
        view_bottom_right = text_item.mapToView(local_bottom_right)
        
        return QRectF(view_top_left, view_bottom_right)

    
    def background_box(self, text_item: TextItem, color: QColor = None, alpha: int = 255) -> QGraphicsRectItem:
        """
        Returns the background box around a TextItem.
        """
        if color is None:
            color = self.plot_widget.composite_on_white(Settings.label_to_color[self.label]) 
            color = color.darker(110) # 10% darker
            color = QColor.fromHsvF(
                color.hueF(), 
                color.saturationF() * 1.8,  # 80% more saturated
                color.valueF(), 
                color.alphaF()
            )
            color.setAlpha(200)

        bbox = self.bounding_box(text_item)

        # create QGraphicsRectItem
        bg_rect = QGraphicsRectItem(QRectF(0, 0, bbox.width(), bbox.height()))
        bg_rect.setBrush(mkBrush(color))  # fill color
        bg_rect.setPen(QPen(Qt.PenStyle.NoPen))  # line color

        # Set Z-value below that of the TextItem, which 
        # renders at Z = 0 regardless of what you set it to.
        # (I think due to pyqtgraph =/= PyQt)
        bg_rect.setZValue(-1)

        return bg_rect
    
    
    
    def update_label_area(self) -> None:
        """
        Redraws the area based on the new y_range. 

        Input:
            y_range: a (y_min, y_max) tuple of the new vertical 
                     range to display the LabelArea over
        Output:
            None

        """
        self.viewbox = self.plot_widget.getPlotItem().getViewBox()
        _, (y_min, y_max) = self.viewbox.viewRange()

        centered_x = self.start_time + self.duration / 2
        label_y = y_min + 0.05 * (y_max - y_min)
        duration_y = y_max - 0.05 * (y_max - y_min)
        
        # update text position
        self.label_text.setPos(centered_x, label_y)
        self.duration_text.setPos(centered_x, duration_y)

        # update backgrounds
        self.label_bbox = self.bounding_box(self.label_text)
        self.duration_bbox = self.bounding_box(self.duration_text)
        
        self.update_rect(self.label_background, self.label_bbox)
        self.update_rect(self.duration_background, self.duration_bbox)

        if self.enable_debug:
            self.toggle_debug_boxes()
            # self.update_rect(self.label_debug_box, self.label_bbox)
            # self.update_rect(self.duration_debug_box, self.duration_bbox)
        else:
            self.toggle_debug_boxes()

        self.update_visibility()
        
    def update_rect(self, rect: QGraphicsRectItem, bbox: QRectF) -> None:
        """
        Updates a QGraphicsRectItem to a new QRectF.
        """
        rect.setRect(0, 0, bbox.width(), bbox.height())
        rect.setPos(bbox.topLeft())

    def update_visibility(self) -> None:
        """
        Updates the visibility of the text and backgrounds 
        based on intersection with transition lines.

        Also handles hiding/showing durations based on the settings
        """

        label_overlapping = self.label_bbox.left() < self.start_time < self.label_bbox.right()
        dur_overlapping = self.duration_bbox.left() < self.start_time < self.duration_bbox.right()

        self.label_text.setOpacity(not label_overlapping)
        self.duration_text.setOpacity(not dur_overlapping and Settings.show_durations)

        self.label_background.setVisible(not label_overlapping)
        self.duration_background.setVisible(not dur_overlapping and Settings.show_durations)


    def getItems(self):
        """
        Returns a list of the items added to the viewbox.
        """
        itemsToRemove = [
            self.area, self.label_text, self.duration_text, self.transition_line,
            self.label_background, self.duration_background
        ]

        if self.plot_widget.enable_debug:
            itemsToRemove.append(self.label_debug_box)
            itemsToRemove.append(self.duration_debug_box)

        return itemsToRemove

    # functions for manually setting label properties
    # unused rn, but maybe could use for manual label editing.


    def set_transition_line(self, x_val: float):
        """
        Sets the LabelArea's transition line to the specified x-value and y_range.

        Inputs:
            x_val: the x-value to draw the line at
        """
        
        self.transition_line.setValue(x_val)


    # def set_label(self, label: str) -> None:
    #     self.label = label
    #     self.label_text.setText(label)

    # def set_duration(self, duration: float) -> None:
    #     self.duration = duration
    #     self.duration_text.setText(duration)