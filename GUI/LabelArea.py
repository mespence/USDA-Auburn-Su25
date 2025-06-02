import numpy as np

from pyqtgraph import *
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsScene
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QFont, QFontMetricsF, QPen

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
        self.duration: float  # duration of the label in sec
        self.duration_text: TextItem  # the text object holding the duration string
        self.duration_bbox: QRectF  # the bounding box around the duration text
        self.duration_background: QGraphicsRectItem  # the background fill for the dur text
        self.transition_line: PlotDataItem  # the vertical line starting the LabelArea
        self.area_lower_line: PlotDataItem  # the lower edge of the area
        self.area_upper_line: PlotDataItem  # the upper edge of the area
        self.area: FillBetweenItem   # the colored FillBetweenItem for the label
        
        
        
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
        
        self.transition_line = PlotDataItem(
            x = [time, time],
            y = [y_min, y_max],
            pen=mkPen(color='black', width=2)
        )
        self.area_lower_line = PlotDataItem(
            x = [time, time + dur],
            y = [y_min, y_min]
        )
        self.area_upper_line = PlotDataItem(
            x = [time, time + dur],
            y = [y_max, y_max]
        )
        self.area = FillBetweenItem(
            self.area_lower_line, 
            self.area_upper_line, 
            brush=mkBrush(color=Settings.label_to_color[self.label])
        )

        self.label_bbox = self.bounding_box(self.label_text)
        self.duration_bbox = self.bounding_box(self.duration_text)

        self.label_background = self.background_box(self.label_text)
        self.duration_background = self.background_box(self.duration_text)

        self.viewbox.addItem(self.label_background)
        self.viewbox.addItem(self.duration_background)


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

    
    def background_box(self, text_item: TextItem, color = None) -> QGraphicsRectItem:
        """
        Returns the background box around a TextItem.
        """
        if color is None:
            color = Settings.label_to_color[self.label].darker(150) # 50% darker

        # NOTE: the grid renders transparency on its own, so an 
        # opacity of 1 will still render a faded grid.
        opacity = 1

        bbox = self.bounding_box(text_item)

        # create QGraphicsRectItem
        bg_rect = QGraphicsRectItem(QRectF(0, 0, bbox.width(), bbox.height()))
        
        bg_rect.setBrush(mkBrush(color))  # fill color
        bg_rect.setPen(QPen(Qt.PenStyle.NoPen))  # line color
        bg_rect.setOpacity(opacity)

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

        NOTE: will the label itself ever need to be updated?
        """
        self.viewbox = self.plot_widget.getPlotItem().getViewBox()
        _, (y_min, y_max) = self.viewbox.viewRange()

        centered_x = self.start_time + self.duration / 2
        label_y = y_min + 0.05 * (y_max - y_min)
        duration_y = y_max - 0.05 * (y_max - y_min)

        # update area and transition line
        self.area_lower_line.setData(x=self.area_lower_line.getData()[0], y = [y_min, y_min])
        self.area_upper_line.setData(x=self.area_upper_line.getData()[0], y = [y_max, y_max])
        self.area.setCurves(self.area_lower_line, self.area_upper_line)
        self.area.setZValue(-10)
        self.transition_line.setData(x = [self.start_time, self.start_time], y = [y_min, y_max])
        
        # update text position
        self.label_text.setPos(centered_x, label_y)
        self.duration_text.setPos(centered_x, duration_y)


        # update backgrounds
        self.label_bbox = self.bounding_box(self.label_text)
        self.duration_bbox = self.bounding_box(self.duration_text)
        
        self.label_background.setRect(0, 0, self.label_bbox.width(), self.label_bbox.height())
        self.duration_background.setRect(0, 0, self.duration_bbox.width(), self.duration_bbox.height())
        
        self.label_background.setPos(self.label_bbox.topLeft())       
        self.duration_background.setPos(self.duration_bbox.topLeft())        


        self.update_visibility()
        

    def update_visibility(self) -> None:
        """
        Updates the visibility of the labels and backgrounds 
        based on intersection with transition lines.

        Also handles redrawing the text backgrounds, since for 
        some reason doing it at this point in the rendering 
        update prevents any desync.
        """

        label_overlapping = self.label_bbox.left() < self.start_time < self.label_bbox.right()
        dur_overlapping = self.duration_bbox.left() < self.start_time < self.duration_bbox.right()

        self.label_text.setOpacity(not label_overlapping)
        self.duration_text.setOpacity(not dur_overlapping)

        self.label_background.setVisible(not label_overlapping)
        self.duration_background.setVisible(not dur_overlapping)

        if self.plot_widget.enable_debug:
            label_box = self.draw_debug_box(self.label_bbox)
            dur_box = self.draw_debug_box(self.duration_bbox)
            self.plot_widget.debug_boxes.append(label_box)
            self.plot_widget.debug_boxes.append(dur_box)

    def draw_debug_box(self, bbox: QRectF, color='red') -> QGraphicsRectItem:
        """
        Adds a QRectF to the scene around a bounding box. The QRectF must have its position 
        set correctly before being passed to this function.

        Inputs:
            bbox: A QRectF with size and position set
            scene: the graphics scene to add the box to
            color: the color of the debug box (str or QColor)
        """
        box = QGraphicsRectItem(bbox)
        box.setPen(mkPen(color, width=1))
        box.setZValue(1000)
        self.viewbox.addItem(box)
        return box  
    




    
    # functions for manually setting label properties
    # unused rn, but maybe could use for manual label editing.


    # def set_transition_line(self, x_val: float, y_range: tuple[float, float]):
    #     """
    #     Sets the LabelArea's transition line to the specified x-value and y_range.

    #     Inputs:
    #         x_val: the x-value to draw the line at
    #         y_range: a (y_min, x_max) pair specifying the range of the line
    #     """
    #     self.transition_line.setData([x_val, x_val], [y_range[0], y_range[1]])

    # def set_area_line(
    #         self, 
    #         x_range: tuple[float, float], 
    #         y_val: float, 
    #         upper: bool = False, 
    #         sampling_rate: float = 100
    #     ) -> None:
    #     """
    #     Sets the corresponding area line based on the given x_range and y-value.

    #     Inputs:
    #         x_range: a (x_min, x_max) pair specifying the length of the line
    #         y_val: the y-value to plot the line at
    #         upper: whether to update the upper or lower area line.
    #         sampling_rate: the rate/sec that the data was sampled at  

    #     Output:
    #         None
    #     """
    #     x_data = np.arange(x_range[0], x_range[1], 1 / sampling_rate)
    #     y_data = np.full_like(x_data, y_val)

    #     if upper:
    #         self.area_upper_line.setData(x_data, y_data)
    #     else:
    #         self.area_lower_line.setData(x_data, y_data)


    # def set_label(self, label: str) -> None:
    #     self.label = label
    #     self.label_text.setText(label)

    # def set_duration(self, duration: float) -> None:
    #     self.duration = duration
    #     self.duration_text.setText(duration)