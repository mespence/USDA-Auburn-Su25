from pyqtgraph import (
    PlotWidget, ViewBox,
    TextItem, PlotDataItem, FillBetweenItem, 
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

        self.label_debug_box = QGraphicsRectItem(self.label_bbox)
        self.label_debug_box.setPen(mkPen(color='red'))
        self.label_debug_box.setVisible(self.plot_widget.enable_debug)
        self.viewbox.addItem(self.label_debug_box)

        self.duration_debug_box = QGraphicsRectItem(self.duration_bbox)
        self.duration_debug_box.setPen(mkPen(color='red'))
        self.duration_debug_box.setVisible(self.plot_widget.enable_debug)
        self.viewbox.addItem(self.duration_debug_box)

        self.viewbox.sigTransformChanged.connect(self.update_label_area)

    def set_duration_visibility(self, state: bool) -> None:
        """
        Toggles the visibility of the duration text and background.
        """
        self.duration_text.setVisible(state)
        self.duration_background.setVisible(state)
        self.update_visibility()  # to handle intersections


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
            color = self.composite_on_white(Settings.label_to_color[self.label]) 
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
    
    def composite_on_white(self, color: QColor) -> QColor:
        """
        Helps function to get the RGB value (no alpha) of 
        an RGBA color displayed on a white background.
        """
        alpha = color.alpha() / 255.0
        r = round(color.red() * alpha + 255 * (1 - alpha))
        g = round(color.green() * alpha + 255 * (1 - alpha))
        b = round(color.blue() * alpha + 255 * (1 - alpha))
        return QColor(r, g, b)
    
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
        
        self.update_rect(self.label_background, self.label_bbox)
        self.update_rect(self.duration_background, self.duration_bbox)

        self.update_rect(self.label_debug_box, self.label_bbox)
        self.update_rect(self.duration_debug_box, self.duration_bbox)

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
            self.area, self.label_text, self.duration_text, 
            self.label_background, self.duration_background
        ]
        if self.plot_widget.enable_debug:
            itemsToRemove.append(self.label_debug_box)
            itemsToRemove.append(self.duration_debug_box)
            
        return itemsToRemove

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