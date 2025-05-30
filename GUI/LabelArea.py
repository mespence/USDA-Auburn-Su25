import numpy as np

from pyqtgraph import *
from PyQt6.QtWidgets import QGraphicsSimpleTextItem, QGraphicsItem, QGraphicsRectItem, QApplication
from PyQt6.QtCore import QPointF, QTimer, QRectF
from PyQt6.QtGui import QFont, QFontMetricsF

from Settings import Settings


class LabelArea():
    """
    
    """
    def __init__(self, time: float, dur: float, label: str, plot_widget: PlotWidget):
        self.plot_widget: PlotWidget  # the parent PlotWidget
        self.viewbox: ViewBox # the ViewBox object that the LabelArea is being rendered in
        self.label: str  # the label string
        self.label_text: TextItem
        self.start_time: float # the time at the start of the label
        self.duration: float  # duration of the label in sec
        self.duration_text: TextItem
        self.transition_line: PlotDataItem  # the vertical line starting the LabelArea
        self.area_lower_line: PlotDataItem  # the lower edge of the area
        self.area_upper_line: PlotDataItem  # the upper edge of the area
        self.area: FillBetweenItem   # the colored FillBetweenItem for the label
        
        # ----- initialize instance variables --------------------------
    
        self.plot_widget = plot_widget
        self.viewbox = self.plot_widget.getPlotItem().getViewBox()
        _, (y_min, y_max) = self.viewbox.viewRange()

        centered_x = time + dur / 2
        label_y = y_min + 0.05 * (y_max - y_min)
        duration_y = y_max - 0.05 * (y_max - y_min)

        font = QFont('Sans')
        font.setPointSize(10)

        self.label = label   
        self.label_text = TextItem(label, color='black', anchor=(0.5, 0.5))
        self.label_text.setFont(font)
        self.label_text.setPos(centered_x, label_y)
        self.viewbox.addItem(self.label_text)
        
        self.start_time = time

        self.duration = dur
        self.duration_text = TextItem(str(round(dur, 2)), color='black', anchor=(0.5, 0.5))
        self.duration_text.setFont(font)
        self.duration_text
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

    def bounding_box(self, text_item: TextItem) -> QRectF:
        # This is the local bounding box (0,0 at top-left of the text layout)
        font = text_item.textItem.font()
        text = text_item.toPlainText()
        metrics = QFontMetricsF(font)
        raw_rect = metrics.tightBoundingRect(text)

        # Get anchor offset
        anchor_x, anchor_y = text_item.anchor
        offset_x = raw_rect.width() * anchor_x
        offset_y = raw_rect.height() * anchor_y

        # Shift origin so that (0, 0) maps to the anchor point
        local_top_left = QPointF(-offset_x, -offset_y)
        local_bottom_right = local_top_left + QPointF(raw_rect.width(), raw_rect.height())

        # Map to scene coordinates
        scene_top_left = text_item.mapToScene(local_top_left)
        scene_bottom_right = text_item.mapToScene(local_bottom_right)

        return QRectF(scene_top_left, scene_bottom_right)
    
    def set_transition_line(self, x_val: float, y_range: tuple[float, float]):
        """
        Sets the LabelArea's transition line to the specified x-value and y_range.

        Inputs:
            x_val: the x-value to draw the line at
            y_range: a (y_min, x_max) pair specifying the range of the line
        """
        self.transition_line.setData([x_val, x_val], [y_range[0], y_range[1]])

    def set_area_line(
            self, 
            x_range: tuple[float, float], 
            y_val: float, 
            upper: bool = False, 
            sampling_rate: float = 100
        ) -> None:
        """
        Sets the corresponding area line based on the given x_range and y-value.

        Inputs:
            x_range: a (x_min, x_max) pair specifying the length of the line
            y_val: the y-value to plot the line at
            upper: whether to update the upper or lower area line.
            sampling_rate: the rate/sec that the data was sampled at  

        Output:
            None
        """
        x_data = np.arange(x_range[0], x_range[1], 1 / sampling_rate)
        y_data = np.full_like(x_data, y_val)

        if upper:
            self.area_upper_line.setData(x_data, y_data)
        else:
            self.area_lower_line.setData(x_data, y_data)

    def update_label_area(self, viewbox: ViewBox) -> None:
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

        self.area_lower_line.setData(x=self.area_lower_line.getData()[0], y = [y_min, y_min])
        self.area_upper_line.setData(x=self.area_upper_line.getData()[0], y = [y_max, y_max])

        self.area.setCurves(self.area_lower_line, self.area_upper_line)
        self.transition_line.setData(x = [self.start_time, self.start_time], y = [y_min, y_max])
        dpi_scale = self.viewbox.scene().views()[0].devicePixelRatioF()
        
        self.label_text.setPos(centered_x, label_y)
        self.duration_text.setPos(centered_x, duration_y)


         
    def set_label(self, label: str) -> None:
        self.label = label
        self.label_text.setText(label)

    def set_duration(self, duration: float) -> None:
        self.duration = duration
        self.duration_text.setText(duration)

    # def place_text_centered(self, text_item: QGraphicsSimpleTextItem, x: float = None, y: float = None) -> None :
    #     """
    #     Places a text object at the x/y coordinates centered on its own width/height.
    #     """
    #     if x is None:
    #         x = self.start_time + self.duration / 2
    #     if y is None:
    #         y = text_item.pos().y()  # possible bug with compounding centerings?
