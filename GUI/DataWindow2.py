import numpy as np
from numpy.typing import NDArray

from pyqtgraph import (
    PlotWidget, ViewBox, PlotItem, 
    TextItem, PlotDataItem, ScatterPlotItem, InfiniteLine,
    mkPen, mkBrush, setConfigOptions
)

from PyQt6.QtGui import (
    QKeyEvent, QWheelEvent, QMouseEvent, QColor, 
    QGuiApplication, QCursor, 
)
from PyQt6.QtCore import Qt, QPointF, QTimer

from PyQt6.QtWidgets import QGraphicsRectItem

from EPGData import EPGData
from Settings import Settings
from LabelArea import LabelArea

# DEBUG ONLY TODO remove imports for testing
from PyQt6.QtWidgets import QApplication  
import sys, time


import platform

if platform.system() == "Windows":
    print("Windows detected, running with OpenGL")
    setConfigOptions(useOpenGL = True)

class PanZoomViewBox(ViewBox):
    """
    Helper class to override the default ViewBox behavior
    of scroll -> zoom and drag -> pan.
    """

    def wheelEvent(self, event: QWheelEvent, axis=None) -> None:
        """
        Handles all wheel + modifier inputs and maps 
        them to the correct pan/zoom action.
        """
        datawindow: DataWindow = self.parentItem().getViewWidget()
        delta = event.angleDelta().y()
        modifiers = event.modifiers()

        ctrl_held = modifiers & Qt.KeyboardModifier.ControlModifier
        shift_held = modifiers & Qt.KeyboardModifier.ShiftModifier

        if ctrl_held:
            zoom_factor = 1.001**delta
            center = self.mapToView(event.position())
            if shift_held: 
                self.scaleBy((1, 1 / zoom_factor), center)
            else:
                self.scaleBy((1 / zoom_factor, 1), center)
        else:
            (x_min, x_max), (y_min, y_max) = self.viewRange()
            if shift_held:
                v_zoom_factor = 5e-4
                self.translateBy(y=delta * v_zoom_factor * (y_max - y_min))
            else:
                h_zoom_factor = 2e-4
                self.translateBy(x=delta * h_zoom_factor * (x_max - x_min))

        event.accept()
        datawindow.update_plot()

    def mouseDragEvent(self, event, axis=None) -> None:
        # Disable all mouse drag panning/zooming
        event.ignore()

class DataWindow(PlotWidget):
    def __init__(self, epgdata: EPGData) -> None:
        super().__init__(plotItem=PlotItem(viewBox=PanZoomViewBox()))
        self.epgdata: EPGData = epgdata
        self.file: str = None
        self.prepost: str = "post"
        self.plot_item: PlotItem = (
            self.getPlotItem()
        )  # the plotting canvas (axes, grid, data, etc.)
        self.xy_data: list[NDArray] = []  # x and y data actually rendered
        self.curve: PlotDataItem = PlotDataItem(antialias=False) 
        self.scatter: ScatterPlotItem = ScatterPlotItem(
            symbol="o", size=4, brush="blue"
        )  # the discrete points shown at high zooms
        self.viewbox: ViewBox = (
            self.plot_item.getViewBox()
        )  # the plotting area (no axes, etc.)
        #self.vertical_mode: bool = False  # whether scroll/zoom actions are vertical
        #self.zoom_mode: bool = False  # whether mouse wheel controls zoom
        self.cursor_mode: str = (
            "normal"  # cursor state, e.g. normal, baseline selection
        )
        self.compression: float = 0
        self.compression_text: TextItem = TextItem()
        self.zoom_level: float = 1
        self.zoom_text: TextItem = TextItem()
        #self.transitions: list[tuple[float, str]] = []   # the x-values of each label transition
        self.transition_mode: str = 'labels'
        self.labels: list[LabelArea] = []  # the list of LabelAreas

        # TODO: may need to clean this up based on how want preview to show up and how edit mode works
        self.baseline: InfiniteLine = InfiniteLine(
            angle = 0, movable=False, pen=mkPen("gray", width = 2)
        )
        self.plot_item.addItem(self.baseline)
        self.baseline.setVisible(False)
        
        self.baseline_preview: InfiniteLine = InfiniteLine(
            angle = 0, movable = False,
            pen=mkPen("gray", style = Qt.PenStyle.DashLine, width = 2),
        )

        self.addItem(self.baseline_preview)
        self.baseline_preview.setVisible(False)

        self.baseline_preview_enabled: bool = True
        self.edit_mode_enabled: bool = True
        self.moving_mode: bool = False  # whether an interactice item is being moved
        self.hovered_item = None
        self.selected_item = None

        self.selected_labels_rect = QGraphicsRectItem()
        self.scene().addItem(self.selected_labels_rect)

        self.initial_downsampled_data: list[NDArray, NDArray]  # cache of the dataset after the initial downsample

        self.viewbox.sigRangeChanged.connect(self.update_plot)

        self.initUI()

    def initUI(self) -> None:
        self.chart_width: int = 400
        self.chart_height: int = 400
        self.setGeometry(0, 0, self.chart_width, self.chart_height)

        self.setBackground("white")
        self.setTitle("<b>SCIDO Waveform Editor</b>", color="black", size="12pt")

        self.viewbox.setBorder(mkPen("black", width=3))

        self.plot_item.addItem(self.curve)
        self.plot_item.addItem(self.scatter)
        self.plot_item.setLabel("bottom", "<b>Time [s]</b>", color="black")
        self.plot_item.setLabel("left", "<b>Voltage [V]</b>", color="black")
        self.plot_item.showGrid(x=Settings.show_grid, y=Settings.show_grid)
        self.plot_item.layout.setContentsMargins(30, 30, 30, 20)
        self.plot_item.enableAutoRange(False)

        # placeholder sine wave
        self.xy_data.append(np.linspace(0, 1, 10000))
        self.xy_data.append(np.sin(2 * np.pi * self.xy_data[0]))
        self.curve.setData(
            self.xy_data[0], self.xy_data[1], pen=mkPen(color="b", width=2)
        )

        self.curve.setClipToView(False)  # already done in manual downsampling
        self.scatter.setVisible(False)
        self.curve.setZValue(-5)
        self.scatter.setZValue(-4)

        QTimer.singleShot(0, self.deferred_init)

        ## DEBUG/DEV TOOLS
        self.enable_debug = False
        self.debug_boxes = []
        Settings.show_durations = True



    def deferred_init(self) -> None:
        """
        Initalizes the items that need to be initalized after
        everything has been rendered to the screen.
        """
        self.compression = 0
        self.compression_text = TextItem(
            text=f"Compression: {self.compression: .1f}", color="black", anchor=(0, 0)
        )
        self.compression_text.setPos(QPointF(80, 15))
        self.scene().addItem(self.compression_text)

        self.zoom_level = 1
        self.zoom_text = TextItem(
            text=f"Zoom: {self.zoom_level * 100}%", color="black", anchor=(0, 0)
        )
        self.zoom_text.setPos(QPointF(80, 30))
        self.scene().addItem(self.zoom_text)
        
        # further defer update until the window is actually rendered to the screen
        QTimer.singleShot(0, self.update_plot)


    def resizeEvent(self, event) -> None:
        """
        Handles window resizing.
        """
        super().resizeEvent(event)
        #self.update_compression()

    def window_to_viewbox(self, point: QPointF) -> QPointF:
        """
        Converts a point from window (screen) coordinates to viewbox (data)
        coordinates.

        Inputs:
            point: point in widget coodinates

        Returns:
            QPointF: corresponding point in viewbox (data) coordinates
        """
        scene_pos = self.mapToScene(point.toPoint())
        data_pos = self.viewbox.mapSceneToView(scene_pos)
        #data_pos1 = self.viewbox.mapSceneToView(point)
        return data_pos

    def viewbox_to_window(self, point: QPointF) -> QPointF:
        """
        Converts from viewbox (data) coordinates to window (widget) coordinates.

        Inputs:
            x, y: x and y coordinates in viewbox coordinates

        Returns:
            (window_x, window_y): window coordinates equivalent
            to the viewbox coordinates.
        """
        return self.viewbox.mapViewToScene(point)
        scene_pos = self.viewbox.mapViewToScene(QPointF(x, y))
        return scene_pos.x(), scene_pos.y()
        #widget_pos = self.mapFromScene(scene_pos)
        #return widget_pos.x(), widget_pos.y()

    def reset_view(self) -> None:
        """
        Resets the viewing window back to default
        settings (default zoom, scrolling, etc.)

        Inputs:
            None

        Returns:
            None
        """
        QGuiApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))

        self.xy_data = [
            self.initial_downsampled_data[0].copy(), 
            self.initial_downsampled_data[1].copy()
        ]

        self.curve.setData(self.xy_data[0], self.xy_data[1])

    
        x_min, x_max = self.xy_data[0][0], self.xy_data[0][-1]
        y_min, y_max = np.min(self.xy_data[1]), np.max(self.xy_data[1])

        self.viewbox.setRange(
            xRange=(x_min, x_max), 
            yRange=(y_min, y_max), 
            padding=0
        )

        QGuiApplication.restoreOverrideCursor()



    def update_plot(self) -> None:
        """
        Updates the displayed data, labels, and compression/zoom 
        indicators.
        """
        #print("updating")

        (x_min, x_max), _ = self.viewbox.viewRange()

        self.viewbox.setLimits(xMin=None, xMax=None, yMin=None, yMax=None) # clear stale data (avoids warning)

        self.downsample_visible(x_range=(x_min, x_max))

        x_data = self.xy_data[0]
        y_data = self.xy_data[1]
        self.curve.setData(x_data, y_data)
        if len(x_data) <= 500:
            self.scatter.setVisible(True)
            self.scatter.setData(x_data, y_data)
        else:
            self.scatter.setVisible(False)

        self.update_compression()
        self.update_zoom()

        print(self.labels)
        for label_area in self.labels:
           label_area.update_label_area()

    def update_compression(self) -> None:
        """
        update_compression updates the compression readout
        based on the zoom level according to the formula
        COMPRESSION = (SECONDS/PIXEL) * 125
        obtained by experimentation with WINDAQ. Note that
        WINDAQ also has 'negative' compression levels for
        high levels of zooming out. We do not implement those here.

        TODO: Verify this formula

        Inputs:
            None

        Outputs:
            None
        """

        # Get the pixel distance of one second
        plot_width = self.viewbox.geometry().width() * self.devicePixelRatioF()

        (x_min, x_max), _ = self.viewbox.viewRange()
        time_span = x_max - x_min

        if time_span == 0:
            return float("inf")  # Avoid division by zero

        pix_per_second = plot_width / time_span
        second_per_pix = 1 / (pix_per_second)

        # Convert to compression based on WinDaq
        self.compression = second_per_pix * 125
        self.compression_text.setText(f"Compression Level: {self.compression :.1f}")

        # ----- CLINIC CODE -----------------
        # Update the compression readout.
        # Get the pixel distance of one second, we use a wide range to avoid rounding issues.
        # width = 1000
        # pix_per_second = (self.chart_to_window(width, 0)[0] - \
        #           self.chart_to_window(0, 0)[0]) / width
        # second_per_pix = 1 / (pix_per_second)
        # # Convert to compression based on WinDaq
        # self.compression = second_per_pix * 125
        # self.compression_text.setPlainText(f'Compression Level: {self.compression :.1f}')

    def update_zoom(self) -> None:
        """
        Updates the zoom readout based on the current
        scaling of the plot compared to a full rendering
        of the data.
        Inputs:
            None
        Outputs:
            None
        """
        plot_width = self.viewbox.geometry().width() * self.devicePixelRatioF()

        (x_min, x_max), _ = self.viewbox.viewRange()
        time_span = x_max - x_min

        pix_per_second = plot_width / time_span

        if time_span == 0:
            return float("inf")  # Avoid division by zero

        file_length_sec = self.epgdata.dfs[self.file]["time"].iloc[-1]
        default_pix_per_second = plot_width / file_length_sec

        self.zoom_level = pix_per_second / default_pix_per_second

        # leave off decimal if zoom level is int
        if abs(self.zoom_level - round(self.zoom_level)) < 1e-9:
            self.zoom_text.setText(f"Zoom: {self.zoom_level * 100: .0f}%")
        else:
            self.zoom_text.setText(f"Zoom: {self.zoom_level * 100: .1f}%")

    def plot_recording(self, file: str, prepost: str = "post") -> None:
        """
        plot_recording creates an NDArray for the time series given by file
        and updates the graph to show it.

        Inputs:
            file: a string containing the key of the recording
            prepost: a string containing either pre or post
                 to specify which recording is desired.

        Outputs:
            None
        """
        self.file = file
        self.prepost = prepost
        times, volts = self.epgdata.get_recording(self.file, self.prepost)

        self.xy_data[0] = times
        self.xy_data[1] = volts
        self.downsample_visible()
        self.curve.setData(self.xy_data[0], self.xy_data[1])
        init_x, init_y = self.xy_data[0].copy(), self.xy_data[1].copy()
        self.initial_downsampled_data = [init_x, init_y]

        self.viewbox.setRange(
            xRange=(np.min(self.xy_data[0]), np.max(self.xy_data[0])), 
            yRange=(np.min(self.xy_data[1]), np.max(self.xy_data[1])), 
            padding=0
        )
        self.update_plot()

    def downsample_visible(
        self, x_range: tuple[float, float] = None, max_points=4000, method = 'peak'
    ) -> tuple[NDArray, NDArray]:
        """
        Downsamples the data displayed in x_range to max_points using
        peak decimation (plotting both the max and min of each window).
        Modifies self.xy_data in-place.

        Inputs:
            x_range: a (x_min, x_max) tuple of the range of the data to be displayed
            max_points: the number of points (i.e., bins) to downsample to.
            method: "subsample" or "peak", which downsampling method to use.

        Output:
           None

        TODO: decide whether to pick a default or adapt to zoom
        """
        x, y = self.epgdata.get_recording(self.file, self.prepost)

        # Filter to x_range if provided
        if x_range is not None:
            x_min, x_max = x_range

            left_idx = np.searchsorted(x, x_min, side="left")
            right_idx = np.searchsorted(x, x_max, side="right")

            if right_idx - left_idx <= 250: 
                # render additional point on each side at very high zooms
                left_idx = max(0, left_idx - 1)
                right_idx = min(len(x), right_idx + 1)
  
            x = x[left_idx:right_idx]
            y = y[left_idx:right_idx]   
        
        
        num_points = len(x)

        if num_points <= max_points or num_points < 2:  # no downsampling needed
            self.xy_data[0] = x
            self.xy_data[1] = y
            return

        if method == 'subsampling':
        #     print("subsampling")
            stride = num_points // max_points
            x_out = x[::stride]
            y_out = y[::stride]
        elif method == 'mean':
            # print("mean")
            stride = num_points // max_points
            num_windows = num_points // stride
            start_idx = stride // 2
            x_out = x[start_idx : start_idx + num_windows * stride : stride] 
            y_out = y[:num_windows * stride].reshape(num_windows,stride).mean(axis=1)
        elif method == 'peak':
            #print("peak decimation")
            stride = max(1, num_points // (max_points // 2))  # each window gives 2 points
            num_windows = num_points // stride

            start_idx = stride // 2  # Choose a representative x (near center) for each window
            x_win = x[start_idx : start_idx + num_windows * stride : stride]
            x_out = np.repeat(x_win, 2)  # repeated for (x, y_min), (x, y_max)

            y_reshaped = y[: num_windows * stride].reshape(num_windows, stride)
            y_out = np.empty(num_windows * 2)
            y_out[::2] = y_reshaped.max(axis=1)
            y_out[1::2] = y_reshaped.min(axis=1)
        else:
            raise ValueError(
                'Invalid "method" arugment. ' \
                'Please select either "subsampling", "mean", or "peak".'
            )

        self.xy_data[0] = x_out
        self.xy_data[1] = y_out

    def plot_transitions(self, file: str) -> None:
        """
        plot_transition creates a vertical line for each transition and
        then colors the region after it and before the next transition
        accordingly.
        Inputs:
            file: a string containing the key of the recording
            prepost: a string indicating whether the pre- or
                     post- rectifier data is used
        Outputs:
            None
        """

        # clear old labels if present
        for label_area in self.labels:
            self.plot_item.removeItem(label_area.area)
            self.plot_item.removeItem(label_area.label_text)
            self.plot_item.removeItem(label_area.duration_text)
            label_area.transition_line.clear()
        self.labels = []


        # load data
        times, _ = self.epgdata.get_recording(self.file, self.prepost)
        transitions = self.epgdata.get_transitions(self.file, self.transition_mode)

        # Only continue if the label column contains labels
        if self.epgdata.dfs[file][self.transition_mode].isna().all():
            return

        durations = []  # elements of (label_start_time, label_duration, label)
        for i in range(len(transitions) - 1):
            time, label = transitions[i]
            next_time, _ = transitions[i + 1]
            durations.append((time, next_time - time, label))
        durations.append((transitions[-1][0], max(times) - transitions[-1][0], transitions[-1][1]))

        for i, (time, dur, label) in enumerate(durations):
            label_area = LabelArea(time, dur, label, self) # init. also adds items to viewbox
            self.labels.append(label_area)

        self.update_plot()

    
    def change_label_color(self, label: str, color: QColor) -> None:
        """
        change_label_color is a slot for the signal emitted by the
        SettingsWindow on changing a label color.

        Inputs:
            label: label for the waveform background to recolor.
            color: color to change the label to.

        Returns:
            None
        """
        for label_area in self.labels:
            if label_area.label == label:
                label_area.area.setBrush(mkBrush(color))

    def change_line_color(self, color: QColor) -> None:
        """
        change_line_color is a slot for the signal emitted by the
        SettingsWindow on changing the line color.

        Inputs:
            color: color to which the recording line is to be changed

        Returns:
            None
        """
        self.curve.setPen(mkPen(color))
        self.scatter.setPen(mkPen(color))  


    def delete_label_area(self, label_area: LabelArea) -> None:
        """
        Deletes the selected label area.
        Inputs:
            label: the LabelArea to delete
        Returns:
            Nothing
        """ 
        current_idx = self.labels.index(label_area)

        if len(self.labels) > 1:
            if label_area == self.labels[0]:
                # expand left
                expanded_label_area = self.labels[current_idx + 1]
                new_start_time = label_area.start_time
                new_range = [new_start_time, expanded_label_area.start_time +  expanded_label_area.duration]

                expanded_label_area.start_time = new_start_time
                expanded_label_area.set_transition_line(new_start_time)

            else: 
                # expand right
                expanded_label_area = self.labels[current_idx - 1]
                new_range = [expanded_label_area.start_time, label_area.start_time + label_area.duration]

            # expanded_label_area.area_lower_line.setData(x=new_x_range, y=lower_ys)
            # expanded_label_area.area_upper_line.setData(x=new_x_range, y=upper_ys)
            expanded_label_area.area.setRegion(new_range)
            new_dur = expanded_label_area.duration + label_area.duration 
            expanded_label_area.duration = new_dur
            expanded_label_area.duration_text.setText(str(round(new_dur, 2)))
            expanded_label_area.update_label_area()


        for item in label_area.getItems():
            self.viewbox.removeItem(item)   

        del self.labels[current_idx]
        #del self.transitions[current_idx]
        

    def highlight_item(self, event: QMouseEvent) -> None:
        """
        transition line > baseline > area
        """
        TRANSITION_THRESHOLD = 3
        BASELINE_THRESHOLD = 3

        if len(self.labels) == 0:  # nothing to highlight
            return 


        point = self.window_to_viewbox(event.position())
        x, y = point.x(), point.y()

        (x_min, x_max), (y_min, y_max) = self.viewbox.viewRange()
        pixelRatio = self.devicePixelRatioF()


        if not (x_min <= x <= x_max and y_min <= y <= y_max):  # cursor outside viewbox
            self.unhover(self.hovered_item)
            return
        
        transition_line, transition_distance = self.get_closest_transition(x)
        baseline_distance = self.get_baseline_distance(y)
        label_area = self.get_closest_label_area(x)
       
        if transition_distance <= TRANSITION_THRESHOLD * pixelRatio:
            if self.hovered_item is None or self.hovered_item != transition_line:
                if self.hovered_item is not None:
                    self.unhover(self.hovered_item)
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                transition_line.setPen(mkPen(width=6, color='#0D6EFD'))
                self.hovered_item = transition_line
        elif baseline_distance <= BASELINE_THRESHOLD * pixelRatio:
            if self.hovered_item is None or self.hovered_item != self.baseline:
                if self.hovered_item is not None:
                    self.unhover(self.hovered_item)
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                self.baseline.setPen(mkPen(width=6, color='#0D6EFD'))
                self.hovered_item = self.baseline
        else:
            if label_area is None:
                self.unhover(self.hovered_item)
                return
            if self.hovered_item is None or self.hovered_item != label_area:  
                if self.hovered_item is not None:
                    self.unhover(self.hovered_item)
                label_color = self.composite_on_white(Settings.label_to_color[label_area.label])
                h, s, l, a = label_color.getHslF()
                selected_color = QColor.fromHslF(h, min(s * 8, 1), l * 0.9, a)  
                selected_color.setAlpha(200)
                label_area.area.setBrush(mkBrush(color=selected_color))
                self.hovered_item = label_area

    
    def unhover(self, item):
        if isinstance(item, InfiniteLine):
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if item == self.baseline:
                item.setPen(mkPen("gray", width=2))
            else:
                item.setPen(mkPen(color='black', width=2))
        if isinstance(item, LabelArea):
            item.area.setBrush(mkBrush(color=Settings.label_to_color[item.label]))

        self.hovered_item = None

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
    
    def darker_hsl(self, color: QColor, amount: float) -> QColor:
        """
        Returns a darker color by reducing HSL lightness by `amount`.
        """
        h, s, l, a = color.getHsl()
        new_l = max(0, l - int(amount * 255))
        new_color = QColor.fromHsl(h, s, new_l, a)
        return new_color
        
    def get_closest_transition(self, x: float) -> tuple[int, float]:
        """
        Returns the index and pixel distance from a given x coordinate 
        to the closest transition line.

        Inputs: 
            x: the queried viewbox x-coordinate

        Outputs:
            index, x_distance: the index of and distance in pixels to the closest transition line
        """  
        transitions = np.array([label_area.start_time for label_area in self.labels])
        idx = np.searchsorted(transitions, x)

        zero_point = self.viewbox_to_window(QPointF(0,0)).x()
        
        if idx == len(transitions):
            dist = abs(transitions[idx-1] - x)
            return self.labels[idx-1].transition_line, self.viewbox_to_window(QPointF(dist, 0)).x() - zero_point
        else:
            # Check which of the two neighbors is closer
            dist_to_left = abs(x - transitions[idx-1])
            dist_to_right = abs(transitions[idx] - x)

            if dist_to_left <= dist_to_right:
                return self.labels[idx-1].transition_line, self.viewbox_to_window(QPointF(dist_to_left, 0)).x()- zero_point
            else:
                return self.labels[idx].transition_line, self.viewbox_to_window(QPointF(dist_to_right, 0)).x()- zero_point
            
    def get_baseline_distance(self, y: float) -> float:
        """
        Returns the pixel distance from a given y coordinate 
        to the baseline.

        Inputs: 
            y: the queried viewbox x-coordinate

        Outputs:
            y_distance: the distance in pixels to the baseline
        """
        if self.baseline is None:
            return float('inf')
        zero_point = self.viewbox_to_window(QPointF(0,0)).y()
        viewbox_distance = abs(y - self.baseline.value())
        return zero_point - self.viewbox_to_window(QPointF(0, viewbox_distance)).y()
    
    def get_closest_label_area(self, x: float) -> LabelArea:
        if x < self.labels[0].start_time or x > (self.labels[-1].start_time + self.labels[-1].duration):
            return None  # outside the labels
        label_ends = np.array([label.start_time + label.duration for label in self.labels])
        idx = np.searchsorted(label_ends, x)  # idk why this works
        if idx >= len(label_ends):
            return self.labels[-1]
        return self.labels[idx]      


    def delete_all_label_instances(self, label: str) -> None:
        """
        TODO: implement
        """

    def handle_transitions(self):
        return

    def handle_labels(self):
        return

    def set_baseline(self, event: QMouseEvent):
        """
        set_baseline creates a horizontal line where the
        user indicates with a click on the graph
        Inputs:
            event: the mouse event and where it was clicked
        Outputs:
            None
        """ 
        # TODO: edit for when have edit mode functionality
        point = self.window_to_viewbox(event.position())
        x, y = point.x(), point.y()

        (x_min, x_max), (y_min, y_max) = self.viewbox.viewRange()

        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return

        if self.baseline.getPos() == [0, 0]:
            self.baseline.setPos(y)
            self.baseline.setVisible(True)
        # else:
        #     # if baseline already placed, update position based on click
        #     # and be able to move it to a new position depending on edit mode
        #     self.baseline.setPos(y)
        #     #self.baseline.setMovable(self.edit_mode_enabled)
    
        return

    def add_drop_transitions(self):
        return

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_R:
            self.reset_view()  
        if event.key() == Qt.Key.Key_B:
            self.baseline_preview_enabled = True
            self.baseline_preview.setVisible(True)
            pos = self.viewbox.mapSceneToView(self.mapToScene(self.mapFromGlobal(QCursor.pos()))).y()
            self.baseline_preview.setPos(pos)
        if event.key() == Qt.Key.Key_Delete:
            if isinstance(self.selected_item, LabelArea):
                self.delete_label_area(self.selected_item)
                self.selected_item = None
            elif (
                isinstance(self.selected_item, list) 
                and all(isinstance(item, LabelArea) for item in self.selected_item)
            ):
                for label_area in self.selected_item:
                    self.delete_label_area(label_area)
                self.selected_item = None

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        return
        if event.key() == Qt.Key.Key_Shift:
            self.vertical_mode = False
        elif event.key() == Qt.Key.Key_Control:
            self.zoom_mode = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # TODO: edit for when have edit mode functionality
        # For testing baseline preview
        if event.button() == Qt.MouseButton.LeftButton:
            if self.baseline_preview_enabled:
                self.set_baseline(event)
                self.baseline_preview_enabled = False
                self.baseline_preview.setVisible(False)

            elif isinstance(self.hovered_item, InfiniteLine):
                self.moving_mode = True
                self.selected_item = self.hovered_item
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

                # if self.hovered_item == self.baseline:
                #     self.moving_mode = True
                #     return
                # else:
                #     pass
 
            elif isinstance(self.hovered_item, LabelArea):
                self.selected_item = self.hovered_item
                idx = self.labels.index(self.selected_item)
                left_line = self.labels[idx].transition_line
                if self.labels[idx] == self.labels[-1]:
                    right_line = left_line  # TODO: fix this to work with whatever we decide for last line
                else:
                    right_line = self.labels[idx+1].transition_line

                left_line.setPen(mkPen(width=6, color='#0D6EFD'))
                right_line.setPen(mkPen(width=6, color='#0D6EFD'))
                self.selected_item.area.setBrush(mkBrush(color='#0D6EFD80'))
                self.selected_item.label_background.setBrush(mkBrush(color='#0D6EFD90'))
                self.selected_item.duration_background.setBrush(mkBrush(color='#0D6EFD90'))
        

        super().mousePressEvent(event)
    
        # return
        # if event.button() == Qt.MouseButton.LeftButton:
        #     if self.cursor_state == "normal":
        #         self.handle_transitions(event, "press")
        #     else:
        #         self.set_baseline(event)
        # elif event.button() == Qt.MouseButton.RightButton:
        #     self.add_drop_transitions(event)
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        if self.moving_mode:
            self.moving_mode = False
            self.selected_item = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        
        return
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_transitions(event, "release")

        
        

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        return
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_labels(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        #super().mouseMoveEvent(event)
        # return

        if self.edit_mode_enabled and not self.moving_mode:
            self.highlight_item(event) 
            self.scene().update()            

        if self.baseline_preview_enabled:
            point = self.window_to_viewbox(event.position())
            y = point.y()
            _, (y_min, y_max) = self.viewbox.viewRange()
            if y_min <= y <= y_max:
                self.baseline_preview.setPos(y)
                self.baseline_preview.setVisible(True)
            else:
                self.baseline_preview.setVisible(False)

        if self.moving_mode and isinstance(self.selected_item, InfiniteLine):
            point = event.position()
            x = point.x()
            y = point.y()
            if self.hovered_item == self.baseline:
                self.baseline.setPos(y)
                return
            else: # must be transition line
                PIXEL_THRESHOLD = 2
                pixelRatio = self.devicePixelRatioF()
                
                for idx, label in enumerate(self.labels):
                    if self.selected_item == label.transition_line:
                        min_x = float("-inf")
                        max_x = float("inf")

                        if idx > 0:
                            prev_label = self.labels[idx-1]
                            prev_x = self.viewbox_to_window(QPointF(prev_label.start_time, 0))
                            min_x = prev_x.x() + PIXEL_THRESHOLD * pixelRatio
                        if idx < len(self.labels) - 1:
                            next_label = self.labels[idx+1]
                            next_x = self.viewbox_to_window(QPointF(next_label.start_time, 0))
                            max_x = next_x.x() - PIXEL_THRESHOLD * pixelRatio

                        bounding_window_x = max(min_x, min(x, max_x))
                        bounding_viewbox_x = self.window_to_viewbox(QPointF(bounding_window_x, y)).x()

                        delta_x = label.start_time - bounding_viewbox_x

                        label.start_time = bounding_viewbox_x
                        label.set_transition_line(bounding_viewbox_x)

                        print("bounding_vb", bounding_viewbox_x)
                        print("x_vb", x)
                        print("bounding_win", bounding_window_x)
                        print("x_win", event.position().x())

                        label.duration += delta_x
                        label.area.setRegion((label.start_time, label.start_time + label.duration))
                        label.duration_text.setText(str(round(label.duration, 2)))
                        label.update_label_area()

                        if idx > 0:
                            prev_label.duration -= delta_x
                            prev_label.area.setRegion((prev_label.start_time,
                                                           prev_label.start_time + prev_label.duration))
                            prev_label.duration_text.setText(str(round(prev_label.duration, 2)))
                            prev_label.update_label_area()

                        return
                # TODO:
                    # cant move past anoter transition line, stop pixels away
                    # add an end transition line with an end label area

            
            



        super().mouseMoveEvent(event)

        return

        self.handle_transitions(event, "move")

        self.baseline_preview: InfiniteLine = InfiniteLine(
            angle = 0, movable = False, pen=mkPen("blue",
                style = Qt.DashLine)
        )
        self.baseline_preview.hide()

    

    def wheelEvent(self, event: QWheelEvent) -> None:
        # """
        # wheelEvent is called automatically whenever the scroll
        # wheel is engaged over the chart. We use it to control
        # horizontal and vertical scrolling along with zoom.
        # """
        self.viewbox.wheelEvent(event)



# TODO: remove after feature-complete and integrated with main
def main():
    Settings()
    app = QApplication([])

    epgdata = EPGData()
    epgdata.load_data("test_recording.csv")
    #epgdata.load_data(r'C:\EPG-Project\Summer\CS-Repository\Exploration\Jonathan\Data\smooth_18mil.csv')
    print("Data Loaded")
    
    window = DataWindow(epgdata)
    window.plot_recording(window.epgdata.current_file, 'pre')
    window.plot_transitions(window.epgdata.current_file)

    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
