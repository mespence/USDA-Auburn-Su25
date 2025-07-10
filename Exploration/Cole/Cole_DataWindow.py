import sys
sys.path.append("/Users/cole/coding/bugs2025/USDA-Auburn-Su25/GUI")
import numpy as np
from numpy.typing import NDArray

from pyqtgraph import (
    PlotWidget, ViewBox, PlotItem, 
    TextItem, PlotDataItem, ScatterPlotItem, 
    mkPen,
)

from PyQt6.QtGui import QKeyEvent, QWheelEvent, QMouseEvent
from PyQt6.QtCore import Qt, QPointF, QTimer

from EPGData import EPGData
from Settings import Settings
from LabelArea import LabelArea

import time
#import pandas as pd

# DEBUG ONLY TODO remove imports for testing
from PyQt6.QtWidgets import QApplication  

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
        self.scroll_mode = False # i added this
        self.epgdata: EPGData = epgdata
        self.file: str = None
        self.prepost: str = "post"
        self.plot_item: PlotItem = (
            self.getPlotItem()
        )  # the plotting canvas (axes, grid, data, etc.)
        self.xy_data: tuple[NDArray] = []  # x and y data actually rendered
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
        self.transitions: list[tuple[float, str]] = []   # the x-values of each label transition
        self.transition_mode: str = 'labels'
        self.labels: list[LabelArea] = []  # the list of LabelAreas

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

        # placeholder sine wave
        self.xy_data.append(np.linspace(0, 1, 10000))
        self.xy_data.append(np.sin(2 * np.pi * self.xy_data[0]))
        self.curve.setData(
            self.xy_data[0], self.xy_data[1], pen=mkPen(color="b", width=2)
        )

        self.curve.setClipToView(True)
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
        self.scroll_mode = False
        super().resizeEvent(event)

    def window_to_chart(self, x: float, y: float) -> tuple[float, float]:
        """
        Converts from window (global) coordinates to chart (data) coordinates.

        Inputs:
            x, y: x and y coordinate of the window coordinate

        Returns:
            (chart_x, chart_y): chart coordinates equivalent to
            the window coordinates.
        """

        scene_pos = self.mapToScene(QPointF(x, y))
        data_pos = self.viewbox.mapSceneToView(scene_pos)
        return data_pos.x(), data_pos.y()

    def chart_to_window(self, x: float, y: float) -> tuple[float, float]:
        """
        Converts from chart (data) coordinates to window (widget) coordinates.

        Inputs:
            x, y: x and y coordinates in chart coordinates

        Returns:
            (window_x, window_y): window coordinates equivalent
            to the chart coordinates.
        """

        scene_pos = self.viewbox.mapViewToScene(QPointF(x, y))
        widget_pos = self.mapFromScene(scene_pos)
        return widget_pos.x(), widget_pos.y()

    def reset_view(self) -> None:
        """
        Resets the viewing window back to default
        settings (default zoom, scrolling, etc.)

        Inputs:
            None

        Returns:
            None
        """
        df = self.epgdata.get_recording(self.file, self.prepost)
        time = df["time"].values
        volts = df[self.prepost + self.epgdata.prepost_suffix].values

        self.viewbox.setRange(
            xRange=(min(time), max(time)), yRange=(min(volts), max(volts)), padding=0
        )
        #self.update_plot()


    def update_plot(self) -> None:
        """
        Updates the displayed data, labels, and compression/zoom 
        indicators.
        """
        print("updating")

        (x_min, x_max), _ = self.viewbox.viewRange()
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
        df = self.epgdata.get_recording(self.file, self.prepost)
        time = df["time"].values
        volts = df[prepost + self.epgdata.prepost_suffix].values

        self.xy_data[0] = time
        self.xy_data[1] = volts
        self.downsample_visible()
        self.curve.setData(self.xy_data[0], self.xy_data[1])

        self.viewbox.setRange(
            xRange=(min(time), max(time)), yRange=(min(volts), max(volts)), padding=0
        )
        self.update_plot()

    def downsample_visible(
        self, x_range: tuple[float, float] = None, max_points=3000
    ) -> tuple[NDArray, NDArray]:
        """
        Downsamples the data displayed in x_range to max_points using
        peak decimation (plotting both the max and min of each window).
        Modifies self.xy_data in-place.

        Inputs:
            x_range: a (x_min, x_max) tuple of the range of the data to be displayed
            max_points: the number of points (i.e., bins) to downsample to.

        Output:
           None

        TODO: Add other methods if this is too slow?
        """
        df = self.epgdata.get_recording(self.file, self.prepost)
        x = df["time"].values
        y = df[self.prepost + self.epgdata.prepost_suffix].values


        # Filter to x_range if provided
        if x_range is not None:
            x_min, x_max = x_range

            mask = (x >= x_min) & (x <= x_max)
            visible_x = x[mask]

            if len(visible_x) <= 250: 
                # render additional point on each side at very high zooms
                left_idx = np.searchsorted(x, x_min, side="left")
                right_idx = np.searchsorted(x, x_max, side="right")

                start = max(0, left_idx - 1)
                end = min(len(x), right_idx + 1)

                x = x[start:end]
                y = y[start:end]
            else:
                x = x[mask]
                y = y[mask]

        num_points = len(x)

        if num_points <= max_points or num_points < 2:  # no downsampling needed
            self.xy_data[0] = x
            self.xy_data[1] = y
            return


        # Peak decimation
        stride = max(1, num_points // (max_points // 2))  # each window gives 2 points
        num_windows = num_points // stride

        start_idx = (
            stride // 2
        )  # Choose a representative x (near center) for each window
        x_win = x[start_idx : start_idx + num_windows * stride : stride]
        x_out = np.repeat(x_win, 2)  # repeated for (x, y_min), (x, y_max)

        y_reshaped = y[: num_windows * stride].reshape(num_windows, stride)
        y_out = np.empty(num_windows * 2)
        y_out[::2] = y_reshaped.max(axis=1)
        y_out[1::2] = y_reshaped.min(axis=1)

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
        df = self.epgdata.get_recording(self.file, self.prepost)
        time = df['time'].values
        volts = df[self.prepost + self.epgdata.prepost_suffix].values
        self.transitions = self.epgdata.get_transitions(self.file, self.transition_mode)

        # Only continue if the label column contains labels
        if self.epgdata.dfs[file][self.transition_mode].isna().all():
            return

        durations = []  # elements of (label_start_time, label_duration, label)
        for i in range(len(self.transitions) - 1):
            time, label = self.transitions[i]
            next_time, _ = self.transitions[i + 1]
            durations.append((time, next_time - time, label))
        durations.append((self.transitions[-1][0], max(df['time']) - self.transitions[-1][0], self.transitions[-1][1]))

        for i, (time, dur, label) in enumerate(durations):
            label_area = LabelArea(time, dur, label, self)

            self.plot_item.addItem(label_area.transition_line)
            self.plot_item.addItem(label_area.area)
            self.plot_item.addItem(label_area.label_text)
            self.plot_item.addItem(label_area.duration_text)
            self.plot_item.addItem(label_area.label_background)
            self.plot_item.addItem(label_area.duration_background)

            self.labels.append(label_area)

        self.update_plot()

        

    def handle_transitions(self):
        return

    def handle_labels(self):
        return

    def set_baseline(self):
        return

    def add_drop_transitions(self):
        return

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_R:
            self.reset_view()
        if event.key() == Qt.Key.Key_A:
            self.scroll_mode = True
            self.autoScrollEvent() #yeet

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        return
        if event.key() == Qt.Key.Key_Shift:
            self.vertical_mode = False
        elif event.key() == Qt.Key.Key_Control:
            self.zoom_mode = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.cursor_state == "normal":
                self.handle_transitions(event, "press")
            else:
                self.set_baseline(event)
        elif event.button() == Qt.MouseButton.RightButton:
            self.add_drop_transitions(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        return
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_labels(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        return
        self.handle_transitions(event, "move")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        return
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_transitions(event, "release")

    def wheelEvent(self, event: QWheelEvent) -> None:
        # """
        # wheelEvent is called automatically whenever the scroll
        # wheel is engaged over the chart. We use it to control
        # horizontal and vertical scrolling along with zoom.
        # """
        self.scroll_mode = False
        self.viewbox.wheelEvent(event)
        # self.zoom_scroll(event)

    # def zoom_scroll(self, event: QWheelEvent):
    #     if self.zoom_mode:
    #         if self.vertical_mode:
    #             self.viewbox.setMouseEnabled(x=False, y=True)
    #         else:
    #             self.viewbox.setMouseEnabled(x=True, y=False)
    #     else:
    #         delta = event.angleDelta().y()
    #         # center = self.viewbox.mapToView(event.pos()
    #     # if self.vertical_mode:


    def autoscrollEvent(self, file: str, prepost: str = "post") -> None:
        
        # this all seems ok, or at least functional
        self.file = file
        self.prepost = prepost
        df = self.epgdata.get_recording(self.file, self.prepost)
        time = df["time"].values #can we make it add to an existing df instead of making a new one each loop?
        volts = df[prepost + self.epgdata.prepost_suffix].values
        (x_min, x_max), (y_min, y_max) = self.viewbox.viewRange() # if this breaks, try deleting "viewbox"
        x_zoomlevel = x_max - x_min # this gets the width of what will be translating across
        y_zoomlevel = y_max - y_min # this shouldnt change but i want it stored..

        # i dont think these things need to be changed
        self.xy_data[0] = time
        self.xy_data[1] = volts
        self.downsample_visible()
        self.curve.setData(self.xy_data[0], self.xy_data[1])

        while True: #this is hella inefficient can we have it only loop when theres new stuff
            if self.scroll_mode == True: #if we can move this into the checking for the lines that call this rather than the def that would be swag
            # at the current zoom, have the rightmost value be the most recent one in a loop that constantly makes this true
                
                with open("testdata1.csv", "r") as f:

                    line = f.readline()
                    if not line:
                        # this assumes we want to check on a schedule rather than every time the csv is updated... the latter would probably be better
                        time.sleep(0.01)  # Wait for new data
                        continue
                    else:
                        #dfline = pd.read_csv(line, sep = ",")
                        #df.loc[len(df)] = dfline #i think this is broken - iunno what line is exactly... i think it might just be text??
                        pass

                self.viewbox.setRange( # this is what's gonna get repeated a lot
                    xRange=(max(time)-x_zoomlevel, max(time)), yRange=(y_min, y_max), padding=0
                )
                self.update_plot()
            else: break #for when scroll mode changes, Stop Doing That


            #do i even need these? i do hope not... but i might... im gonna get it working and THEN ill optimize it
            #v_zoom_factor = 5e-4 #why?
            #self.translateBy(y=delta * v_zoom_factor * (y_max - y_min))
            #h_zoom_factor = 2e-4
            #self.translateBy(x=delta * h_zoom_factor * (x_max - x_min))
            
            #TODO: find out what delta is in this specific case

            # TODO: find what calls datawindow and copy it to mess with, and also figure out exactly what code makes an autoscroller autoscroll
            # and also possibly make a version of the program that's fed data from testdata1 which i tell to run. three terminals timeee
            # currently it calls itself, i think




# TODO: remove after feature-complete and integrated with main
def main():
    print("Testing new DataWindow class")
    Settings()
    app = QApplication([])

    epgdata = EPGData()
    epgdata.load_data("test_recording.csv")
    # epgdata.load_data(r'C:\EPG-Project\Summer\CS-Repository\Exploration\Jonathan\Data\smooth_18mil.csv')
    print("Data Loaded")
    
    window = DataWindow(epgdata)
    window.plot_recording(window.epgdata.current_file, 'post')
    window.plot_transitions(window.epgdata.current_file)

    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
