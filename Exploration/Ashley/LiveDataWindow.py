from numpy.typing import NDArray
from collections import deque
from queue import Queue
import numpy as np
import pandas as pd
import sys, time, threading

from pyqtgraph import InfiniteLine, PlotWidget, PlotItem, ScatterPlotItem, PlotDataItem, ViewBox, mkPen 

from PyQt6.QtCore import QTimer, QRect, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QPushButton

def simulate_incoming_data(receive_queue):
    t = 0
    while True:
        val = np.sin(t)
        receive_queue.put(f"{t:.2f},{val:.4f}")
        # simulate 100 times a second incoming data
        t += 0.01
        time.sleep(0.01)

def test_recording_data(receive_queue):
    df = pd.read_csv('/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/GUI/test_recording.csv', usecols=['time', 'post_rect'])
    times = df['time'].values
    volts = df['post_rect'].values
    interval = 0.01
    next_time = time.perf_counter()

    for i in range(len(times)):
        t = times[i]
        v = volts[i]
        receive_queue.put(f"{t:.2f},{v:.4f}")

        next_time += interval
        sleep_time = sleep_time = max(0, next_time - time.perf_counter())
        time.sleep(sleep_time)

class PanZoomViewBox(ViewBox):
    """
    Custom ViewBox that overrides default mouse/scroll behavior to support
    pan and zoom using wheel + modifiers.

    Pan/Zoom behavior:
    - Ctrl + Scroll: horizontal/vertical zoom (with Shift)
    - Scroll only: pan (horizontal or vertical based on Shift)
    """

    def __init__(self) -> None:
        super().__init__()
        self.datawindow: LiveDataWindow = None

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
            width, height = x_max - x_min, y_max - y_min

            if shift_held:
                v_zoom_factor = 5e-4
                dy = delta * v_zoom_factor * height
                self.translateBy(y=dy)
            else:
                h_zoom_factor = 2e-4
                dx = delta * h_zoom_factor * width

                new_x_min = x_min + dx
                new_x_max = new_x_min + width

                left_limit = 0 - 0.8*width

                # don't pan if it moves x=0 more than 0.80% across the ViewBox
                if new_x_min < left_limit:
                    pass  
                else:
                    self.translateBy(x=dx)

        event.accept()
        self.datawindow.update_plot()

    def mouseDragEvent(self, event, axis=None) -> None:
        event.ignore()

class LiveDataWindow(PlotWidget):
    """
    [EDIT]
    """
    def __init__(self, receive_queue):
        super().__init__(viewBox=PanZoomViewBox())

        self.plot_item: PlotItem = self.getPlotItem()
        self.viewbox: PanZoomViewBox = self.plot_item.getViewBox() # the plotting area (no axes, etc.)
        self.viewbox.datawindow = self

        self.receive_queue: Queue = receive_queue
        self.xy_data: list[NDArray] = [np.array([]), np.array([])]
        self.xy_rendered: list[NDArray] = [np.array([]), np.array([])]
        self.curve: PlotDataItem = self.plot(pen=mkPen("blue", width=2))
        self.scatter: ScatterPlotItem = ScatterPlotItem(
            symbol="o", size=4, brush="blue"
        )  # the discrete points shown at high zooms
        self.zoom_level: float = 1
        self.leading_line: InfiniteLine = InfiniteLine(pos=0, angle=90, movable=False, pen=mkPen("red", width=3))
        self.addItem(self.leading_line)

        # Setting up UI of graph (mimics datawindow)
        self.chart_width: int = 400
        self.chart_height: int = 400
        self.setGeometry(0, 0, self.chart_width, self.chart_height)

        self.setBackground("white")
        self.setTitle("<b>Live Waveform Viewer<b>", color="black", size="12pt")
        self.viewbox.setBorder(mkPen("black", width=3))

        self.plot_item.addItem(self.curve)
        self.plot_item.addItem(self.scatter)
        self.plot_item.setLabel("bottom", "<b>Time [s]</b>", color="black")
        self.plot_item.setLabel("left", "<b>Voltage [V]</b>", color="black")
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.layout.setContentsMargins(30, 30, 30, 20)
        self.plot_item.enableAutoRange("y", True)

        # Live mode button
        self.live_mode = True
        self.button = QPushButton("Pause Live View", self)
        self.button.setCheckable(True)
        self.button.setChecked(True)
        self.button.clicked.connect(self.live_mode_enabled)

        self.current_time = 0
        # for live view, follow only 8 seconds of visible data
        self.auto_scroll_window = 8
        # for live view, have 2 seconds of empty time in front of leading line
        self.leading_line_pos = 2
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_from_queue)

        # update occurs 100 times / sec, can change
        self.timer.start(10)
    
    def read_from_queue(self):
        updated = False
        while not self.receive_queue.empty():
            line = self.receive_queue.get()
            # assuming here text input is t,v  t,v  t,v
            time_str, volt_str = line.strip().split(",")
            time, volt = float(time_str), float(volt_str)

            # this copies the np array each time to append so O(n)
            self.xy_data[0] = np.append(self.xy_data[0], time)
            self.xy_data[1] = np.append(self.xy_data[1], volt)

            # update latest time input
            self.current_time = time
            updated = True
        
        if not updated:
            return
    
        # updated 100 times a sec, from self.timer.start(10)
        self.update_plot()

    def update_plot(self):
        """
        if live mode enabled then only plot visible data across 10 sec
        else plot visible data of what the user scrolls to view
        """
        (x_min, x_max), _ = self.viewbox.viewRange()

        self.viewbox.setLimits(xMin=None, xMax=None, yMin=None, yMax=None) # clear stale data (avoids warning)

        self.downsample_visible(x_range=(x_min, x_max))

        x_data = self.xy_data[0]
        y_data = self.xy_data[1]

        if self.live_mode:
            end = self.current_time
            start = end - self.auto_scroll_window
            visible = (x_data >= start) & (x_data <= end)
            self.xy_rendered = [x_data[visible], y_data[visible]]

            self.viewbox.setXRange(start, end + self.leading_line_pos, padding=0)
            self.leading_line.setPos(end)

            # dont show scatter during live mode
            self.zoom_level = 1
            self.scatter.setVisible(False)
        
        else:
            (x_min, x_max), _ = self.viewbox.viewRange()
            # ensure that the lines do not disappear when zooming in because near points out of view
            dx = 0.01
            visible = (x_data >= x_min - dx) & (x_data <= x_max + dx)
            self.xy_rendered = [x_data[visible], y_data[visible]]
            self.leading_line.setPos(self.current_time)

            # show scatter if zoom is greater than 300%
            plot_width = self.viewbox.geometry().width() * self.devicePixelRatioF()
            time_span = x_max - x_min
            pix_per_second = plot_width / time_span

            if time_span == 0:
                return float("inf")  # Avoid division by zero
            
            default_pix_per_second = plot_width / (self.auto_scroll_window + self.leading_line_pos)
            self.zoom_level = pix_per_second / default_pix_per_second

            if self.zoom_level >= 3:
                self.scatter.setVisible(True)
                self.scatter.setData(self.xy_rendered[0], self.xy_rendered[1])
            else:
                self.scatter.setVisible(False)

        self.curve.setData(self.xy_rendered[0], self.xy_rendered[1])
        self.viewbox.update()

    def live_mode_enabled(self):
        self.live_mode = self.button.isChecked()
        self.button.setText("Pause Live View" if self.live_mode else "Live View")

        if self.live_mode :
            self.plot_item.enableAutoRange("y", True)
        else:
            self.plot_item.enableAutoRange(False)
            (x_min, x_max), (y_min, y_max) = self.viewbox.viewRange()
            self.viewbox.setXRange(x_min, x_max, padding=0)
            self.viewbox.setYRange(y_min, y_max, padding=0)


            # tried doing the above in one setRange call but it weirdly still autoadjusts ?
                # (x_range), (y_range) = self.viewbox.viewRange()
                # self.viewbox.setRange(x_range, y_range, padding = 0)

    
    def downsample_visible(
        self, x_range: tuple[float, float] = None, max_points=4000, method = 'peak'
    ) -> tuple[NDArray, NDArray]:
        """
        Downsamples waveform data in the visible range using the selected method. Modifies self.xy_data in-place.

        Parameters:
            x_range (tuple[float, float]): Optional x-axis range to downsample.
            max_points (int): Max number of points to plot.
            method (str): 'subsample', 'mean', or 'peak' downsampling method.
        
        NOTE: 
            `subsample` samples the first point of each bin (fastest)
            `mean` averages each bin
            `peak` returns the min and max point of each bin (slowest, best looking)
        """
        x, y = self.xy_data

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
            self.xy_rendered[0] = x
            self.xy_rendered[1] = y
            return

        if method == 'subsampling': 
            stride = num_points // max_points
            x_out = x[::stride]
            y_out = y[::stride]
        elif method == 'mean':
            stride = num_points // max_points
            num_windows = num_points // stride
            start_idx = stride // 2
            x_out = x[start_idx : start_idx + num_windows * stride : stride] 
            y_out = y[:num_windows * stride].reshape(num_windows,stride).mean(axis=1)
        elif method == 'peak':
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

        self.xy_rendered[0] = x_out
        self.xy_rendered[1] = y_out

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Forwards a scroll event to the custom viewbox.
        """
        if not self.live_mode:
            self.viewbox.wheelEvent(event)
        else:
            event.ignore()
        
if __name__ == "__main__":

    receive_queue = Queue()
    # threading.Thread(target=simulate_incoming_data, args=(receive_queue,), daemon=True).start()
    threading.Thread(target=test_recording_data, args=(receive_queue,), daemon=True).start()

    app = QApplication([])
    window = LiveDataWindow(receive_queue)
    window.resize(1000, 500)
    window.show()
    sys.exit(app.exec())