from numpy.typing import NDArray
from collections import deque
from queue import Queue
import numpy as np
import sys, time, threading

from pyqtgraph import InfiniteLine, PlotWidget, PlotItem, PlotDataItem, ViewBox, mkPen 

from PyQt6.QtCore import QTimer, QRect
from PyQt6.QtWidgets import QApplication, QPushButton

def simulate_incoming_data(receive_queue):
    t = 0
    while True:
        val = np.sin(t)
        receive_queue.put(f"{t:.2f},{val:.4f}")
        # simulate 100 times a second incoming data
        t += 0.01
        time.sleep(0.01)

class LiveDataWindow(PlotWidget):
    """
    [EDIT]
    """
    def __init__(self, receive_queue):
        super().__init__()

        self.plot_item: PlotItem = self.getPlotItem()
        self.viewbox: ViewBox = self.plot_item.getViewBox()

        self.receive_queue: Queue = receive_queue
        self.xy_data: list[NDArray] = [np.array([]), np.array([])]
        self.curve: PlotDataItem = self.plot(pen=mkPen("blue", width=2))
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

        # follow only 8 seconds of visible data
        self.auto_scroll_window = 8
        self.current_time = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_from_queue)

        # update occurs 100 times / sec, can change
        self.timer.start(10)
    
    def read_from_queue(self):
        updated = False
        while not self.receive_queue.empty():
            line = self.receive_queue.get()
            print("Got point:", line)
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
        x_data = self.xy_data[0]
        y_data = self.xy_data[1]

        if self.live_mode:
            end = self.current_time
            start = end - self.auto_scroll_window
            visible = (x_data >= start) & (x_data <= end)
            x_plot = x_data[visible]
            y_plot = y_data[visible]

            self.viewbox.setXRange(start, end + 2, padding=0)
            self.leading_line.setPos(end)
        
        else:
            (x_min, x_max), _ = self.viewbox.viewRange()
            visible = (x_data >= x_min) & (x_data <= x_max)
            x_plot = x_data[visible]
            y_plot = y_data[visible]

            self.leading_line.setPos(x_data[-1])

        self.curve.setData(x_plot, y_plot)
        print("Plotting", len(x_plot), "VISIBLE points")

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
        
if __name__ == "__main__":

    receive_queue = Queue()
    threading.Thread(target=simulate_incoming_data, args=(receive_queue,), daemon=True).start()

    app = QApplication([])
    window = LiveDataWindow(receive_queue)
    window.resize(1000, 500)
    window.show()
    sys.exit(app.exec())