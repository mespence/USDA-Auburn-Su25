from numpy.typing import NDArray
from collections import deque
from queue import Queue, Empty
import numpy as np
import pandas as pd
from pandas import DataFrame
import sys
import threading
import json

from pyqtgraph import PlotWidget, PlotItem, ScatterPlotItem, PlotDataItem, mkPen, InfiniteLine

from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QWheelEvent, QMouseEvent, QCursor, QKeyEvent
from PyQt6.QtWidgets import QApplication, QPushButton, QDialog, QVBoxLayout, QLabel, QTextEdit, QDialogButtonBox

from PanZoomViewBox import PanZoomViewBox
from CommentMarker import CommentMarker
from TextEdit import TextEdit

# def simulate_incoming_data(receive_queue):
#     t = 0
#     while True:
#         val = np.sin(t)
#         receive_queue.put(f"{t:.2f},{val:.4f}")
#         # simulate 100 times a second incoming data
#         t += 0.01
#         time.sleep(0.01)

# def test_recording_data(receive_queue):
#     df = pd.read_csv(r'C:\EPG-Project\Summer\CS-Repository\GUI\test_recording.csv', usecols=['time', 'post_rect'])
#     times = df['time'].values
#     volts = df['post_rect'].values
#     interval = 0.01
#     next_time = time.perf_counter()

#     for i in range(len(times)):
#         t = times[i]
#         v = volts[i]
#         receive_queue.put(f"{t:.2f},{v:.4f}")

#         next_time += interval
#         sleep_time = sleep_time = max(0, next_time - time.perf_counter())
#         time.sleep(sleep_time)

class LiveDataWindow(PlotWidget):
    """
    Widget for visualizing real-time waveforms for incoming data streams

    Includes:
    - Displays live waveform data from a queue
    - Live mode with automatic scrolling
    - Pause/resume live view to scroll back on data
    - Zooming and panning (via `PanZoomViewBox`)
    
    Also handles rendering and downsampling for performance
    """
    def __init__(self):
        super().__init__(viewBox=PanZoomViewBox(datawindow=self))

        self.plot_item: PlotItem = self.getPlotItem()
        self.viewbox: PanZoomViewBox = self.plot_item.getViewBox() # the plotting area (no axes, etc.)
        self.viewbox.datawindow = self

        self.xy_data: list[NDArray] = [np.array([]), np.array([])]
        self.xy_rendered: list[NDArray] = [np.array([]), np.array([])]
        self.curve: PlotDataItem = PlotDataItem(pen=mkPen("blue", width=2))
        self.scatter: ScatterPlotItem = ScatterPlotItem(
            symbol="o", size=4, brush="blue"
        )  # the discrete points shown at high zooms
        self.zoom_level: float = 1

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
        self.leading_line: InfiniteLine = InfiniteLine(pos=0, angle=90, movable=False, pen=mkPen("red", width=3))
        self.addItem(self.leading_line)

        # first sec enable auto range
        self.autorange_disabled = False
        self.plot_item.enableAutoRange()

        # Live mode button
        self.live_mode = True

        self.current_time = 0
        # for live view, follow only 10 seconds of visible data
        self.default_scroll_window = 10
        self.auto_scroll_window = 10

        # COMMENTS
        self.comments: dict[float, CommentMarker] = {} # the dict of Comments
        self.comment_editing = False

        # comment preview only for moving comment maybe can put this in moving comment func
        self.comment_preview: InfiniteLine = InfiniteLine(
            angle = 90, movable = False,
            pen=mkPen("gray", style = Qt.PenStyle.DashLine, width = 3),
        )

        self.addItem(self.comment_preview)
        self.comment_preview.setVisible(False)

        self.comment_preview_enabled: bool = False
        self.moving_comment: CommentMarker = None
        
        self.update_plot()

    def closeEvent(self, event):
        self.socket_server.stop()
        super().closeEvent(event)

    def window_to_viewbox(self, point: QPointF) -> QPointF:
        """
        Converts between window (screen) coordinates and data (viewbox) coordinates.

        Parameters:
            point (QPointF): Point in global coordinates.

        Returns:
            QPointF: The corresponding point in data coordinates.
        """      
        scene_pos = self.mapToScene(point.toPoint())
        data_pos = self.viewbox.mapSceneToView(scene_pos)
        return data_pos
    
    def recv_queue_loop(self):
        while self.socket_client.running:
            try:
                # can include multiple commands/data in one message
                raw_message = self.socket_client.recv_queue.get(timeout=1.0)
            except Empty:
                continue  # restart the loop

            try:
                # parse message
                message_list = raw_message.split("\n")
                messages = [json.loads(s) for s in message_list if s.strip()]

                if '' in message_list:
                    message_list.remove('')

                for message in messages:
                    if message['type'] != 'data':
                        # message is for sliders
                        continue

                    time = float(message['value'][0])
                    volt = float(message['value'][1])

                    # this copies the np array each time to append so O(n)
                    self.xy_data[0] = np.append(self.xy_data[0], time)
                    self.xy_data[1] = np.append(self.xy_data[1], volt)   

                    # update latest time input
                    self.current_time = time
                self.update_plot()

            except Exception as e:
                print("[RECIEVE LOOP ERROR]", e)
        
    def update_plot(self):
        """
        if live mode enabled then only plot visible data across 10 sec
        else plot visible data of what the user scrolls to view
        """
        self.viewbox.setLimits(xMin=None, xMax=None, yMin=None, yMax=None) # clear stale data (avoids warning)

        # only enable autorange for first second
        if self.current_time > 1 and not self.autorange_disabled:
            self.plot_item.disableAutoRange()
            self.autorange_disabled = True

        if self.live_mode:
            end = self.current_time
            start = end - self.auto_scroll_window
            self.viewbox.setXRange(start, end, padding=0)
            self.downsample_visible(x_range=(start, end))
            self.leading_line.setPos(end)

        else:
            (x_min, x_max), _ = self.viewbox.viewRange()
            self.downsample_visible(x_range=(x_min, x_max))
            self.leading_line.setPos(self.current_time)

        # SCATTER
        (x_min, x_max), _ = self.viewbox.viewRange()
        plot_width = self.viewbox.geometry().width() * self.devicePixelRatioF()
        time_span = x_max - x_min

        if time_span == 0:
            return float("inf")  # avoid division by zero
        
        pix_per_second = plot_width / time_span
        default_pix_per_second = plot_width / self.default_scroll_window
        self.zoom_level = pix_per_second / default_pix_per_second

        # scatter if zoom is greater than 300%
        if self.zoom_level >= 3:
            self.scatter.setVisible(True)
            self.scatter.setData(self.xy_rendered[0], self.xy_rendered[1])
        else:
            self.scatter.setVisible(False)

        self.curve.setData(self.xy_rendered[0], self.xy_rendered[1])
        self.viewbox.update()

    def set_live_mode(self, enabled: bool):
        self.live_mode = enabled
        self.update_plot()
        return
    
    def downsample_visible(
        self, x_range: tuple[float, float] = None, max_points=4000, method = 'peak'
    ) -> tuple[NDArray, NDArray]:
        """
        Downsamples waveform data in the visible range using the selected method. Modifies self.xy_rendered.
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
  
  
            x = x[left_idx:right_idx].copy()
            y = y[left_idx:right_idx].copy()   
    
        num_points = len(x)

        if num_points <= max_points:  # no downsampling needed
            # referencing self.xy_data
            self.xy_rendered[0] = x
            self.xy_rendered[1] = y
            return

        if method == 'subsampling': 
            stride = num_points // max_points
            x_out = x[::stride].copy()
            y_out = y[::stride].copy()

        elif method == 'mean':
            stride = num_points // max_points
            num_windows = num_points // stride
            start_idx = stride // 2
            x_out = x[start_idx : start_idx + num_windows * stride : stride].copy()
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

            # if receive another mismatched shape error try this
                # # Safely calculate number of full windows
                # stride = max(1, num_points // (max_points // 2))
                # num_windows = num_points // stride
                # total_pts = num_windows * stride

                # # Slice to full window size
                # x_window = x[:total_pts]
                # y_window = y[:total_pts]

                # x_win = x_window[stride // 2::stride][:num_windows]  # in case of rounding issues
                # y_reshaped = y_window.reshape(num_windows, stride)

                # # Now generate x and y downsampled
                # x_out = np.repeat(x_win, 2)
                # y_out = np.empty(num_windows * 2)
                # y_out[::2] = y_reshaped.max(axis=1)
                # y_out[1::2] = y_reshaped.min(axis=1)
        else:
            raise ValueError(
                'Invalid "method" arugment. ' \
                'Please select either "subsampling", "mean", or "peak".'
            )

        self.xy_rendered[0] = x_out
        self.xy_rendered[1] = y_out

        # sort xy_rendered for ability to add comment to past and move comment
        sort_idx = np.argsort(self.xy_rendered[0])
        self.xy_rendered[0] = self.xy_rendered[0][sort_idx]
        self.xy_rendered[1] = self.xy_rendered[1][sort_idx]

    def add_comment_dialog(self, comment_time: float) -> str | None:
        """ Opens dialog to enter a comment for given timestamp.
        Returns the comment text if accepted and non-empty, otherwise
        returns None """

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Add Comment @ {comment_time:.2f}s")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Add Comment:"))
        text = TextEdit()
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # enter pressed, dialog accepts
        text.returnPressed.connect(dialog.accept)

        # focus + default button behavior
        text.setFocus()
        buttons.button(QDialogButtonBox.StandardButton.Save).setAutoDefault(True)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        text = text.toPlainText().strip()
        return text if text else None

        save_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # if the text was just spaces/an empty comment, then don't create a comment
        text = text.toPlainText().strip()
        return text if text else None

    def add_comment_to_past(self, click_time: float) -> None:
        comment_time = self.find_nearest_time(click_time)

        text = self.add_comment_dialog(comment_time)
        if text is None:
            return
    
        # create comment
        new_marker = CommentMarker(comment_time, text, self)
        self.comments[comment_time] = new_marker

        self.update_plot()
    
    def add_comment_at_current(self) -> None:
        # called when click add comment button or shift+Space when in live/paused mode

        # have live view paused in background
        comment_time = self.current_time
        text = self.add_comment_dialog(comment_time)
        if text is None:
            return
    
        # create comment
        # i thnk commentmarker handles viisbility out of range
        new_marker = CommentMarker(comment_time, text, self)
        self.comments[comment_time] = new_marker

        self.update_plot()

    def move_comment_helper(self, marker: CommentMarker):
        self.moving_comment = marker
        self.comment_preview_enabled = True
        self.comment_preview.setVisible(True)

        x_pos = self.viewbox.mapSceneToView(self.mapToScene(self.mapFromGlobal(QCursor.pos()))).x()
        self.comment_preview.setPos(x_pos)
        
        self.viewbox.update()
        return
    
    def move_comment(self, marker: CommentMarker, click_time: float) -> None:
        old_time = marker.time
        text = self.comments[old_time].text

        # update marker in viewbox
        # update comments dict
        old_marker = self.comments.pop(old_time)
        old_marker.remove()
        new_time = self.find_nearest_time(click_time)
        new_marker = CommentMarker(new_time, text, self)
        self.comments[new_time] = new_marker

        self.comment_preview_enabled = False
        self.comment_preview.setVisible(False)

        self.update_plot()
        return
    
    def find_nearest_time(self, time: float) -> float:
        """ for add comment in past and move comment need to find nearest valid time index"""
        # xy rendered sorted in downsampling
        # find insertion point
        x = self.xy_rendered[0]
        idx = np.searchsorted(x, time)

        # find nearest point
        if idx == 0:
            nearest_idx = 0
        elif idx >= len(x):
            nearest_idx = len(x) - 1
        else:
            left = x[idx - 1]
            right = x[idx]
            if abs(right - time) < abs(time - left):
                nearest_idx = idx
            else:
                nearest_idx = idx - 1

        nearest_time = x[nearest_idx]
        return nearest_time
    
    def delete_comment(self, time: float) -> None:
        # update dict
        marker = self.comments.pop(time)
        # remove marker from viewbox
        marker.remove()
        return
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """ only move commment and add past comment func for now"""
        super().mousePressEvent(event)

        point = self.window_to_viewbox(event.position())
        x, _ = point.x(), point.y()

        if event.button() == Qt.MouseButton.LeftButton:
            # for moving comment
            if self.comment_preview_enabled and self.moving_comment is not None:
                self.move_comment(self.moving_comment, x)
                self.moving_comment = None
        elif event.button() == Qt.MouseButton.RightButton:
            # for past comment creation
            self.add_comment_to_past(x)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handles key shortcuts for adding comment at current time ("Shift+Space").

        Parameters:
            event (QKeyEvent): The key press event.
        """
        if event.key() == Qt.Key.Key_Space and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            print("add")
            self.add_comment_at_current() 

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Delegates interaction to comment handlers.

        Parameters:
            event (QMouseEvent): The mouse move event.
        """
        super().mouseMoveEvent(event)

        point = self.window_to_viewbox(event.position())
        x, _ = point.x(), point.y()

        (x_min, x_max), (y_min, y_max) = self.viewbox.viewRange()

        if self.comment_preview_enabled:
            if x_min <= x <= x_max:
                self.comment_preview.setPos(x)
                self.comment_preview.setVisible(True)
            else:
                self.comment_preview.setVisible(False)

        self.update_plot()
        
        return

    def export_comments(self):
        """ what form should the comments be in """
        # have this as a menu option 
        pass

    def export_df(self) -> pd.DataFrame:
        """
        Exports the most recent data as a new df
        Returns a dataframe with time, voltage, and comments column
        """
        times = self.xy_data[0]
        volts = self.xy_data[1]

        if len(times) == 0:
            # no data, return empty df
            return DataFrame()
        
        df = DataFrame({
            "time": times,
            "voltages": volts, # may need to change based on what engineers plot
            "comments": [None] * len(times)
        })

        # add current comments to df
        for comment_time, comment_text in self.comment.items():
            df.loc[df['time'] == comment_time, 'comments'] = comment_text

        return df

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Forwards a scroll event to the custom viewbox.
        """
        self.viewbox.wheelEvent(event)
        
if __name__ == "__main__":

    #receive_queue = Queue()
    # threading.Thread(target=simulate_incoming_data, args=(receive_queue,), daemon=True).start()
    #threading.Thread(target=test_recording_data, args=(receive_queue,), daemon=True).start()

    app = QApplication([])
    window = LiveDataWindow()
    window.resize(1000, 500)
    window.show()
    sys.exit(app.exec())