import numpy as np
from numpy.typing import NDArray
import os

from pyqtgraph import (
    PlotWidget, ViewBox, PlotItem, 
    TextItem, PlotDataItem, ScatterPlotItem, InfiniteLine,
    mkPen, mkBrush, setConfigOptions
)

from PyQt6.QtGui import (
    QKeyEvent, QWheelEvent, QMouseEvent, QColor, 
    QGuiApplication, QCursor, QAction
)
from PyQt6.QtCore import Qt, QPointF, QTimer, QObject, QEvent

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QLabel, QDialog, QTextEdit, QMessageBox, QMenu

from EPGData import EPGData
from Settings import Settings
from LabelArea import LabelArea
from CommentMarker import CommentMarker
from SelectionManager import Selection


# DEBUG ONLY TODO remove imports for testing
from PyQt6.QtWidgets import QApplication  
import sys, time


if os.name == "nt":
    print("Windows detected, running with OpenGL")
    setConfigOptions(useOpenGL = True)

class PanZoomViewBox(ViewBox):
    """
    Helper class to override the default ViewBox behavior
    of scroll -> zoom and drag -> pan.
    """

    def __init__(self):
        super().__init__()
        self.datawindow: DataWindow = None

    def wheelEvent(self, event: QWheelEvent, axis=None) -> None:
        """
        Handles all wheel + modifier inputs and maps 
        them to the correct pan/zoom action.
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

                # don't pan if it moves x=0 more than halfway across the ViewBox
                if 0 > new_x_min + 0.5 * width:
                    pass  
                else:
                    self.translateBy(x=dx)

        event.accept()
        self.datawindow.update_plot()

    def mouseDragEvent(self, event, axis=None) -> None:
        # Disable all mouse drag panning/zooming
        event.ignore()


    def contextMenuEvent(self, event):
        if self.datawindow is None:
            self.datawindow = self.parentItem().getViewWidget()

        item = self.datawindow.selection.hovered_item
        if isinstance(item, InfiniteLine):
            print('Right-clicked InfiniteLine')
            return  # TODO: infinite line context menu not yet implemented
        
        menu = QMenu()
        label_type_dropdown = QMenu("Change Label Type", menu)

        label_names = list(Settings.label_to_color.keys())
        label_names.remove("END AREA")
        for label in label_names:            
            action = QAction(label, menu)
            action.setCheckable(True)

            if item.label == label:
                action.setChecked(True)
                
            action.triggered.connect(
                lambda checked, label_area=item, label=label:
                self.datawindow.selection.change_label_type(label_area, label)
            )
        
            label_type_dropdown.addAction(action)

        action2 = QAction("Custom Option 2")
        menu.addMenu(label_type_dropdown)
        menu.addAction(action2)

        action = menu.exec(event.screenPos())
        if action == label_type_dropdown:
            print("Option 1 selected")
        elif action == action2:
            print("Option 2 selected")

class GlobalMouseTracker(QObject):
    """
    Helper class to track mouse position through pop-ups and menus.
    """
    def __init__(self, datawindow):
        super().__init__()
        self.datawindow: DataWindow = datawindow

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseMove:
            global_pos = event.globalPosition().toPoint()
            local_pos = self.datawindow.mapFromGlobal(global_pos)
            point = self.datawindow.window_to_viewbox(QPointF(local_pos))
            x, y = point.x(), point.y()

            selection = self.datawindow.selection
            selection.last_cursor_pos = (x, y)
            selection.hovered_item = selection.get_hovered_item(x, y)
            if isinstance(selection.hovered_item, LabelArea):
                print(selection.hovered_item.label) 
        return super().eventFilter(obj, event)

class DataWindow(PlotWidget):
    def __init__(self, epgdata: EPGData) -> None:
        super().__init__(viewBox=PanZoomViewBox())
        self.plot_item: PlotItem = self.getPlotItem() # the plotting canvas (axes, grid, data, etc.)
        self.viewbox: PanZoomViewBox = self.plot_item.getViewBox() # the plotting area (no axes, etc.)
        self.viewbox.datawindow = self

        self.epgdata: EPGData = epgdata
        self.file: str = None
        self.prepost: str = "post"
        
        self.xy_data: list[NDArray] = []  # x and y data actually rendered to the screen
        self.curve: PlotDataItem = PlotDataItem(antialias=False) 
        self.scatter: ScatterPlotItem = ScatterPlotItem(
            symbol="o", size=4, brush="blue"
        )  # the discrete points shown at high zooms
        
        self.cursor_mode: str = "normal"  # cursor state, e.g. normal, baseline selection
        self.compression: float = 0
        self.compression_text: TextItem = TextItem()
        self.zoom_level: float = 1
        self.zoom_text: TextItem = TextItem()
        #self.transitions: list[tuple[float, str]] = []   # the x-values of each label transition
        self.transition_mode: str = 'labels'
        self.labels: list[LabelArea] = []  # the list of LabelAreas

        self.selection: Selection = Selection(self)

        self.viewbox.menu = None  # Disable default menu

        # BASELINE
        self.baseline: InfiniteLine = InfiniteLine(
            angle = 0, movable=False, pen=mkPen("gray", width = 3)
        )
        self.plot_item.addItem(self.baseline)
        self.baseline.setVisible(False)
        
        self.baseline_preview: InfiniteLine = InfiniteLine(
            angle = 0, movable = False,
            pen=mkPen("gray", style = Qt.PenStyle.DashLine, width = 3),
        )

        self.addItem(self.baseline_preview)
        self.baseline_preview.setVisible(False)

        self.baseline_preview_enabled: bool = False

        # COMMENTS
        self.comments: dict[float, CommentMarker] = {} # the dict of Comments
        self.comment_editing = False

        self.comment_preview: InfiniteLine = InfiniteLine(
            angle = 90, movable = False,
            pen=mkPen("gray", style = Qt.PenStyle.DashLine, width = 3),
        )

        self.addItem(self.comment_preview)
        self.comment_preview.setVisible(False)

        self.comment_preview_enabled: bool = False

        self.moving_mode: bool = False  # whether an interactice item is being moved
        self.edit_mode_enabled: bool = True
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
        self.update_compression()

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
        # scene_pos = self.viewbox.mapViewToScene(QPointF(x, y))
        # return scene_pos.x(), scene_pos.y()
        # widget_pos = self.mapFromScene(scene_pos)
        # return widget_pos.x(), widget_pos.y()

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

        if not self.isVisible():
            return  # don't run prior to initialization

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
        df = self.epgdata.dfs[file]

        # create a comments column if doesn't yet exist in df
        if 'comments' not in df.columns:
            df['comments'] = None

        # testing comment appearance
        tar_time = 420
        nearest_idx = (df['time'] - tar_time).abs().idxmin()
        df.at[nearest_idx, 'comments'] = "Test comment"

        self.viewbox.setRange(
            xRange=(np.min(self.xy_data[0]), np.max(self.xy_data[0])), 
            yRange=(np.min(self.xy_data[1]), np.max(self.xy_data[1])), 
            padding=0
        )

        self.update_plot()
        self.plot_comments(file)

    def plot_comments(self, file: str) -> None:
        """
        plot_comments adds pre-existing comments from the file
        to the viewbox.

        Inputs:
            file: a string containing the key of the recording
        Outputs:
            None
        """
        if file is not None:
            self.file = file

        df = self.epgdata.dfs[self.file]

        for marker in self.comments:
            marker.remove()
        self.comments.clear()
        
        comments_df = df[~df["comments"].isnull()]
        for time, text in zip(comments_df["time"], comments_df["comments"]):
            marker = CommentMarker(time, text, self, icon_path="message.svg")
            self.comments[time] = marker
        
        return

    def add_comment(self, event: QMouseEvent) -> None:
        """
        add_comments adds a new comment at the time indicated
        from a click event to the viewbox.

        Inputs:
            event: the mouse event and where it was clicked
        Outputs:
            None
        """
        point = self.window_to_viewbox(event.position())
        x = point.x()

        df = self.epgdata.dfs[self.file]
        # find nearest time clicked
        nearest_idx = (df['time'] - x).abs().idxmin()
        comment_time = df.at[nearest_idx, 'time']
        existing = df.at[nearest_idx, 'comments']
        
        # if there's already a comment at the time clicked, give an option to replace
        if existing and str(existing).strip():
            confirm = QMessageBox.question(
                self,
                "Overwrite Comment?",
                f"A comment already exists at {df.at[nearest_idx, 'time']:.2f}s:\n\n\"{existing}\"\n\nReplace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
            if confirm == QMessageBox.StandardButton.No:
                return

        # Create the dialog popup
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Add Comment @ {comment_time:.2f}s")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Add Comment:"))
        text = QTextEdit()
        layout.addWidget(text)

        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        layout.addWidget(save_button)
        layout.addWidget(cancel_button)
        save_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # if the text was just spaces/an empty comment, then don't create a comment
        text = text.toPlainText().strip()
        if not text:
            return
    
        # create comment
        df.at[nearest_idx, 'comments'] = text
        marker = self.comments.get(comment_time)
        if marker:
            marker.set_text(text)
        else:
            new_marker = CommentMarker(comment_time, text, self)
            self.comments[comment_time] = new_marker
        
        self.comment_preview_enabled = False
        self.comment_preview.setVisible(False)
        return

# TODO need to implement when understand format of comment
    # def delete_comment(self, time: float) -> None:
    #     df = self.epgdata.dfs[self.file]
    #     nearest_idx = (df['time'] - time).abs().idxmin()
    #     comment_time = df.at[nearest_idx, 'time']
    #     df.at[nearest_idx, 'comments'] = None

    #     marker = self.comments.pop(comment_time, None)
    #     if marker:
    #         marker.remove()
    #     return
    
    # def update_comment(self, )

    def downsample_visible(
        self, x_range: tuple[float, float] = None, max_points=4000, method = 'peak'
    ) -> tuple[NDArray, NDArray]:
        """
        Downsamples the data displayed in x_range to max_points using
        the specifed downsampling method.
        Modifies self.xy_data in-place.

        Inputs:
            x_range: a (x_min, x_max) tuple of the range of the data to be displayed
            max_points: the number of points (i.e., bins) to downsample to.
            method: `subsample`, `mean`, or `peak`, which downsampling method to use.

        Output:
           None

        NOTE: 
            `subsample` samples the first point of each bin (fastest)
            `mean` averages each bin
            `peak` returns the min and max point of each bin (slowest, best looking)
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
            self.plot_item.removeItem(label_area.transition_line)
            self.plot_item.removeItem(label_area.label_text)
            self.plot_item.removeItem(label_area.label_background)
            self.plot_item.removeItem(label_area.duration_text)
            self.plot_item.removeItem(label_area.duration_background)
            if self.enable_debug:
                self.plot_item.removeItem(label_area.label_debug_box)
                self.plot_item.removeItem(label_area.duration_debug_box)
            
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

        # NOTE: Each LabelArea has one transition line, but we need an additional ending
        # line so that the final LabelArea doesn't end without a transition line. We chose
        # to do this by adding a zero-width, invisible LabelArea, called the "end area", to 
        # the end of the labels, so that just the transition line appears.
        time, dur, _ = durations[-1]
        end_start_time = time + dur

        label_area = LabelArea(end_start_time, 0, 'END AREA', self)
        label_area.label_text.setVisible(False)
        label_area.label_background.setVisible(False)
        label_area.duration_text.setVisible(False)
        label_area.duration_background.setVisible(False)
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

    def composite_on_white(self, color: QColor) -> QColor:
        """
        Helps function to get the RGB value (no alpha) of 
        an RGBA color displayed on a white background.
        """
        r, g, b, a = color.getRgb()
        a = a / 255

        new_r = round(r * a + 255 * (1 - a))
        new_g = round(g * a + 255 * (1 - a))
        new_b = round(b * a + 255 * (1- a))
        return QColor(new_r, new_g, new_b)
    
    # def darker_hsl(self, color: QColor, amount: float) -> QColor:
    #     """
    #     Returns a darker color by reducing HSL lightness by `amount`.
    #     """
    #     h, s, l, a = color.getHsl()
    #     new_l = max(0, l - int(amount * 255))
    #     new_color = QColor.fromHsl(h, s, new_l, a)
    #     return new_color
        
    def get_closest_transition(self, x: float) -> tuple[int, float]:
        """
        Returns the index and pixel distance from a given x coordinate 
        to the closest transition line.

        Inputs: 
            x: the queried viewbox x-coordinate

        Outputs:
            index, x_distance: the index of and distance in pixels to the closest transition line
        """  
        if not self.labels:
            return float('inf')  # no labels present
        
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
        Returns the baseline and pixel distance from a given y coordinate 
        to the baseline.
        """
        if self.baseline is None:
            return float('inf')
        zero_point = self.viewbox_to_window(QPointF(0,0)).y()
        viewbox_distance = abs(y - self.baseline.value())
        return self.baseline, zero_point - self.viewbox_to_window(QPointF(0, viewbox_distance)).y()
    
    def get_closest_label_area(self, x: float) -> LabelArea:
        if not self.labels:
            return None
        
        # don't include the last label
        visible_labels = [label for label in self.labels if not label == self.labels[-1]]

        if x < visible_labels[0].start_time or x > (visible_labels[-1].start_time + visible_labels[-1].duration):
            return None  # outside the labels
        label_ends = np.array([label.start_time + label.duration for label in visible_labels])
        idx = np.searchsorted(label_ends, x)  # idk why this works
        if idx >= len(visible_labels):
            return visible_labels[-1]
        return visible_labels[idx]

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

        
        self.baseline.setPos(y)
        self.baseline.setVisible(True)

        self.baseline_preview_enabled = False
        self.baseline_preview.setVisible(False)


    def add_drop_transitions(self):
        return

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_R:
            self.reset_view()  
        if event.key() == Qt.Key.Key_B:
            if self.baseline_preview_enabled:
                # Turn it off
                self.baseline_preview_enabled = False
                self.baseline_preview.setVisible(False)
            else:
                # disable simultaneous "c" click 
                self.comment_preview_enabled = False
                self.comment_preview.setVisible(False)
                # prepare to set baseline
                self.baseline.setVisible(False)
                self.baseline_preview_enabled = True
                self.baseline_preview.setVisible(True)
                y_pos = self.viewbox.mapSceneToView(self.mapToScene(self.mapFromGlobal(QCursor.pos()))).y()
                self.baseline_preview.setPos(y_pos)
            self.selection.deselect_all()
            self.selection.unhighlight_item(self.selection.hovered_item)
        if event.key() == Qt.Key.Key_C:
            if self.comment_preview_enabled:
                # Turn it off
                self.comment_preview_enabled = False
                self.comment_preview.setVisible(False)
            else:
                # disable simultaneous "b" click 
                self.baseline_preview_enabled = False
                self.baseline_preview.setVisible(False)
                # prepare to add new comment
                self.comment_preview_enabled = True
                self.comment_preview.setVisible(True)
                x_pos = self.viewbox.mapSceneToView(self.mapToScene(self.mapFromGlobal(QCursor.pos()))).x()
                self.comment_preview.setPos(x_pos)
            self.selection.deselect_all()
            self.selection.unhighlight_item(self.selection.hovered_item)

        self.selection.key_press_event(event)
        self.viewbox.update()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        return
        if event.key() == Qt.Key.Key_Shift:
            self.vertical_mode = False
        elif event.key() == Qt.Key.Key_Control:
            self.zoom_mode = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)

        # TODO: edit for when have edit mode functionality
        # For testing baseline preview

        point = self.window_to_viewbox(event.position())
        x, y = point.x(), point.y()

        if event.button() == Qt.MouseButton.LeftButton:
            if self.baseline_preview_enabled:
                self.set_baseline(event)
            elif self.comment_preview_enabled:
                self.add_comment(event)
            else:
                self.selection.mouse_press_event(event)
        # elif event.button() == Qt.MouseButton.RightButton:

        # (x_min, x_max), (y_min, y_max) = self.viewbox.viewRange()
        # if not (x_min <= x <= x_max and y_min <= y <= y_max):
        #     print('click outside of box')
        #     for item in self.selection.selected_items:
        #         self.selection.deselect(item)
        #     self.scene().update()
        #     return
        
        # if event.button() == Qt.MouseButton.LeftButton:
        #     # if self.baseline_preview_enabled:
        #     #     self.set_baseline(event)
        #     #     self.baseline_preview_enabled = False
        #     #     self.baseline_preview.setVisible(False)

        #     # elif isinstance(self.hovered_item, InfiniteLine):
        #     #     # if already part of selection, do not reset selection
        #     #     if self.selection.is_selected(self.hovered_item):  
        #     #         self.moving_mode = True
        #     #         self.setCursor(Qt.CursorShape.ClosedHandCursor)
        #     #     else:
        #     #         self.moving_mode = True
        #     #         self.selection.deselect(self.selection)
        #     #         self.selection = self.hovered_item
        #     #         self.setCursor(Qt.CursorShape.ClosedHandCursor)

        #     if isinstance(self.hovered_item, LabelArea):

        #         # no shift: select new label area
        #         if event.modifiers() == Qt.KeyboardModifier.NoModifier:
        #             self.deselect(self.selection)
        #             self.selection = self.hovered_item
        #         # shift held: create/update list 
        #         elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
        #             if self.selection == None:
        #                 self.selection = self.hovered_item
        #             elif not isinstance(self.selection, list):
        #                 if self.hovered_item != self.selection:
        #                     self.selection = [self.selection]
        #                     self.selection.append(self.hovered_item)
        #             else:
        #                 if self.hovered_item not in self.selection:
        #                     self.selection.append(self.hovered_item)
                    
        #         self.select_label_area(self.hovered_item)
        #         print(self.selection)
        
    
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

        self.selection.mouse_release_event(event)

        if self.moving_mode:
            # if transition line was released, update data transition line
            if isinstance(self.selected_item, InfiniteLine) and self.selected_item is not self.baseline:
                transitions = [(label_area.start_time, label_area.label) for label_area in self.labels]
                self.epgdata.set_transitions(self.file, transitions, self.transition_mode)
        return
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_transitions(event, "release")

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        return
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_labels(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super().mouseMoveEvent(event)

        point = self.window_to_viewbox(event.position())
        x, y = point.x(), point.y()

        (x_min, x_max), (y_min, y_max) = self.viewbox.viewRange()

        if self.baseline_preview_enabled:
            if y_min <= y <= y_max:
                self.baseline_preview.setPos(y)
                self.baseline_preview.setVisible(True)
            else:
                self.baseline_preview.setVisible(False)
        elif self.comment_preview_enabled:
            if x_min <= x <= x_max:
                self.comment_preview.setPos(x)
                self.comment_preview.setVisible(True)
            else:
                self.comment_preview.setVisible(False)
        else:
            self.selection.mouse_move_event(event)

        return

        self.handle_transitions(event, "move")

    

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
    #print("Data Loaded")
    
    window = DataWindow(epgdata)
    window.plot_recording(window.epgdata.current_file, 'pre')
    window.plot_transitions(window.epgdata.current_file)

    window.showMaximized()

    tracker = GlobalMouseTracker(window)
    app.installEventFilter(tracker)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
