import numpy as np
from math import isclose

from pyqtgraph import *
from PyQt6.QtWidgets import QApplication    # DEBUG ONLY TODO remove imports for testing
from PyQt6.QtGui import QKeyEvent, QWheelEvent, QMouseEvent 
from PyQt6.QtCore import Qt, QPointF, QTimer

from EPGData import EPGData
from Settings import Settings


class PanZoomViewBox(ViewBox):
    """
    Helper class to override the default ViewBox behavior
    of scroll -> zoom and drag -> pan.
    """
    def wheelEvent(self, event: QWheelEvent, axis=None):
        datawindow: DataWindow = self.parentItem().getViewWidget()
        delta = event.angleDelta().y()

        if datawindow.zoom_mode:
            zoom_factor = 1.001 ** delta
            center = self.mapToView(event.position())
            if datawindow.vertical_mode:
                self.scaleBy((1, 1 / zoom_factor), center)
            else:
                self.scaleBy((1 / zoom_factor, 1), center)     
        else:
            (x_min, x_max), (y_min, y_max) = self.viewRange()
            if datawindow.vertical_mode:
                v_zoom_factor = 5e-4
                self.translateBy(y = delta * v_zoom_factor * (y_max - y_min))   
            else:
                h_zoom_factor = 2e-4
                self.translateBy(x = delta * h_zoom_factor * (x_max - x_min)) 
    
        event.accept()
        datawindow.update_plot()
        
    def mouseDragEvent(self, event, axis = None):
        # Disable all mouse drag panning/zooming
        event.ignore()
  


class DataWindow(PlotWidget):
    def __init__(self, epgdata: EPGData):
        super().__init__(plotItem = PlotItem(viewBox = PanZoomViewBox()))  
        self.epgdata: EPGData = epgdata
        self.file: str = None
        self.prepost: str = 'pre'
        self.plot_item: PlotItem = self.getPlotItem()  # the plotting canvas (axes, grid, data, etc.)
        self.xy_data: list[np.ndarray] = [np.array([]), np.array([])]  # x and y data to be actually rendered
        self.curve: PlotDataItem = PlotDataItem(antialias = False)
        self.viewbox : ViewBox = self.plot_item.getViewBox()  # the plotting area (no axes, etc.)
        self.vertical_mode: bool = False  # whether scroll/zoom actions are vertical
        self.zoom_mode: bool = False  # whether mouse wheel controls zoom
        self.cursor_mode: str = 'normal'  # cursor state, e.g. normal, baseline selection
        self.compression: float = 0
        self.compression_text: TextItem = TextItem()
        self.zoom_level: float = 1
        self.zoom_text: TextItem = TextItem()

        self.initUI()
        

    def initUI(self):
        self.chart_width: int = 400
        self.chart_height: int = 400
        self.setGeometry(0, 0, self.chart_width, self.chart_height)
        #self.setContentsMargins(50, 500, 50, 50)

        self.setBackground('white')
        self.setTitle('<b>SCIDO Waveform Editor</b>', color='black', size='12pt')

        self.viewbox.setBorder(mkPen('black', width=3))

        self.plot_item.addItem(self.curve)
        self.plot_item.setLabel('bottom', '<b>Time [s]</b>', color='black')
        self.plot_item.setLabel('left', '<b>Voltage [V]</b>', color='black')
        self.plot_item.showGrid(x=Settings.show_grid, y=Settings.show_grid)
        self.plot_item.layout.setContentsMargins(30, 30, 30, 20)


        # placeholder sine wave
        self.xy_data[0] = np.linspace(0, 1, 10000)
        self.xy_data[1] = np.sin(2 * np.pi * self.xy_data[0])  
        self.curve.setData(self.xy_data[0], self.xy_data[1], pen=mkPen(color='b', width=2))
        self.curve.setClipToView(True)

        QTimer.singleShot(0, self.deferred_init)

        
    def deferred_init(self):
        """
        Initalizes the items that need to be initalized after 
        everything has been rendered to the screen.
        """
        self.compression = 0
        self.compression_text = TextItem(
            text=f'Compression: {self.compression: .1f}', color='black', anchor=(0, 0)
        )
        self.compression_text.setPos(QPointF(80,15))
        self.scene().addItem(self.compression_text)

        self.zoom_level = 1
        self.zoom_text = TextItem(
            text=f'Zoom: {self.zoom_level * 100}%', color='black', anchor=(0, 0)
        )
        self.zoom_text.setPos(QPointF(80,30))
        self.scene().addItem(self.zoom_text)

        # further defer init. compression and zoom calculations
        # until the window is actually rendered to the screen
        QTimer.singleShot(0, lambda: self.update_zoom()) 
        QTimer.singleShot(0, lambda: self.update_compression()) 


    def plot_recording(self, file: str, prepost: str = 'pre') -> None:
        """
        plot_recording creates an ndarray for the time series given by file
        and updates the graph to show it.

        Inputs:
            file: a string containing the key of the recording
            prepost: a string containing either pre or post
                 to specify which recording is desired.

        Outputs:
            None
        """
        self.file = file
        df = self.epgdata.get_recording(self.file, prepost)
        time = df['time'].values
        volts = df[prepost + self.epgdata.prepost_suffix].values

        self.xy_data[0] = time
        self.xy_data[1] = volts
        self.downsample_visible()
        self.curve.setData(self.xy_data[0], self.xy_data[1])

        self.viewbox.setRange(xRange=(min(time), max(time)), yRange=(min(volts), max(volts)), padding=0)
        self.update_plot()

    def downsample_visible(self, x_range: tuple[float, float] = None, max_points = 3000) -> tuple[np.ndarray, np.ndarray]:
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
        x = df['time'].values
        y = df[self.prepost + self.epgdata.prepost_suffix].values

        # Filter to x_range if provided
        if x_range is not None:
            mask = (x >= x_range[0]) & (x <= x_range[1])  # boolean mask for x_range
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
        
        start_idx = stride // 2  # Choose a representative x (near center) for each window
        x_win = x[start_idx : start_idx + num_windows * stride : stride]
        x_out = np.repeat(x_win, 2)  # repeated for (x, y_min), (x, y_max)

        y_reshaped = y[:num_windows * stride].reshape(num_windows, stride)
        y_out = np.empty(num_windows * 2)
        y_out[::2] = y_reshaped.max(axis=1)
        y_out[1::2] = y_reshaped.min(axis=1)

        self.xy_data[0] = x_out
        self.xy_data[1] = y_out
    


    def resetView(self) -> None:
        """
        Resets the viewing window back to default 
        settings (default zoom, scrolling, etc.)

        Inputs:
            None

        Returns:
            None
        """
        df = self.epgdata.get_recording(self.file, self.prepost)
        time = df['time'].values
        volts = df[self.prepost + self.epgdata.prepost_suffix].values

        self.viewbox.setRange(xRange=(min(time), max(time)), yRange=(min(volts), max(volts)), padding=0)
        self.update_plot()
        self.update_compression()
        self.update_zoom()

    def update_plot(self):
        """
        Updates the displayed data and compression/zoom indicators.
        """
        (x_min, x_max), (y_min, y_max) = self.viewbox.viewRange()
        self.downsample_visible(x_range=(x_min, x_max))
        self.curve.setData(self.xy_data[0], self.xy_data[1])

        self.update_compression()
        self.update_zoom()


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
            return float('inf')  # Avoid division by zero

        pix_per_second = plot_width / time_span
        second_per_pix = 1 / (pix_per_second)

        # Convert to compression based on WinDaq
        self.compression = second_per_pix * 125
        self.compression_text.setText(f'Compression Level: {self.compression :.1f}')

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
            return float('inf')  # Avoid division by zero
        
        file_length_sec = self.epgdata.dfs[self.file]['time'].iloc[-1]
        default_pix_per_second = plot_width / file_length_sec

        self.zoom_level = pix_per_second / default_pix_per_second

        # leave off decimal if zoom level is int
        if isclose(self.zoom_level, round(self.zoom_level), abs_tol = 1e-9):
            self.zoom_text.setText(f'Zoom: {self.zoom_level * 100: .0f}%')
        else:
            self.zoom_text.setText(f'Zoom: {self.zoom_level * 100: .1f}%')


    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Shift:
            # holding shift allows for vertical zoom / scroll
            self.vertical_mode = True
        elif event.key() == Qt.Key.Key_Control:
            # holding control allows for zoom, otherwise
            # scrolling scrolls the plot
            self.zoom_mode = True
        elif event.key() == Qt.Key.Key_R:
            # r resets zoom
            self.resetView()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Shift:
            self.vertical_mode = False
        elif event.key() == Qt.Key.Key_Control:
            self.zoom_mode = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.cursor_state == 'normal':
                self.handle_transitions(event, 'press')
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
        self.handle_transitions(event, 'move')

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        return
        if event.button() == Qt.MouseButton.LeftButton:
            self.handle_transitions(event, 'release')

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        wheelEvent is called automatically whenever the scroll
        wheel is engaged over the chart. We use it to control
        horizontal and vertical scrolling along with zoom.
        """
        self.viewbox.wheelEvent(event)
        #self.zoom_scroll(event)

    def zoom_scroll(self, event: QWheelEvent):
        if self.zoom_mode:
            if self.vertical_mode:
                self.viewbox.setMouseEnabled(x = False, y = True)
            else: 
                self.viewbox.setMouseEnabled(x = True, y = False)
        else:
            delta = event.angleDelta().y()
            #center = self.viewbox.mapToView(event.pos()
           # if self.vertical_mode:
                
            


    def handle_transitions(self):
        return

    def handle_labels(self):
        return

    def set_baseline(self):
        return

    def add_drop_transitions(self):
        return


def main():
    print('Testing new DataWindow class')
    Settings()
    app = QApplication([])

    epgdata = EPGData()
    epgdata.load_data('test_recording.csv')
    #epgdata.load_data(r'C:\EPG-Project\Summer\Code\summer-code\smooth_18mil.csv')
    print('Data Loaded')

    window = DataWindow(epgdata)
    window.plot_recording(window.epgdata.current_file)
    window.showMaximized()
    
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()