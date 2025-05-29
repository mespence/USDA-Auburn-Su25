from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QGridLayout
import sys, math, numpy as np, pandas as pd
import PyQt6.QtCharts as qtc
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QPen

import pyqtgraph as pg

MAXIMUM_DATA_RESOLUTION = 4000
# NUM_POINTS = int(1e7)
FILE = "smooth_18mil.csv"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.width = 400
        self.height = 400
        self.setGeometry(0, 0, self.width, self.height)
        self.setWindowTitle("Jomo")

        self.datawindow = pg.PlotWidget()

        # self.plot_widget = pg.PlotWidget()
        # self.datawindow = qtc.QChartView()
        # self.lineseries = qtc.QLineSeries()  # Create a series to hold our data
        # self.scatterseries = qtc.QScatterSeries()

        self.data = pd.read_csv(FILE)
        print("Data Loaded")
        #self.data = self.downsample_data(self.data, 0, len(self.data)-1)

        x = self.data.iloc[:,0].to_numpy()
        y = self.data.iloc[:,1].to_numpy()

        self.curve: pg.PlotItem = self.datawindow.plot(x,y, downsample = 4500, autoDownsample = True) # initial plot
        self.datawindow.plotItem.vb.sigXRangeChanged.connect(
            lambda vb: self.update_view_range(vb.viewRange()[0])
        )
        #self.curve.setDownsampling('peak', ds=1000, auto=True)
        print("Data Plotted")

        # self.chart = qtc.QChart()
        # self.chart.setPlotAreaBackgroundVisible(True)
        # self.chart.setPlotAreaBackgroundBrush(QBrush(Qt.GlobalColor.white))
        # self.chart.setPlotAreaBackgroundPen(QPen(Qt.GlobalColor.black, 3))
        # self.chart.setTitle("<b>Qt w/ many data points</b>")
        # self.chart.legend().hide()
        # self.chart.addSeries(self.lineseries)
        # self.chart.addSeries(self.scatterseries)

        centralWidget = QWidget()
        layout = QGridLayout()
        layout.addWidget(self.datawindow, 1, 0, 1, 0)
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def downsample_visible(self, x, y, max_points=2000, x_range=None):
        # x data, y data, (xmin, xmax) tuple
        x = np.asarray(x)
        y = np.asarray(y)

        # Filter to x_range if provided
        if x_range is not None:
            mask = (x >= x_range[0]) & (x <= x_range[1])
            x = x[mask]
            y = y[mask]

        n = len(x)
        if n <= max_points or n < 2:
            return x, y

        # Peak decimation
        ds = max(1, n // (max_points // 2))  # each window gives 2 points
        n_windows = n // ds
        stx = ds // 2

        # Choose a representative x (near center) for each window
        x_win = x[stx : stx + n_windows * ds : ds]
        x_out = np.repeat(x_win, 2)

        y_reshaped = y[:n_windows * ds].reshape(n_windows, ds)
        y_out = np.empty(n_windows * 2)
        y_out[::2] = y_reshaped.max(axis=1)
        y_out[1::2] = y_reshaped.min(axis=1)


        return x_out, y_out
    def update_view_range(self, view_range):
        xmin, xmax = view_range
        x_ds, y_ds = self.downsample_visible(self.data.iloc[:,0].to_numpy(), self.data.iloc[:,1].to_numpy(), x_range=(xmin, xmax))
        self.curve.setData(x_ds, y_ds)
    

    def downsample_data(
        self,
        data: pd.DataFrame,
        start_index: int,
        end_index: int,
        data_resolution=MAXIMUM_DATA_RESOLUTION,
    ) -> pd.DataFrame:

        x = data.iloc[:,0].to_numpy()
        y = data.iloc[:,1].to_numpy()

        data_size = end_index - start_index

        if data_size <= data_resolution:
            print("No downsampling needed.")
            return data

        stride = math.ceil(data_size / data_resolution)
        num_bins = data_size // stride

        print(num_bins)
        x1 = np.empty((num_bins, 2))
        start_x = start_index + stride // 2  # approx centered start of x-values

        x1[:] = x[start_x : start_x + num_bins * stride : stride, np.newaxis]
        x = x1.reshape(num_bins * 2)


        y1 = np.empty((num_bins, 2))
        y2 = y[: num_bins * stride].reshape((num_bins, stride))
        y1[:, 0] = y2.max(axis=1)
        y1[:, 1] = y2.min(axis=1)
        y = y1.reshape(num_bins * 2)

        
        return pd.DataFrame(zip(x,y))


def main():
    app = QApplication([])
    window = MainWindow()
    # window.showNormal()
    window.showMaximized()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
