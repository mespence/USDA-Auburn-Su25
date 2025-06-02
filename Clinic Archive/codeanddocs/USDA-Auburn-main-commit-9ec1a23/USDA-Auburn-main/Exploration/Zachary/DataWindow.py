from PyQt6.QtCharts import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from math import sin

class DataWindow(QChartView):
	def __init__(self, parent = None):
		super().__init__()
		self.initUI()
	
	def initUI(self):
		self.setGeometry(0, 0, 400, 400)

		# Create a series to hold our data
		self.series = QLineSeries()
		for i in range(10000):
			self.series.append(QPointF(i, sin(2*3.14 * i/ 10000)))

		# Vertical line, we can move it around?
		self.vline = QLineSeries()
		self.vline.append(QPointF(1, 0))
		self.vline.append(QPointF(1, 1))
		self.track = False

		# Create a chart to put our data in
		self.chart = QChart()
		self.chart.addSeries(self.series)
		self.chart.addSeries(self.vline)

		# Axes
		self.x_axis = QValueAxis()
		self.x_axis.setTitleText("Time")
		self.chart.addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
		self.series.attachAxis(self.x_axis)
		self.vline.attachAxis(self.x_axis)
		
		# Set the chart of this QChartView to be our chart
		self.setChart(self.chart)
		self.setRenderHint(QPainter.RenderHint.Antialiasing)
	
	def mousePressEvent(self, event):
		if event.button() == Qt.MouseButton.LeftButton:
			self.track = True
	
	def mouseMoveEvent(self, event):
		if not self.track:
			return
		x = event.pos().x()
		y = event.pos().y()
		scene_coords = self.mapToScene(x, y)
		chart_coords = self.chart.mapFromScene(scene_coords)
		val_coords = self.chart.mapToValue(chart_coords)
		self.vline.clear()
		self.vline.append(QPointF(val_coords.x(), 0))
		self.vline.append(QPointF(val_coords.x(), 10))
		self.update()

	def mouseReleaseEvent(self, event):
		self.track = False


