import sys
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCharts import *
from math import sin

from DataWindow import DataWindow

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.initUI()
	
	def initUI(self):
		self.setWindowTitle("EPG Labeler 5000")

		# Make widgets
		datawindow = DataWindow()
		button = QPushButton("Test Button")
		
		# Arrange the widgets
		centralWidget = QWidget()
		layout = QGridLayout()
		layout.addWidget(datawindow, 0, 0, 1, 0)
		layout.addWidget(button, 1, 1)
		centralWidget.setLayout(layout)
		self.setCentralWidget(centralWidget)

def main():
	app = QApplication([])
	window = MainWindow()
	window.show()
	sys.exit(app.exec())

if __name__ == '__main__':
	main()
