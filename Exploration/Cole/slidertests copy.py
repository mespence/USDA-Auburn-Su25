import sys
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QLabel, QFormLayout
from PyQt6.QtCore import Qt
from anotherslider import DoubleSlider

startVal = 50.8

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyQt QSlider')
        self.setMinimumWidth(200)

        # create a grid layout
        layout = QFormLayout()
        self.setLayout(layout)

        slider = DoubleSlider(Qt.Orientation.Horizontal, decimals=2)
        slider.setRange(0, 200)
        slider.setValue(startVal)
        #slider.setSingleStep(1) #not doing anything
        #slider.setPageStep(1) #also not doing anything
        #slider.setTickPosition(QSlider.TickPosition.TicksAbove)

        slider.valueChanged.connect(self.update)

        self.result_label = QLabel('', self)
        self.result_label.setText(f'Current Value: {startVal}')

        layout.addRow(slider)
        layout.addRow(self.result_label)

        # show the window
        self.show()

    def update(self, value):
        self.result_label.setText(f'Current Value: {value}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())