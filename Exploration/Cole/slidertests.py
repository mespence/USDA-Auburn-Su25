import sys
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QDoubleSpinBox
from PyQt6.QtCore import Qt

startVal = 50

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyQt QSlider')
        self.setMinimumWidth(200)

        # create a grid layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.setValue(startVal)
        #slider.setSingleStep(1) #not doing anything
        #slider.setPageStep(1) #also not doing anything
        #slider.setTickPosition(QSlider.TickPosition.TicksAbove)

        self.slider.valueChanged.connect(self.update)

        self.result_label = QLabel('', self)
        self.result_label.setText(f'Current Value: {startVal}')
        
        self.numBox = QDoubleSpinBox(self)
        self.numBox.setValue(startVal)
        self.numBox.valueChanged.connect(self.update)
        
        layout.addWidget(self.slider)
        layout.addWidget(self.result_label)
        layout.addWidget(self.numBox)

        # show the window
        self.show()

    def update(self, value):
        self.result_label.setText(f'Current Value: {value}')
        self.numBox.setValue(value)
        sliderVal = int(value)
        self.slider.setValue(sliderVal)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())