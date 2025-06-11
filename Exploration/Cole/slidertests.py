import sys
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QDoubleSpinBox
from PyQt6.QtCore import Qt, pyqtSignal

startVal = 50

class DoubleSlider(QSlider):

    # create our our signal that we can connect to if necessary
    doubleValueChanged = pyqtSignal(float)

    def __init__(self, *args, decimals=3, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)
    
    def setRange(self, min, max):
        return super(DoubleSlider, self).setRange(min * self._multi, max * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyQt QSlider')
        self.setMinimumWidth(200)

        # create a grid layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.slider = DoubleSlider(Qt.Orientation.Horizontal, self, decimals = 2)
        self.slider.setRange(0, 100)
        self.slider.setValue(startVal)
        #slider.setSingleStep(1) #not doing anything
        #slider.setPageStep(1) #also not doing anything
        #slider.setTickPosition(QSlider.TickPosition.TicksAbove)

        self.slider.doubleValueChanged.connect(self.update)

        self.result_label = QLabel('', self)
        self.result_label.setText(f'Current Value: {startVal}')
        
        self.numBox = QDoubleSpinBox(self, decimals = 4, maximum = 100, minimum = 0)
        self.numBox.setValue(startVal)
        self.numBox.valueChanged.connect(self.update)
        
        layout.addWidget(self.slider)
        layout.addWidget(self.result_label)
        layout.addWidget(self.numBox)

        # show the window
        self.show()

    def update(self, value):
        # Block signals to prevent recursion
        self.result_label.setText(f'Current Value: {value}')
        if self.numBox.value() != value:
            self.numBox.blockSignals(True)
            self.numBox.setValue(value)
            self.numBox.blockSignals(False)
        sliderVal = int(value)
        if self.slider.value() != sliderVal:
            self.slider.blockSignals(True)
            self.slider.setValue(sliderVal)
            self.slider.blockSignals(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())