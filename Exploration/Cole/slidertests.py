import sys
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QDoubleSpinBox, QLineEdit, QComboBox, QDial, QProgressBar, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator

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
        self.setMinimumWidth(800)
        self.setMinimumHeight(200)

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

        self.lineEdit = QLineEdit(str(startVal), self)
        lineValidator = QDoubleValidator(0.0,100.0, 4)
        lineValidator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.lineEdit.setValidator(lineValidator)
        self.lineEdit.returnPressed.connect(self.lineEditEntered)

        self.comboBox = QComboBox(self)
        self.dial = QDial(self)
        self.progressBar = QProgressBar(self)
        self.pushButton = QPushButton(self)
        
        layout.addWidget(self.slider)
        layout.addWidget(self.result_label)
        layout.addWidget(self.numBox)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.comboBox)
        layout.addWidget(self.dial)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.pushButton)

        # show the window
        self.show()

    def update(self, rawVal):
        value = float(rawVal)
        # Block signals to prevent recursion
        self.result_label.setText(f'Current Value: {value}')
        if self.numBox.value() != value:
            self.numBox.blockSignals(True)
            self.numBox.setValue(value)
            self.numBox.blockSignals(False)
        if self.slider.value() != value:
            self.slider.blockSignals(True)
            self.slider.setValue(value)
            self.slider.blockSignals(False)
        if self.lineEdit.text() != value:
            self.lineEdit.blockSignals(True)
            self.lineEdit.setText(str(value))
            self.lineEdit.blockSignals(False)

    def lineEditEntered(self):
        text = self.lineEdit.text()
        self.update(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())