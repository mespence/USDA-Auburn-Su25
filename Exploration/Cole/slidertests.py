import sys
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QDoubleSpinBox, QLineEdit, QComboBox, QDial, QProgressBar, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator

startVal = 50
max = 100
min = 0

class CDoubleSlider(QSlider):

    # create our our signal that we can connect to if necessary
    doubleValueChanged = pyqtSignal(float)

    def __init__(self, *args, decimals=3, **kargs):
        super(CDoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(CDoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(CDoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(CDoubleSlider, self).setMinimum(value * self._multi)
    
    def setRange(self, min, max):
        return super(CDoubleSlider, self).setRange(min * self._multi, max * self._multi)

    def setMaximum(self, value):
        return super(CDoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(CDoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(CDoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(CDoubleSlider, self).setValue(int(value * self._multi))

class CLineEdit(QLineEdit):
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Always call your handler, regardless of validator state
            self.parent().lineEditEntered()
        super().keyPressEvent(event)

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyQt QSlider')
        self.setMinimumWidth(800)
        self.setMinimumHeight(200)

        # create a grid layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.slider = CDoubleSlider(Qt.Orientation.Horizontal, self, decimals = 2)
        self.slider.setRange(min, max)
        self.slider.setValue(startVal)
        #slider.setSingleStep(1) #not doing anything
        #slider.setPageStep(1) #also not doing anything
        #slider.setTickPosition(QSlider.TickPosition.TicksAbove)

        self.slider.doubleValueChanged.connect(self.update)

        self.result_label = QLabel('', self)
        self.result_label.setText(f'Current Value: {startVal}')
        
        self.numBox = QDoubleSpinBox(self, decimals = 2, maximum = max, minimum = min)
        self.numBox.setValue(startVal)
        self.numBox.valueChanged.connect(self.update)

        self.lineEdit = CLineEdit(str(startVal), self)
        lineValidator = QDoubleValidator(float(min),float(max), 2)
        lineValidator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.lineEdit.setValidator(lineValidator) # this is always gonna validate, unless i want to set and reset it a ton of times which i don't want to do unless necessary
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
        if value > max or value < min:
            value = self.numBox.value() # this should only trigger with lineedit so its fine if this cues off another widget
            # i could also do it by having it store its previous value for all widgets but I think that would slow things down
            # NOTE: if there are mysterious errors when messing with values, blame this line first
        
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
        print(self.lineEdit.validator().validate(text, 0))
        self.update(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())