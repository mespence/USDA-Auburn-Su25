import sys
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QDoubleSpinBox, QLineEdit, QComboBox, QDial, QProgressBar, QPushButton, QRadioButton, QButtonGroup
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator

startVal = 50
max = 100
min = 0
decimals = 2

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

        self.setWindowTitle('PyQt Franken-Widget')
        self.setMinimumWidth(800)
        self.setMinimumHeight(200)

        # create a grid layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.slider = CDoubleSlider(Qt.Orientation.Horizontal, self, decimals = decimals)
        self.slider.setRange(min, max)
        self.slider.setValue(startVal)

        self.slider.doubleValueChanged.connect(self.update)

        self.result_label = QLabel('', self)
        self.result_label.setText(f'Current Value: {startVal}')
        
        self.numBox = QDoubleSpinBox(self, decimals = decimals, maximum = max, minimum = min)
        self.numBox.setValue(startVal)
        self.numBox.valueChanged.connect(self.update)

        self.lineEdit = CLineEdit(str(startVal), self)
        lineValidator = QDoubleValidator(float(min),float(max), decimals)
        lineValidator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.lineEdit.setValidator(lineValidator) # this is always gonna be checking, unless i want to set and reset it a ton of times which i don't want to do unless necessary
        self.lineEdit.returnPressed.connect(self.lineEditEntered)

        self.group1 = QButtonGroup()
        self.radio_button1 = QRadioButton("Add")
        self.radio_button1.setChecked(True)
        self.radio_button2 = QRadioButton("Subtract")

        self.comboBox = QComboBox(self)
        self.comboBox.addItems(['1', '5', '10', '20', '50'])

        self.pushButton = QPushButton(self)
        self.pushButton.pressed.connect(self.buttonUpdate)

        self.dial = QDial(self)
        self.dial.setRange(min,max)
        self.dial.setValue(startVal)
        self.dial.valueChanged.connect(self.update)
        
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0,1000)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True) #not 100% sure what this is doing, tbh
        
        layout.addWidget(self.slider)
        layout.addWidget(self.result_label)
        layout.addWidget(self.numBox)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.comboBox)
        layout.addWidget(self.radio_button1)
        layout.addWidget(self.radio_button2)
        layout.addWidget(self.pushButton)
        layout.addWidget(self.dial)
        layout.addWidget(self.progressBar)

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
        dialVal = int(value)
        if self.dial.value() != dialVal:
            self.dial.blockSignals(True)
            self.dial.setValue(dialVal)
            self.dial.blockSignals(False)
        if self.progressBar.value() == 1000:
            self.progressBar.setValue(0)
        self.progressBar.setValue(self.progressBar.value()+1)

    def lineEditEntered(self):
        text = self.lineEdit.text()
        # print(self.lineEdit.validator().validate(text, 0)) NOTE: if there are errors at the other note, this is a good line to test with
        self.update(text)
    
    def buttonUpdate(self):
        addVal = float(self.comboBox.currentText())
        if self.radio_button1.isChecked():
            newVal = self.numBox.value() + addVal # NOTE: this is also a good place to check if there are weird errors. I made it dependent on numbox, but any "source of truth" should work
        else:
            newVal = self.numBox.value() - addVal
        if newVal > max:
            newVal = max
        if newVal < min:
            newVal = min
        self.update(newVal)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())