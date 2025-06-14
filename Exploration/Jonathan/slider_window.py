import sys 
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QGridLayout, QGroupBox, QComboBox, QLineEdit,
    QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QSlider, QLabel, QSizePolicy
)


class SliderWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EPG Control Sliders")
        self.resize(400, 300)

        layout = QVBoxLayout()
        grid = QGridLayout()


        grid.addWidget(self.createExampleGroup(), 0, 0)
        grid.addWidget(self.createExampleGroup(), 1, 0)
        grid.addWidget(self.createExampleGroup(), 0, 1)
        grid.addWidget(self.createExampleGroup(), 1, 1)
        self.setLayout(grid)

    def add_slider(
            grid: QGridLayout, row: int,
            min_val: float, max_val: float,
            label_text: str, unit: str = None
    
        ) -> tuple[QSlider, QLineEdit]:
        grid.addWidget(QLabel(label_text), row, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setFixedWidth(150)
        slider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        grid.addWidget(slider, row, 1)

        entry = QLineEdit("0")
        entry.setFixedWidth(50)
        grid.addWidget(entry, row, 2)

        if unit:
            grid.addWidget(QLabel(unit), row, 3)
        return slider, entry


    def createExampleGroup(self):
        groupBox = QGroupBox("Slider Example")

        radio1 = QRadioButton("&Radio horizontal slider")

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        slider.setTickInterval(10)
        slider.setSingleStep(1)

        radio1.setChecked(True)

        vbox = QVBoxLayout()
        vbox.addWidget(radio1)
        vbox.addWidget(slider)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    def input_resistance_combobox(self):
        box = QComboBox()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = SliderWindow()
    clock.show()
    sys.exit(app.exec())
