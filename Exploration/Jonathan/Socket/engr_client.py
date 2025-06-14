
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QComboBox, QLineEdit, QPushButton
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
import sys, json


from SocketClient import SocketClient

class EngrUI(QWidget):
    def __init__(self, socket_client: SocketClient):
        super().__init__()
        self.setWindowTitle("Engineering UI (no sliders)")
        
        self.socket_client = socket_client
        self.control_values = {
            "Input Resistance": "100K",
            "PGA 1": 0,
            "PGA 2": 0,
            "SCA": 0,
            "SCO": 0,
            "DDS": 0,
            "DDSA": 0,
            "D0": 0,
            "D1": 0,
            "D2": 0,
            "D3": 0,
            "Excitation Frequency": 1000,
        }
        

        self.input_widgets = {}
        layout = QVBoxLayout()
        self.grid = QGridLayout()

        # Text label to display the dictionary
        self.dict_display = QLabel()
        self.dict_display.setFont(QFont("Courier New", 10))
        self.dict_display.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.dict_display.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.dict_display.setMinimumWidth(250)
        self.dict_display.setWordWrap(True)

        # Fill grid with controls and attach update events
        for i, (key, val) in enumerate(self.control_values.items()):
            label = QLabel(key)
            self.grid.addWidget(label, i, 0)

            if key == "Input Resistance":
                widget = QComboBox()
                widget.addItems(["100K", "1M", "10M", "100M", "1G", "10G", "100G", "Mux:7"])
                widget.setCurrentText(val)
                widget.currentTextChanged.connect(self.update_control_values)
            elif key == "Excitation Frequency":
                widget = QComboBox()
                widget.addItems(["1000", "0", ""])
                widget.setCurrentText(str(val))
                widget.currentTextChanged.connect(self.update_control_values)
            else:
                widget = QLineEdit(str(val))
                widget.setAlignment(Qt.AlignmentFlag.AlignRight)
                widget.textChanged.connect(self.update_control_values)

            self.input_widgets[key] = widget
            self.grid.addWidget(widget, i, 1)

        self.grid.addWidget(self.dict_display, 0, 2, len(self.control_values), 1)  # span rows

        layout.addLayout(self.grid)

        self.setLayout(layout)
        self.update_control_values()  # initialize display

    def update_control_values(self):
        for key, widget in self.input_widgets.items():
            if isinstance(widget, QComboBox):
                self.control_values[key] = widget.currentText()
            else:
                text = widget.text()
                try:
                    self.control_values[key] = int(text)
                except ValueError:
                    self.control_values[key] = text  # fallback to raw string
        self.dict_display.setText(json.dumps(self.control_values, indent=2))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    socket_client = SocketClient()
    window = EngrUI(socket_client)
    window.show()
    sys.exit(app.exec())
