from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, 
    QSlider, QLineEdit, QPushButton, QVBoxLayout, 
    QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot

import sys



class SliderPanel(QWidget):
    def __init__(self, parent: str = None):
        super().__init__(parent=parent)
        self.setWindowTitle("Control Panel")

        self.socket_client = self.parent().socket_client
        self._suppress = False  # whether slider signals are suppressed

        # self.suppress_signal: bool = False # whether slider change signals are hidden from the socket
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title_label = QLabel("EPG Controls")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title_label)

        hr = QFrame()
        hr.setFrameShape(QFrame.Shape.HLine)
        hr.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(hr)

        grid = QGridLayout()

        # Row 0: Input Resistance
        grid.addWidget(QLabel("Input Resistance"), 0, 0)
        self.input_resistance = QComboBox()
        self.input_resistance.addItems(["100K", "1M", "10M", "100M", "1G", "10G", "100G", "Mux:7"])
        grid.addWidget(self.input_resistance, 0, 1)
        grid.addWidget(QLabel("Ω"), 0, 2)
        
        def sync_slider_and_entry(slider: QSlider, entry: QLineEdit):
            # (1) Slider → Entry
            slider.valueChanged.connect(lambda val: entry.setText(str(val)))

            # (2) Entry → Slider
            def on_text_edited(text):
                try:
                    val = int(text)
                    val = max(slider.minimum(), min(val, slider.maximum()))  # clamp
                    if slider.value() != val:
                        slider.setValue(val)
                except ValueError:
                    pass  # Ignore invalid input

            entry.textEdited.connect(on_text_edited)

        def add_slider_row(row, label_text, unit=None):
            grid.addWidget(QLabel(label_text), row, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setFixedWidth(150)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            grid.addWidget(slider, row, 1)

            entry = QLineEdit("0")
            entry.setFixedWidth(50)
            grid.addWidget(entry, row, 2)

            if unit:
                grid.addWidget(QLabel(unit), row, 3)

            sync_slider_and_entry(slider, entry)    

            return slider


        # Add rows
        self.pga1_slider = add_slider_row(1, "PGA 1", "dB")
        self.pga1_slider.setRange(0,7)

        self.pga2_slider = add_slider_row(2, "PGA 2", "dB")
        self.pga2_slider.setRange(0,7)

        self.sca_slider = add_slider_row(3, "Signal Chain Amplification", "V")
        self.sca_slider.setRange(1,700)

        self.sco_slider = add_slider_row(4, "Signal Chain Offset", "V")
        self.sco_slider.setRange(-33,33)

        self.dds_slider = add_slider_row(5, "DDS Offset", "V")
        self.dds_slider.setRange(-33,33)

        self.ddsa_slider = add_slider_row(6, "DDSA Amplitude", "V")
        self.ddsa_slider.setRange(-500,-1)

        for i in range(4):
            label = f"Digipot Channel {i}"
            slider = add_slider_row(7 + i, label)
            slider.setRange(0,255)
            setattr(self, f"d{i}_slider", slider)

        # Excitation Frequency row
        row = 11
        grid.addWidget(QLabel("Excitation Frequency"), row, 0)
        self.excitation_freq = QComboBox()
        self.excitation_freq.addItems(["1000", "0", ""])
        grid.addWidget(self.excitation_freq, row, 1)
        grid.addWidget(QLabel("Hz"), row, 2)


        self.controls = {
            "inputResistance": self.input_resistance,
            "pga1": self.pga1_slider,
            "pga2": self.pga2_slider,
            "sca": self.sca_slider,
            "sco": self.sco_slider,
            "ddso": self.dds_slider,
            "ddsa": self.ddsa_slider,
            "d0": self.d0_slider,
            "d1": self.d1_slider,
            "d2": self.d2_slider,
            "d3": self.d3_slider,
            "excitationFrequency": self.excitation_freq,
        }

        for label, item in self.controls.items():
            if isinstance(item, QSlider):
                item.valueChanged.connect(lambda val, l=label: self.send_control_update(l, val))
            elif isinstance(item, QComboBox):
                item.currentTextChanged.connect(lambda text, l=label: self.send_control_update(l, text))


        layout.addLayout(grid)

        # Buttons
        button_grid = QGridLayout()
        button_grid.addWidget(QPushButton("ON"), 0, 0)
        button_grid.addWidget(QPushButton("START"), 0, 1)
        button_grid.addWidget(QPushButton("OFF"), 0, 2)

        button_grid.addWidget(QPushButton("Cancel"), 1, 0)
        button_grid.addWidget(QPushButton("Revert to Defaults"), 1, 1)

        button_grid.addWidget(QPushButton("Apply"), 2, 0)
        button_grid.addWidget(QPushButton("Apply and Close"), 2, 1)

        layout.addLayout(button_grid)
        layout.addStretch(1)
        self.setLayout(layout)

    def send_control_update(self, name, value):
        if self._suppress:
            return

        self.socket_client.send({
            "source": self.socket_client.client_id,
            "type": "control",
            "name": name,
            "value": value
        })
            
    @pyqtSlot(str, object, str)
    def set_control_value(self, name, value, source = None):
        if source == self.socket_client.client_id:
            return
        
        widget = self.controls.get(name)
        if widget is None:
            print(f"[CS] Unknown control name: {name}")
            return
        
        self._suppress = True
    
        if isinstance(widget, QSlider):
            value = int(value)
            if widget.value() != value:
                widget.setValue(value)
        if isinstance(widget, QComboBox):
            index = widget.findText(value)
            if index != -1 and widget.currentIndex() != index:
                widget.setCurrentIndex(index)

        QTimer.singleShot(0, lambda: setattr(self, "_suppress", False))

    @pyqtSlot(dict)        
    def set_all_controls(self, full_state: dict):
        """
        Sets all controls to the values given by a full control state dictionary.
        """
        for name, value in full_state.items():
            self.set_control_value(name, value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SliderPanel()
    window.show()
    sys.exit(app.exec())
