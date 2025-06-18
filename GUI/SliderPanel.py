from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, 
    QSlider, QLineEdit, QPushButton, QVBoxLayout, 
    QGridLayout, QFrame
)
from PyQt6.QtCore import Qt

import sys, json



class SliderPanel(QWidget):
    def __init__(self, parent: str = None):
        super().__init__(parent=parent)
        self.setWindowTitle("Control Panel")

        self.socket_client = self.parent().socket_client
        
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
        self.ddsa_slider.setRange(-500,0)

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
            "Input Resistance": self.input_resistance,
            "PGA 1": self.pga1_slider,
            "PGA 2": self.pga2_slider,
            "SCA": self.sca_slider,
            "SCO": self.sco_slider,
            "DDS": self.dds_slider,
            "DDSA": self.ddsa_slider,
            "D0": self.d0_slider,
            "D1": self.d1_slider,
            "D2": self.d2_slider,
            "D3": self.d3_slider,
            "Excitation Frequency": self.excitation_freq,
        }

        for label, item in self.controls.items():
            if isinstance(item, QSlider):
                item.valueChanged.connect(lambda val, l=label: self.on_control_change(l, val))
            elif isinstance(item, QComboBox):
                item.currentTextChanged.connect(lambda text, l=label: self.on_control_change(l, text))


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
        

    def get_values(self):
        print("-------------------------")
        for label, item in self.controls.items():
            if isinstance(item, QSlider):
                print(f"{label}: {item.value()}")
            elif isinstance(item, QComboBox):
                 print(f"{label}: {item.currentText()}")

    def get_items(self):
        return self.controls.items()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_P:
            self.get_values()
        # if event.key() == Qt.Key.Key_R:
        #     print(len(self.socket_client.recv_queue.queue))
        super().keyPressEvent(event)

    def on_control_change(self, label, value):
        #print(f"{label} changed to {value}")
        data_dict = {"type":"control", "control_type":label, "value":value}
        self.socket_client.send(data_dict)


    def recv_loop(self):
        while self.socket_client._running:
            if not self.socket_client.recv_queue.empty():
                raw_str = self.socket_client.receive()

                message_list = raw_str.split("\n")
                message_list.remove('')
            
                messages = [json.loads(s) for s in message_list]
                for message in messages:
                    if message['type'] == 'data':
                        print('data received')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SliderPanel()
    window.show()
    sys.exit(app.exec())
