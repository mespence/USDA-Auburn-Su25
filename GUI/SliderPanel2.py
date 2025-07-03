from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, 
    QSlider, QLineEdit, QPushButton, QVBoxLayout, 
    QHBoxLayout, QGridLayout, QFrame
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

        # DC/AC Toggle Switch

        grid.addWidget(QLabel("Mode"), 0, 0)
        self.mode_toggle = QComboBox()
        self.mode_toggle.addItems(["DC", "AC"])

        self.mode_toggle.currentIndexChanged.connect(self.on_mode_change)
        grid.addWidget(self.mode_toggle, 0, 1)
        
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

        self.dds_slider = add_slider_row(1, "Excitation Voltage", "V")
        self.dds_slider.setRange(-33,33)
        self.dds_slider.setVisible(False)

        self.ddsa_slider = add_slider_row(2, "Excitation Voltage", "Vrms")
        self.ddsa_slider.setRange(-500,-1)
        self.ddsa_slider.setVisible(False)

        self.sca_slider = add_slider_row(3, "Gain", "V")
        self.sca_slider.setRange(1,700)

        self.sco_slider = add_slider_row(4, "Offset", "V")
        self.sco_slider.setRange(-33,33)

        layout.addLayout(grid)

        # Buttons
        self.on_button = QPushButton("ON", self)
        self.start_button = QPushButton("START", self)
        self.off_button  = QPushButton("OFF", self)
        self.cancel_button  = QPushButton("Cancel", self)
        self.revert_default_button  = QPushButton("Revert to Defaults", self)
        self.apply_button  = QPushButton("Apply", self)
        self.apply_close_button  = QPushButton("Apply & Close", self)

        button_layout1 = QHBoxLayout()
        button_layout1.addWidget(self.on_button)
        button_layout1.addWidget(self.start_button)
        button_layout1.addWidget(self.off_button)
        layout.addLayout(button_layout1)

        button_layout2 = QGridLayout()
        button_layout2.addWidget(self.cancel_button, 0, 0)
        button_layout2.addWidget(self.revert_default_button, 0, 1)
        button_layout2.addWidget(self.apply_button, 1, 0)
        button_layout2.addWidget(self.apply_close_button, 1, 1)

        layout.addLayout(button_layout2)

        layout.addStretch(1)
        self.setLayout(layout)

        self.controls = {
            "sca": self.sca_slider,
            "sco": self.sco_slider,
            "ddso": self.dds_slider,
            "ddsa": self.ddsa_slider,
            "on": self.on_button,
            "start": self.start_button,
            "off": self.off_button,
            "cancel": self.cancel_button,
            "revert": self.revert_default_button,
            "apply": self.apply_button,
            "applyClose": self.apply_close_button
        }

        for label, item in self.controls.items():
            if isinstance(item, QSlider):
                item.valueChanged.connect(lambda val, l=label: self.send_control_update(l, val))
            elif isinstance(item, QComboBox):
                item.currentTextChanged.connect(lambda text, l=label: self.send_control_update(l, text))
            elif isinstance(item, QPushButton):
                item.clicked.connect(lambda _, l=label: self.send_control_update(l, "clicked"))

    def on_mode_change(self):
            selected_text = self.mode_toggle.currentText()
            print("mode change:", selected_text)

            if selected_text == "DC":
                self.dds_slider.setVisible(True)
                self.ddsa_slider.setVisible(False)
                self.ddsa_slider.setValue(1)
                print("dds offset", self.dds_slider.value())
                print("ddsa amp", self.ddsa_slider.value())
            elif selected_text == "AC":
                self.dds_slider.setVisible(False)
                self.ddsa_slider.setVisible(True)
                self.dds_slider.setValue(-0.3) # check on abiliy of slider to be set to non-int
                print("dds offset", self.dds_slider.value())
                print("ddsa amp", self.ddsa_slider.value())
    
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
