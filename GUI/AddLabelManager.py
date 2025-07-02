from pyqtgraph import InfiniteLine, LinearRegionItem, mkBrush
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from TextEdit import TextEdit  # reuse your existing text input field
from Settings import Settings
from LabelArea import LabelArea


class AddLabelManager:
    def __init__(self, datawindow):
        self.dw = datawindow
        self.active = False
        self.first_time = None
        self.dragging = False

        self.line1 = InfiniteLine(angle=90, movable=False, pen='b')
        self.line2 = InfiniteLine(angle=90, movable=False, pen='b')
        self.region = LinearRegionItem(orientation='vertical', brush=mkBrush(QColor(0, 0, 255, 50)), movable=False)

        for item in [self.line1, self.line2, self.region]:
            item.setZValue(-20)
            item.setVisible(False)
            self.dw.plot_item.addItem(item)

    def start(self):
        self.active = True
        self.first_time = None
        self.dragging = False
        self.dw.selection.deselect_all()

    def cancel(self):
        self.active = False
        self.first_time = None
        self.dragging = False
        for item in [self.line1, self.line2, self.region]:
            item.setVisible(False)

    def mouse_press(self, x: float):
        if not self.active:
            return
        if self.first_time is None:
            self.first_time = x
            for item in [self.line1, self.line2, self.region]:
                item.setVisible(True)
            self.line1.setPos(x)
            self.line2.setPos(x)
            self.region.setRegion((x, x))
            self.dragging = True
        else:
            self.finalize_label(x)

    def mouse_release(self, x: float):
        if self.active and self.first_time is not None and self.dragging:
            self.finalize_label(x)

    def mouse_move(self, x: float):
        if self.active and self.first_time is not None:
            self.line2.setPos(x)
            self.region.setRegion((min(self.first_time, x), max(self.first_time, x)))

    def finalize_label(self, second_time: float):
        start, end = sorted([self.first_time, second_time])
        duration = end - start

        # Prompt for label
        dialog = QDialog(self.dw)
        dialog.setWindowTitle("Enter Waveform")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Waveform Label:"))
        input_field = TextEdit()
        layout.addWidget(input_field)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog.setLayout(layout)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.cancel()
            return

        label = input_field.toPlainText().strip().upper()
        if not label:
            self.cancel()
            return
        
        for item in [self.line1, self.line2, self.region]:
            item.setVisible(False)

        # Build updated label list
        new_labels = []
        for la in self.dw.labels:
            s0 = la.start_time
            e0 = s0 + la.duration

            if start <= s0 and e0 <= end:
                # Fully contained — remove this label
                for item in la.getItems():
                    if item.scene() is not None:
                        item.scene().removeItem(item)
                continue

            elif s0 < start < e0:
                la.duration = start - s0
                la.set_transition_line('right', start)
                la.update_label_area()

            elif s0 < end < e0:
                la.start_time = end
                la.duration = e0 - end
                la.set_transition_line('left', end)
                la.update_label_area()

            new_labels.append(la)

        # Replace label list
        self.dw.labels = new_labels

        # Add new label
        new_label = LabelArea(start, duration, label, self.dw)
        self.dw.labels.append(new_label)

        # Sort
        self.dw.labels.sort(key=lambda la: la.start_time)

        # Push to epgdata
        transitions = [(la.start_time, la.label) for la in self.dw.labels]
        self.dw.epgdata.set_transitions(self.dw.file, transitions, self.dw.transition_mode)

        self.dw.update_plot()
        self.cancel()

