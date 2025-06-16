from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QCheckBox, QSpinBox, 
    QDoubleSpinBox, QColorDialog, QMessageBox, QComboBox,
    QLineEdit, QPushButton, QLabel
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import pyqtSignal, QRandomGenerator, Qt, QSettings

from Settings import Settings

class SettingsWindow(QWidget):
    # Emit signal from the settings window to objects that need
    # update when color is changed.
    label_color_changed = pyqtSignal(object, object)
    line_color_changed = pyqtSignal(object)
    label_deleted = pyqtSignal(object)
    label_added = pyqtSignal(object, object)
    #gridline_toggled = pyqtSignal(object)
    label_text_toggled = pyqtSignal(object)
    duration_toggled = pyqtSignal(object)
    comments_toggled = pyqtSignal(object)
    #h_gridline_changed = pyqtSignal(object)
    #v_gridline_changed = pyqtSignal(object)
    #h_tick_anchor_changed = pyqtSignal(object)
    #v_tick_anchor_changed = pyqtSignal(object)
    label_hidden = pyqtSignal(object, object)
    delete_baseline = pyqtSignal()

    # Variables
    label_combo_box: QComboBox
    add_label_text: QLineEdit
    add_label_button: QPushButton
    label_color_button: QPushButton
    remove_label_button: QPushButton
    line_color: QLabel
    line_color_button: QPushButton

    def __init__(self):
        super().__init__()
        self.settings = QSettings("USDA", "SCIDO")

        self.setWindowTitle("Settings")
        self.setGeometry(300, 300, 300, 200)

        layout = QGridLayout()
        self.setLayout(layout)

        self.add_label_line = QLineEdit()
        self.add_label_button = QPushButton("Add label")
        self.add_label_button.clicked.connect(self.add_label_dialog)

        # ComboBox
        self.label_combo_box = QComboBox()
        self.label_combo_box.addItems(Settings.label_to_color.keys())
        self.label_combo_box.currentIndexChanged.connect(self.changed_label)
        # Edit colors
        self.label_color_button = QPushButton("Color...")
        self.label_color_button.clicked.connect(self.open_label_color_picker)
        # Delete labels
        self.remove_label_button = QPushButton("Delete")
        self.remove_label_button.clicked.connect(self.delete_label_dialog)
        # Show/hide label
        self.show_label = QCheckBox("Show label color")
        self.show_label.stateChanged.connect(lambda state: self.set_show_label(self.label_combo_box.currentText(), state))
        self.show_label.setCheckState(Qt.CheckState.Checked) # All labels shown on start

        self.line_color = QLabel("Line color")
        self.line_color_button = QPushButton("Color...")
        self.line_color_button.clicked.connect(self.open_line_color_picker)

        # self.gridline_vis_check = QCheckBox("Show grid")
        # self.gridline_vis_check.stateChanged.connect(lambda state: self.gridline_toggled.emit(state == Qt.CheckState.Checked.value))
        # self.gridline_vis_check.setCheckState(Qt.CheckState.Checked if Settings.show_grid else Qt.CheckState.Unchecked)

        self.labeltext_vis_check = QCheckBox("Show label text")
        self.labeltext_vis_check.setCheckState(Qt.CheckState.Checked)
        self.labeltext_vis_check.stateChanged.connect(lambda state: self.label_text_toggled.emit(state == Qt.CheckState.Checked.value))

        self.durationtext_vis_check = QCheckBox("Show durations")
        self.durationtext_vis_check.setCheckState(Qt.CheckState.Unchecked)
        self.durationtext_vis_check.stateChanged.connect(self.handle_toggle_duration_visibility)

        self.comments_vis_check = QCheckBox("Show comments")
        self.comments_vis_check.setCheckState(Qt.CheckState.Checked)
        self.comments_vis_check.stateChanged.connect(lambda state: self.comments_toggled.emit(state == Qt.CheckState.Checked.value))

        # self.h_gridlines_spacing_label = QLabel("Hori. spacing")
        # self.h_gridlines = QDoubleSpinBox()
        # #self.h_gridlines.setValue(Settings.h_maj_gridline_spacing)
        # self.h_gridlines.setRange(1., 200.)
        # self.h_gridlines.setDecimals(1)
        # self.h_gridlines.valueChanged.connect(self.handle_change_h_gridline)
        # self.h_gridlines_offset_label = QLabel("Hori. offset")
        # self.h_gridline_offset = QSpinBox()
        # self.h_gridline_offset.setValue(Settings.h_tick_anchor)
        # self.h_gridline_offset.setRange(-2147483648, 2147483647)
        # self.h_gridline_offset.valueChanged.connect(self.handle_change_h_tick_anchor)

        # self.v_gridlines_spacing_label = QLabel("Vert. spacing")
        # self.v_gridlines = QDoubleSpinBox()
        # self.v_gridlines.setValue(Settings.v_maj_gridline_spacing)
        # self.v_gridlines.setRange(1., 10.)
        # self.v_gridlines.setDecimals(1)
        # self.v_gridlines.valueChanged.connect(self.handle_change_v_gridline)
        # self.v_gridlines_offset_label = QLabel("Vert. offset")
        # self.v_gridline_offset = QSpinBox()
        # self.v_gridline_offset.setValue(Settings.v_tick_anchor)
        # self.v_gridline_offset.setRange(-2147483648, 2147483647)
        # self.v_gridline_offset.valueChanged.connect(self.handle_change_v_tick_anchor)

        self.delete_baseline_button = QPushButton("Remove baseline")
        self.delete_baseline_button.clicked.connect(self.handle_delete_baseline)
        
        self.save_settings_button = QPushButton("Save settings")
        self.save_settings_button.clicked.connect(self.save_settings)

        layout.addWidget(self.add_label_line, 0, 0)
        layout.addWidget(self.add_label_button, 0, 1)

        layout.addWidget(self.label_combo_box, 1, 0)
        layout.addWidget(self.label_color_button, 1, 1)
        layout.addWidget(self.remove_label_button, 1, 2)
        layout.addWidget(self.show_label, 1, 3)

        layout.addWidget(self.line_color, 2, 0)
        layout.addWidget(self.line_color_button, 2, 1)

        #layout.addWidget(self.gridline_vis_check, 3, 0)
        layout.addWidget(self.labeltext_vis_check, 3, 1)
        layout.addWidget(self.durationtext_vis_check, 3, 2)
        layout.addWidget(self.comments_vis_check, 3, 3)

        # layout.addWidget(self.h_gridlines_spacing_label, 4, 0)
        # layout.addWidget(self.h_gridlines, 4, 1)
        # layout.addWidget(self.h_gridlines_offset_label, 4, 2)
        # layout.addWidget(self.h_gridline_offset, 4, 3)

        # layout.addWidget(self.v_gridlines_spacing_label, 5, 0)
        # layout.addWidget(self.v_gridlines, 5, 1)
        # layout.addWidget(self.v_gridlines_offset_label, 5, 2)
        # layout.addWidget(self.v_gridline_offset, 5, 3)

        layout.addWidget(self.save_settings_button, 6, 0)
        layout.addWidget(self.delete_baseline_button, 6, 3)

    def open_label_color_picker(self):
        color = QColorDialog.getColor() 
        if color.isValid():
            color.setAlpha(Settings.alpha)
            self.handle_label_color_change(self.label_combo_box.currentText(), color)

    def open_line_color_picker(self):
        color = QColorDialog.getColor() 
        if color.isValid():
            self.handle_line_color_change(color)
        Settings.line_color = color

    # Private handlers
    def delete_label_dialog(self):
        label = self.label_combo_box.currentText()
        msg = QMessageBox(self)
        msg.setWindowTitle("Delete label")
        msg.setText(f"Are you sure you wish to delete label {label}?")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | 
                     QMessageBox.StandardButton.No)
        
        result = msg.exec()
        if result == QMessageBox.StandardButton.No:
            return
        try:
            Settings.label_to_color.pop(label)
            Settings.labels_to_show.pop(label)
        except KeyError:
            print(f"Could not find label {label} to be deleted")
            return
        self.handle_remove_label(label)

    def add_label_dialog(self):
        label = self.add_label_line.text()
        label = label.upper()
        if len(label) == 0:
            msg = QMessageBox(self)
            msg.setWindowTitle("Add label")
            msg.setText(f"Cannot add empty label")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return
        if label == Settings.unset_label:
            msg = QMessageBox(self)
            msg.setWindowTitle("Add label")
            msg.setText(f"Label {label} is reserved")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        """Generates a random QColor."""
        self.handle_add_label(label)

    def handle_label_color_change(self, label: str, color: QColor):
        Settings.label_to_color[label] = color
        print(f"Selected color for {label}: {color.name()}")
        self.label_color_changed.emit(label, color)

    def handle_line_color_change(self, color: QColor):
        self.line_color_changed.emit(color)

    def handle_remove_label(self, label: str):
        self.label_combo_box.removeItem(self.label_combo_box.currentIndex())
        self.label_deleted.emit(label)

    def handle_add_label(self, label: str):
        r = QRandomGenerator.global_().bounded(256)
        g = QRandomGenerator.global_().bounded(256)
        b = QRandomGenerator.global_().bounded(256)
        color = QColor(r, g, b)
        color.setAlpha(Settings.alpha)
        if label in Settings.label_to_color or label in Settings.labels_to_show:
            msg = QMessageBox(self)
            msg.setWindowTitle("Add label")
            msg.setText(f"Label {label} already exists")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            print(f"Label {label} already exists")
            return
        Settings.label_to_color[label] = color
        Settings.labels_to_show[label] = True
        self.label_combo_box.addItem(label)
        # Since the dialog in DataWindow reads from the label dict every time, 
        # the signal here does not have to be captured and handled, unlike delete.
        self.label_added.emit(label, color)

    # def handle_change_h_gridline(self, value: float):
    #     Settings.h_maj_gridline_spacing = value
    #     self.h_gridline_changed.emit(value)

    # def handle_change_v_gridline(self, value: float):
    #     Settings.v_maj_gridline_spacing = value
    #     self.v_gridline_changed.emit(value)

    # def handle_change_h_tick_anchor(self, value: float):
    #     Settings.h_tick_anchor = value
    #     self.h_tick_anchor_changed.emit(value)

    # def handle_change_v_tick_anchor(self, value: float):
    #     Settings.v_tick_anchor = value
    #     self.v_tick_anchor_changed.emit(value)

    def handle_delete_baseline(self):
        self.delete_baseline.emit()

    def set_show_label(self, label: str, state: Qt.CheckState):
        # Checkbox click is agnostic to what state is being selected
        if label not in Settings.label_to_color or label not in Settings.labels_to_show:
            print(f"Cannot hide {label}: label does not exist")
            return
        value = state == Qt.CheckState.Checked.value
        Settings.labels_to_show[label] = value
        self.label_hidden.emit(label, value)

    def handle_toggle_duration_visibility(self, state: int):
        value = state == Qt.CheckState.Checked.value
        Settings.show_durations = value
        self.duration_toggled.emit(value)

    # def set_show_duration(self, state: Qt.CheckState):
    #     # Checkbox click is agnostic to what state is being selected
    #     duration = self.duration_combo_box.currentText()
    #     value = state == Qt.CheckState.Checked.value
    #     Settings.durations_to_show[duration] = value
    #     self.duration_hidden.emit(duration, value)

    def changed_label(self, index: int):
        label = self.label_combo_box.currentText()
        if label not in Settings.labels_to_show:
            self.show_label.setCheckable(False)
            return
        self.show_label.setCheckable(True)
        show = Settings.labels_to_show[label]
        state = Qt.CheckState.Checked if show else Qt.CheckState.Unchecked
        self.show_label.setCheckState(state)

    def save_settings(self):
        self.settings.setValue('version', '1.0')
        #self.settings.setValue("show_label", self.show_label.isChecked())
        #self.settings.setValue("gridline_vis_check", self.gridline_vis_check.isChecked())
        # self.settings.setValue("h_gridlines", self.h_gridlines.value())
        # self.settings.setValue("v_gridlines", self.v_gridlines.value())
        # self.settings.setValue("h_gridline_offset", self.h_gridline_offset.value())
        # self.settings.setValue("v_gridline_offset", self.v_gridline_offset.value())
        self.settings.setValue("line_color", Settings.line_color.name())
        self.settings.setValue("alpha", Settings.alpha)
        self.settings.setValue("show_comments", Settings.show_comments)

        for label, color in Settings.label_to_color.items():
            self.settings.setValue(f"label_colors/{label}", color.name(QColor.NameFormat.HexArgb))
        for label, visibility in Settings.labels_to_show.items():
            self.settings.setValue(f"labels_to_show/{label}", visibility)

    def load_settings(self):
        if not self.settings.contains('version'):
            print("No settings file found, using default values.")
            return
        #show_label = self.settings.value("show_label", False, type=bool)
        # Settings.show_grid = self.settings.value("gridline_vis_check", Settings.show_grid, type=bool)
        # Settings.h_maj_gridline_spacing = self.settings.value("h_gridlines", Settings.h_maj_gridline_spacing, type=int)
        # Settings.v_maj_gridline_spacing = self.settings.value("v_gridlines", Settings.v_maj_gridline_spacing, type=int)
        # Settings.h_tick_anchor = self.settings.value("h_gridline_offset", Settings.h_maj_gridline_spacing, type=int)
        # Settings.v_tick_anchor = self.settings.value("v_gridline_offset", Settings.v_maj_gridline_spacing, type=int)
        Settings.show_comments = self.settings.value("show_comments", Settings.show_comments, type=bool)
        color_str: str = self.settings.value("line_color", "#2987cd")
        alpha = self.settings.value("alpha", 30, type=int)

        #self.show_label.setChecked(show_label)
        # self.gridline_vis_check.setCheckState(Qt.CheckState.Checked if Settings.show_grid else Qt.CheckState.Unchecked)
        # self.comments_vis_check.setCheckState(Qt.CheckState.Checked if Settings.show_comments else Qt.CheckState.Unchecked)
        # self.h_gridlines.setValue(Settings.h_maj_gridline_spacing)
        # self.v_gridlines.setValue(Settings.v_maj_gridline_spacing)
        # self.h_gridline_offset.setValue(Settings.h_tick_anchor)
        # self.v_gridline_offset.setValue(Settings.v_tick_anchor)
        # Using default color if none is loaded
        if color_str != "#2987cd":
            Settings.line_color = QColor(color_str)
            self.handle_line_color_change(Settings.line_color)
        Settings.alpha = alpha

        restored_label_to_color = {}
        restored_labels_to_show = {}
        self.settings.beginGroup("label_colors")
        for label in self.settings.childKeys():
            color_str = self.settings.value(label)
            restored_label_to_color[label] = QColor(color_str)
        self.settings.endGroup()
        self.settings.beginGroup("labels_to_show")
        for label in self.settings.childKeys():
            visibility = self.settings.value(label)
            restored_labels_to_show[label] = bool(visibility)
        self.settings.endGroup()
        Settings.label_to_color = restored_label_to_color
        Settings.labels_to_show = restored_labels_to_show

        # Send all these changes to the line series themselves.
        for label, visibility in Settings.labels_to_show.items():
            self.handle_label_color_change(label, Settings.label_to_color[label])
            self.set_show_label(label, Qt.CheckState.Checked.value if visibility else Qt.CheckState.Unchecked.value)
