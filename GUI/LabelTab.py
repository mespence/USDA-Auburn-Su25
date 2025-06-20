
# LabelTab.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QProgressBar, QToolButton, QSizePolicy
)
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon

from DataWindow2 import DataWindow
from EPGData import EPGData
from Labeler import Labeler
from Settings import Settings
from FileSelector import FileSelector
from SettingsWindow import SettingsWindow



# TODO: fix the cause of the warning
import warnings
warnings.filterwarnings('ignore', module="torch", message="enable_nested_tensor is True.*")


class LabelTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.epgdata = EPGData()
        self.labeler = Labeler(parent=self)

        file = self.epgdata.current_file
        self.epgdata.load_data(file)

        self.datawindow = DataWindow(self.epgdata)
        self.datawindow.plot_recording(file, "pre")
        self.datawindow.plot_transitions(file)
        self.datawindow.plot_comments(file)

        openDataButton = QPushButton("Open Data")
        openDataButton.clicked.connect(lambda: FileSelector.load_new_data(self.epgdata, self.datawindow))

        self.modelChooser = QComboBox()
        self.modelChooser.addItem("Select model...")
        self.modelChooser.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)  # Disable default item
        self.modelChooser.addItems([
            'UNet (Block)', 
            'UNet (Attention)',
            'SegTransformer', 
            'TCN', 
            'Random Forests (CSVs only)'
        ])
        self.modelChooser.currentTextChanged.connect(self.labeler.load_model)
        #self.labeler.load_model("UNet (Block)")

        startSplittingButton = QPushButton("Start Probe Splitter")
        startSplittingButton.clicked.connect(lambda: self.labeler.start_probe_splitting(self.epgdata, self.datawindow))

        startLabelingButton = QPushButton("Start Automated Labeling")
        startLabelingButton.clicked.connect(lambda: self.labeler.start_labeling(self.epgdata, self.datawindow))

        saveDataButton = QPushButton("Save Labeled Data")
        saveDataButton.clicked.connect(lambda: FileSelector.export_labeled_data(self.epgdata, self.epgdata.current_file))

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setMaximumSize(400, 100)
        self.labeler.start_labeling_progress.connect(self.update_progress)
        self.labeler.stopped_labeling.connect(lambda: self.update_progress(0, 100))

        self.baselineCursorButton = QPushButton("Change to Baseline Cursor")
        self.baselineCursorButton.clicked.connect(self.switch_cursor_state)

        # A (not great) solution to stopping labeling. Clicking sets a stop flag, the process checks before
        # altering plots (i.e. changing the state) of the GUI. If clicked after, it does nothing. Not superb,
        # but unless we inject some stop signaling way down in the ML code, this is the best as far as I see.
        stopLabelingButton = QPushButton("Stop Labeling")
        stopLabelingButton.clicked.connect(self.labeler.stop_labeling)

        settingsButton = QPushButton()
        settingsIcon = QIcon.fromTheme("applications-utilities")
        if settingsIcon.isNull():
            settingsButton.setText("Settings")
        else:
            settingsButton.setIcon(settingsIcon)
        settingsButton.clicked.connect(self.openSettings)

        resetButton = QPushButton("Reset Zoom (R)")
        resetButton.clicked.connect(self.datawindow.reset_view)



        # Top layout
        top_controls = QHBoxLayout()
        top_controls.addWidget(settingsButton)
        top_controls.addWidget(self.modelChooser)
        top_controls.addWidget(resetButton)

        # Bottom layout
        bottom_controls = QHBoxLayout()
        bottom_controls.addWidget(openDataButton)
        bottom_controls.addWidget(startSplittingButton)
        bottom_controls.addWidget(startLabelingButton)
        bottom_controls.addWidget(saveDataButton)

        bottom_controls_2 = QHBoxLayout()
        bottom_controls_2.addWidget(self.progressBar)
        bottom_controls_2.addWidget(stopLabelingButton)
        bottom_controls_2.addWidget(self.baselineCursorButton)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_controls)
        main_layout.addWidget(self.datawindow)
        main_layout.addLayout(bottom_controls)
        main_layout.addLayout(bottom_controls_2)


        self.setLayout(main_layout)


        
        self.settings_window = SettingsWindow()
        self.settings_window.label_color_changed.connect(self.datawindow.change_label_color)
        self.settings_window.line_color_changed.connect(self.datawindow.change_line_color)
        self.settings_window.duration_toggled.connect(self.datawindow.set_durations_visible)
        self.settings_window.destroyed.connect(lambda: setattr(self, 'settings_window', None))
        self.settings_window.load_settings()

        #self.openSettings()

    def update_progress(self, current, total):
        self.progressBar.setValue(int((current / total) * 100))

    def switch_cursor_state(self):
        self.datawindow.cursor_state = not self.datawindow.cursor_state
        self.baselineCursorButton.setText(
            "Change to Normal Cursor" if self.datawindow.cursor_state else "Change to Baseline Cursor"
        )

    def openSettings(self):
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()




if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    Settings()
    app = QApplication([])
    tab = LabelTab()
    tab.showMaximized()

    sys.exit(app.exec())


