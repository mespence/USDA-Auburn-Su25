import ctypes
import os
# Qt Imports
import sys
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCharts import *

# Project imports
from DataWindow import DataWindow
from EPGData import EPGData
from Labeler import Labeler
from settings.Settings import Settings
from FileSelector import FileSelector
from SettingsWindow import SettingsWindow
from ModelSelector import ModelSelector

class LabelingTask(QRunnable):
    def __init__(self, labeler, epgdata, datawindow):
        super().__init__()
        self.labeler = labeler
        self.epgdata = epgdata
        self.datawindow = datawindow

    def run(self):
        self.labeler.start_labeling(self.epgdata, self.datawindow)

class MainWindow(QMainWindow):
    start_labeling = pyqtSignal()

    def __init__(self):
            if os.name == 'nt':
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('company.app.1')
            super().__init__()
            self.initUI()
    
    def initUI(self):
            # Supervised Classification of Insect Data and Observations
            self.setWindowTitle("SCIDO EPG Labeler")
            self.setWindowIcon(QIcon("SCIDO.png"))

            # Make widgets
            epgdata = EPGData()
            file = epgdata.current_file
            epgdata.load_data(file)
            self.datawindow = DataWindow(epgdata)
            self.datawindow.plot_recording(file, "pre")
            self.datawindow.plot_transitions(file, "pre")
            self.datawindow.plot_comments()
            openDataButton = QPushButton("Open Data")
            openDataButton.clicked.connect(lambda: FileSelector.load_new_data(epgdata, self.datawindow))

            # Choose the labeler
            labeler = Labeler()
            modelChooser = QComboBox()
            modelChooser.addItems(['UNet (Block)', 
                                   'UNet (Attention)', 
                                   'SegTransformer',
                                   'TCN',
                                   'Random Forests (CSVs only)'])
            modelChooser.currentTextChanged.connect(labeler.load_model)
            labeler.load_model("UNet (Block)") # So something loads without changing anything

            startSplittingButton = QPushButton("Start Probe Splitter")
            startSplittingButton.clicked.connect(lambda: labeler.start_probe_splitting(epgdata, self.datawindow))

            startLabelingButton = QPushButton("Start Automated Labeling")
            #startLabelingButton.clicked.connect(self.start_labeling)
            startLabelingButton.clicked.connect(lambda: labeler.start_labeling(epgdata, self.datawindow))
            
            saveDataButton = QPushButton("Save Labeled Data")
            saveDataButton.clicked.connect(lambda: FileSelector.export_labeled_data(epgdata, epgdata.current_file))

            labeler.start_labeling_progress.connect(self.update_progress)
            labeler.stopped_labeling.connect(lambda: self.update_progress(0, 100))

            self.progressBar = QProgressBar()
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(100)
            self.progressBar.setValue(0)
            self.progressBar.setMaximumSize(400, 100)

            self.baselineCursorButton = QPushButton("Change to Baseline Cursor")
            self.baselineCursorButton.clicked.connect(lambda: self.switch_cursor_state())

            # A (not great) solution to stopping labeling. Clicking sets a stop flag, the process checks before
            # altering plots (i.e. changing the state) of the GUI. If clicked after, it does nothing. Not superb,
            # but unless we inject some stop signaling way down in the ML code, this is the best as far as I see.
            self.stopLabelingButton = QPushButton("Stop Labeling")
            self.stopLabelingButton.clicked.connect(labeler.stop_labeling)

            # Make Settings
            settingsButton = QPushButton()
            settingsIcon = QIcon.fromTheme("applications-utilities")
            if settingsIcon.isNull():
                settingsButton.setText("Settings")
            else:
                settingsButton.setIcon(settingsIcon)

            settingsButton.clicked.connect(self.openSettings)

            resetButton = QPushButton("Reset Zoom (R)", self)
            resetButton.setGeometry(10, 10, 100, 30)
            resetButton.clicked.connect(self.datawindow.resetView)
            
            # Arrange the widgets
            centralWidget = QWidget()
            layout = QGridLayout()
            layout.addWidget(settingsButton, 0, 0)
            layout.addWidget(self.datawindow, 1, 0, 1, 0)
            layout.addWidget(openDataButton, 3, 0)
            layout.addWidget(startSplittingButton, 3, 1)
            layout.addWidget(startLabelingButton, 3, 2)
            layout.addWidget(saveDataButton, 3, 3)
            layout.addWidget(self.progressBar, 4, 0)
            layout.addWidget(self.stopLabelingButton, 4, 1)
            layout.addWidget(self.baselineCursorButton, 4, 2)
            layout.addWidget(modelChooser, 0, 1)
            layout.addWidget(resetButton, 0, 3)
        #     layout.addWidget(self.datawindow.h_scrollbar, 2, 0, 1, 0)
            centralWidget.setLayout(layout)
            self.setCentralWidget(centralWidget)
            
            if not hasattr(self, 'settings_window') or self.settings_window is None:
                    self.settings_window = SettingsWindow()
                    self.settings_window.label_color_changed.connect(self.datawindow.change_label_color)
                    self.settings_window.line_color_changed.connect(self.datawindow.change_line_color)
                    self.settings_window.label_deleted.connect(self.datawindow.delete_label)
                    self.settings_window.comments_toggled.connect(self.datawindow.set_comments_visible)
                    self.settings_window.gridline_toggled.connect(self.datawindow.set_gridlines)
                    self.settings_window.label_text_toggled.connect(self.datawindow.set_text_visible)
                    self.settings_window.duration_toggled.connect(self.datawindow.set_durations_visible)
                    self.settings_window.h_gridline_changed.connect(self.datawindow.set_h_gridline_spacing)
                    self.settings_window.v_gridline_changed.connect(self.datawindow.set_v_gridline_spacing)
                    self.settings_window.h_tick_anchor_changed.connect(self.datawindow.set_h_offset)
                    self.settings_window.v_tick_anchor_changed.connect(self.datawindow.set_v_offset)
                    self.settings_window.delete_baseline.connect(self.datawindow.delete_baseline)
                    self.settings_window.label_hidden.connect(self.datawindow.hide_label)
                    self.settings_window.destroyed.connect(lambda: setattr(self, 'settings_window', None))
                    self.settings_window.load_settings()

            self.threadpool = QThreadPool()
            self.labeler = labeler
            self.epgdata = epgdata

    def start_labeling(self):
            task = LabelingTask(self.labeler, self.epgdata, self.datawindow)
            self.threadpool.start(task)

    # To stop labeling, the labeling cannot run in the same thread as the GUI, which it currently this.
    # This resolves that, but does introduce the possibility of multithreading bugs.
    def update_progress(self, current, total):
            percentage = int((current / total) * 100)
            self.progressBar.setValue(percentage)
    
    def switch_cursor_state(self):
            self.datawindow.cursor_state = not self.datawindow.cursor_state
            if self.datawindow.cursor_state == 0:
                    self.baselineCursorButton.setText("Change to Baseline Cursor")
            elif self.datawindow.cursor_state == 1:
                    self.baselineCursorButton.setText("Change to Normal Cursor")

    def openSettings(self):
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow() 

def main():
        Settings()
        app = QApplication([])
        window = MainWindow()
        window.showMaximized()
        window.show()
        sys.exit(app.exec())

if __name__ == '__main__':
        main()
