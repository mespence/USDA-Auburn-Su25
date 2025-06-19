import os
import sys
import ctypes

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QProgressBar,
    QToolButton,
    QMenuBar,
    QMenu,
    QTabWidget,
)
from PyQt6.QtCore import QRunnable, pyqtSignal, QThreadPool, QObject, QEvent, Qt, QSize
from PyQt6.QtGui import QIcon

from DataWindow2 import DataWindow
from EPGData import EPGData
from Labeler import Labeler
from Settings import Settings
from FileSelector import FileSelector
from SettingsWindow import SettingsWindow

from LiveViewTab import LiveViewTab
from LabelTab import LabelTab

# from ModelSelector import ModelSelector


class LabelingTask(QRunnable):
    def __init__(
        self, labeler: Labeler, epgdata: EPGData, datawindow: DataWindow
    ) -> None:
        super().__init__()
        self.labeler: Labeler = labeler
        self.epgdata: EPGData = epgdata
        self.datawindow: DataWindow = datawindow

    def run(self) -> None:
        self.labeler.start_labeling(self.epgdata, self.datawindow)


class MainWindow(QMainWindow):
    start_labeling = pyqtSignal()

    def __init__(self) -> None:
        if os.name == "nt":  # windows
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "company.app.1"
            )
        super().__init__()
        self.initUI()

    def initUI(self):
        # Supervised Classification of Insect Data and Observations
        self.setWindowTitle("SCIDO EPG Labeler")
        self.setWindowIcon(QIcon("SCIDO.png"))
        self.move(0, 0)

        # === Menu Bar ===
        menubar = QMenuBar(self)
        file_menu = QMenu("File", self)
        file_menu.addAction("Open")
        file_menu.addAction("Save")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        help_menu = QMenu("Help", self)
        help_menu.addAction("About")

        # TODO add menu functionality
        menubar.addMenu(file_menu)
        menubar.addMenu(QMenu("Edit", self))
        menubar.addMenu(QMenu("View", self))
        menubar.addMenu(help_menu)

        self.setMenuBar(menubar)

        # === Tab Widget ===
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # First tab: Live View
        self.live_view_tab = LiveViewTab()
        self.tabs.addTab(self.live_view_tab, "Live View")

        # Second tab: Labelling View
        self.label_tab = LabelTab()
        self.tabs.addTab(self.label_tab, "Label")
    
    def closeEvent(self, event):
        self.live_view_tab.socket_server.stop()
        super().closeEvent(event)


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

    def openSliders(self):
        is_visible = self.slider_panel.isVisible()
        self.slider_panel.setVisible(not is_visible)


class GlobalMouseTracker(QObject):
    """
    Global event filter that updates the cursor hover position inside popups
    (e.g., menus) by tracking global mouse coordinates and mapping them to
    DataWindow viewbox space.
    """

    def __init__(self, mainwindow: MainWindow):
        super().__init__()
        self.mainwindow: MainWindow = mainwindow
        self.datawindow: DataWindow = mainwindow.label_tab.datawindow

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseMove:
            global_pos = (
                event.globalPosition()
            )  # pos relative to top left corner of screen
            local_pos = self.mainwindow.mapFromGlobal(
                global_pos
            )  # pos rel. to top left of application
            self.datawindow.last_cursor_pos = local_pos

            view_pos = self.datawindow.window_to_viewbox(
                local_pos
            )  # pos rel. to origin of plot
            selection = self.datawindow.selection
            selection.hovered_item = selection.get_hovered_item(
                view_pos.x(), view_pos.y()
            )
        return super().eventFilter(obj, event)


def main():
    Settings()
    app = QApplication([])
    window = MainWindow()
    window.showMaximized()
    # window.show()

    tracker = GlobalMouseTracker(window)
    app.installEventFilter(tracker)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
