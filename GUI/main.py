import os
import sys
import ctypes

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QMessageBox,
    QMenuBar,
    QMenu,
    QTabWidget,
    QTabBar
)
from PyQt6.QtCore import QRunnable, pyqtSignal, QThreadPool, QObject, QEvent, Qt, QSize, QTimer
from PyQt6.QtGui import QIcon, QFont, QFontDatabase, QAction, QKeySequence
from pyqtgraph import setConfigOptions

from LoadingScreen import LoadingScreen
from EPGData import EPGData
from FileSelector import FileSelector

from settings.Settings import Settings
from SettingsWindow_old import SettingsWindow

from live_view.LiveViewTab import LiveViewTab
from label_view.LabelTab import LabelTab
from label_view.LabelDataWindow import LabelDataWindow # TODO: remove this from main (import through LabelTab)
from label_view.Labeler import Labeler

# from ModelSelector import ModelSelector


if os.name == "nt":
    print("Windows detected, running with OpenGL")
    setConfigOptions(useOpenGL = True)


class LabelingTask(QRunnable):
    def __init__(
        self, labeler: Labeler, epgdata: EPGData, datawindow: LabelDataWindow
    ) -> None:
        super().__init__()
        self.labeler: Labeler = labeler
        self.epgdata: EPGData = epgdata
        self.datawindow: LabelDataWindow = datawindow

    def run(self) -> None:
        self.labeler.start_labeling(self.epgdata, self.datawindow)


class MainWindow(QMainWindow):
    start_labeling = pyqtSignal()

    def __init__(self) -> None:
        if os.name == "nt":  # windows
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "company.app.1"  # needed to set taskbar icon on windows
            )
        super().__init__()
        self.initUI()

    def initUI(self):
        # Supervised Classification of Insect Data and Observations
        self.setWindowTitle("SCIDO EPG Labeler")
        self.setWindowIcon(QIcon("SCIDO.png"))
        self.move(0,0)

        # === Menu Bar ===
        menubar = QMenuBar(self)
        file_menu = QMenu("File", self)

        new_recording = file_menu.addAction("New recording")
        new_recording.setShortcut(QKeySequence("Ctrl+N"))
        #new_recording.triggered.connect(self.create_recording())

        open_recording = file_menu.addAction("Open existing recording")
        file_menu.addSeparator()
        export_comment_csv = file_menu.addAction("Export Comments")
        export_comment_csv.triggered.connect(self.export_comments_from_current_tab)
        export_data = file_menu.addAction("Export Data")
        export_data.triggered.connect(self.export_data)
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
        self.tabs.currentChanged.connect(self.handle_tab_change)
        QTimer.singleShot(0, self.set_initial_focus)

        self.tabs.setStyleSheet("""
            QTabBar::tab {
                font-size: 14px;
                font-weight: bold;
                padding: 8px 20px 10px 20px;
                border: none;
                margin-right: 0px;
            }
            QTabBar::tab:selected {
                background: #1D2934;
                padding-bottom: 8px;
                border-bottom: 3px solid #4aa8ff;
            }

            QTabBar::tab:!selected {
                background: #1e1e1e;
                border-bottom: 1px solid #282828;
            }
                                
            QTabWidget::pane {
                border: 1px solid palette(mid);
                border-radius: 0px;
                top: -1px;
            }
        """)
        self.tabs.tabBar().setCursor(Qt.CursorShape.PointingHandCursor)
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


    def export_comments_from_current_tab(self):
        current_widget = self.tabs.currentWidget()
        if isinstance(current_widget, LiveViewTab) or isinstance(current_widget, LabelTab):
            current_widget.datawindow.export_comments()
        else:
            msg = QMessageBox(self)
            msg.setWindowTitle("Cannot Export Comments")
            msg.setText("Current tab does not support exporting comments.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
    
    def export_data(self):
        current_widget = self.tabs.currentWidget()
        if isinstance(current_widget, LiveViewTab):
            current_widget.datawindow.export_df()
        else:
            msg = QMessageBox(self)
            msg.setWindowTitle("Cannot Export Data")
            msg.setText("Current tab does not support exporting comments.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

    def set_initial_focus(self):
        current_widget = self.tabs.currentWidget()
        if isinstance(current_widget, LiveViewTab):
            current_widget.datawindow.setFocus()
        elif isinstance(current_widget, LabelTab):
            current_widget.datawindow.setFocus()
    
    def handle_tab_change(self, index: int):
        widget = self.tabs.widget(index)
        if isinstance(widget, LiveViewTab):
            widget.datawindow.setFocus()
        elif isinstance(widget, LabelTab):
            widget.datawindow.setFocus()
    
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
        self.datawindow: LabelDataWindow = mainwindow.label_tab.datawindow

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

def load_fonts():
    relative_paths = [
        "fonts/Inter-Regular.otf",
        "fonts/Inter-Bold.otf", 
        "fonts/Inter-Italic.otf", 
        "fonts/Inter-BoldItalic.otf"
    ]

    for font in relative_paths:
        QFontDatabase.addApplicationFont(font)


def main():
    Settings()
    app = QApplication([])

    load_fonts()
    splash = LoadingScreen()
    splash.show()
    QApplication.processEvents()

    window = MainWindow()

    window.tabs.setCurrentIndex(1)
    QApplication.processEvents()
    window.tabs.setCurrentIndex(0)
    
    # Display Focused
    window.showMaximized()
    window.raise_()
    window.activateWindow()

    
    
    tracker = GlobalMouseTracker(window)
    app.installEventFilter(tracker)

    splash.close()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
