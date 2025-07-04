from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QDialogButtonBox, QPushButton
import sys

# Import both your dialog and main window classes
from NewRecordingDialog import NewRecordingDialog
from main import start_main_application
from SettingsWindow import SettingsWindow
from Settings import Settings
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
import sys

class AppLauncherDialog(QDialog):
    launchMainWindow = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SCIDO Launcher")
        self.setFixedSize(500, 250)

        main_layout = QVBoxLayout(self) # Set main layout for the dialog

        # Add a welcoming label
        welcome_label = QLabel("Welcome to SCIDO!\nChoose an action:")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(welcome_label)

        button_layout = QHBoxLayout() # Layout for the buttons

        self.settings_button = QPushButton("Settings")
        self.settings_button.setMinimumHeight(40)
        self.settings_button.clicked.connect(self.open_settings_dialog)
        button_layout.addWidget(self.settings_button)

        self.new_recording_button = QPushButton("New Recording")
        self.new_recording_button.setMinimumHeight(40)
        self.new_recording_button.clicked.connect(self.open_new_recording_dialog)
        button_layout.addWidget(self.new_recording_button)

        main_layout.addLayout(button_layout)

        # Add a simple exit button for convenience if user doesn't want to proceed
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.reject) # Rejecting the dialog will close it
        main_layout.addWidget(exit_button, alignment=Qt.AlignmentFlag.AlignRight)


    def open_settings_dialog(self):
        # need to work on settings dialog
        settings_dialog = SettingsWindow()
        settings_dialog.exec()

    def open_new_recording_dialog(self):
        new_recording_dialog = NewRecordingDialog(self)
        if new_recording_dialog.exec(): # open modally
            recording_settings = new_recording_dialog.get_data()
            # TODO somehow connect recording settings to main then to live data window
            self.launchMainWindow.emit(recording_settings)
            self.accept() # Accept and close the AppLauncherDialog

def launch_application():
    app = QApplication(sys.argv)
    launcher_dialog = AppLauncherDialog()

    main_window_instance = None

    def launch_main_window_with_settings(settings):
        nonlocal main_window_instance

        main_window_instance = start_main_application(settings=settings)
        launcher_dialog.accept() 

    launcher_dialog.launchMainWindow.connect(launch_main_window_with_settings)

    result = launcher_dialog.exec()

    if result == QDialog.DialogCode.Accepted:
        pass
    else:
        sys.exit(0)

    sys.exit(app.exec())

if __name__ == "__main__":
    launch_application()

    