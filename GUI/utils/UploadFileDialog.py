from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QDialogButtonBox,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
import os
import sys
from settings.Settings import Settings

class UploadFileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Upload Recording File")
        self.setMinimumWidth(500)
        self.setModal(True)

        self.selected_file_path = None

        # --- Widgets ---
        self.file_path_label = QLabel("Selected File:")
        self.file_path_display = QLineEdit()
        self.file_path_display.setReadOnly(True) # User can't type here
        self.file_path_display.setPlaceholderText("No file selected")

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_for_file)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept_selection)
        self.button_box.rejected.connect(self.reject)

        file_selection_layout = QHBoxLayout()
        file_selection_layout.addWidget(self.file_path_display)
        file_selection_layout.addWidget(self.browse_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.file_path_label)
        main_layout.addLayout(file_selection_layout)
        main_layout.addStretch() # Pushes content to top
        main_layout.addWidget(self.button_box)

    def browse_for_file(self):
        """
        Opens a QFileDialog to let the user select a CSV file.
        """
        # You can set a default directory if desired, e.g., os.path.expanduser("~")
        # filter: "CSV Files (*.csv);;All Files (*)"
        # caption: "Select EPG Recording File"
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select EPG Recording File")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile) # user must select an existing file
        file_dialog.setNameFilter("EPG Files (*.csv *.wdq *.daq);;All Files (*)")
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

        file_dialog.setDirectory(Settings.default_recording_directory)

        if file_dialog.exec(): # Show the dialog
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.selected_file_path = selected_files[0]
                self.file_path_display.setText(os.path.basename(self.selected_file_path)) # display just the filename
                print(f"DEBUG: File selected: {self.selected_file_path}")

    def accept_selection(self):
        """
        Validates the selected file path and emits the signal if valid.
        """
        if self.selected_file_path and os.path.exists(self.selected_file_path):
            self.accept() # close the dialog with Accepted result
        else:
            QMessageBox.warning(self, "No File Selected", "Please select a valid recording file to upload.")
    
    def get_file_path(self):
        return self.selected_file_path

def main():
    app = QApplication(sys.argv)
    dialog = UploadFileDialog()
    if dialog.exec():
        print(dialog.get_file_path())
    else:
        print("User cancelled.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
