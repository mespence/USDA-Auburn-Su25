import sys
from PyQt6.QtWidgets import *

class ModelSelector:
    def load_new_model(labeler):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileUrl()
        file_path = file_path.toLocalFile()
        if file_path:
            labeler.load_model(file_path)
