from settings.window.SettingsWindow import SettingsWindow

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    win = SettingsWindow()
    win.show()
    sys.exit(app.exec())