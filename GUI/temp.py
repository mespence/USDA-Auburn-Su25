from PyQt6.QtCore import QSettings

settings = QSettings("USDA", "SCIDO")
settings.clear()
settings.sync()