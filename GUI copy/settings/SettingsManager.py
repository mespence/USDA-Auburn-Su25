# settings/SettingsManager.py
from PyQt6.QtCore import QObject, pyqtSignal, QSettings
from PyQt6.QtGui import QColor
import os, json


class SettingsManager(QObject):
    setting_changed = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = QSettings("USDA", "SCIDO")

        # Internal state dict (defaults below)
        self.state = {
            "plot_theme": self.PLOT_LIGHT,
            "label_colors": {},
            "data_line_color": QColor("#4A82E2"),
            "data_line_width": 2,
            "show_h_grid": True,
            "show_v_grid": False,
            "show_labels": True,
            "show_durations": False,
            "show_comments": True,
            "default_recording_directory": os.getcwd(),
            "backup_recording_directory": os.getcwd(),
        }

        self.settings_type_map = { # the 
            "plot_theme": dict, 
            "label_colors": dict, 
            "data_line_color": QColor, 
            "data_line_width": int,
            "show_h_grid": bool, 
            "show_v_grid": bool, 
            "show_labels": bool, 
            "show_durations": bool, 
            "show_comments": bool,
            "default_recording_directory": str,
            "backup_recording_directory":  str,        
        }

        self.load()

    # === Static theme dicts ===
    PLOT_LIGHT = {
        "NAME": "LIGHT",
        "BACKGROUND": "#F5F5F6",
        "FOREGROUND": "#111132",
        "AXIS_COLOR": "#111132",
        "FONT_COLOR_1": "#111132",
        "TRANSITION_LINE_COLOR": "#464650",
    }

    PLOT_DARK = {
        "NAME": "DARK",
        "BACKGROUND": "#1E1E1E",
        "FOREGROUND": "#888888",
        "AXIS_COLOR": "#EBEBEB",
        "FONT_COLOR_1": "#EBEBEB",
        "TRANSITION_LINE_COLOR": "#909092",
    }

    def get(self, key):
        return self.state.get(key)

    def set(self, key, value):
        if key not in self.settings_type_map:
            raise KeyError(f"Unrecognized setting: {key}")

        self.state[key] = value
        self.save(key, value)
        self.setting_changed.emit(key, value)

    def save(self, key, value):
        if isinstance(value, QColor):
            value = value.name()
        elif key == "label_colors":
            value = {
                label: {"LIGHT": d["LIGHT"], "DARK": d["DARK"]}
                for label, d in value.items()
            }

        self._settings.setValue(key, value)
        self._settings.sync()

    def load(self):
        for key, type_ in self.settings_type_map.items():
            if self._settings.contains(key):
                raw = self._settings.value(key)
            else:
                raw = self.state[key]

            if type_ is QColor:
                self.state[key] = QColor(raw)
            elif type_ is bool:
                self.state[key] = str(raw).lower() in ("true", "1", "yes")
            elif key == "plot_theme":
                theme = raw.get("NAME", "LIGHT").upper()
                self.state[key] = self.PLOT_LIGHT if theme == "LIGHT" else self.PLOT_DARK
            else:
                self.state[key] = type_(raw)

    def reset(self):
        self._settings.clear()
        self._settings.sync()
        self.load()
        for key, value in self.state.items():
            self.setting_changed.emit(key, value)