from PyQt6.QtGui import QColor
from PyQt6.QtCore import QRandomGenerator
import os

class Settings:
    PLOT_LIGHT = {
        "NAME":                     "LIGHT",
        "BACKGROUND":             "#F5F5F6",
        "FOREGROUND":             "#111132",
        "AXIS_COLOR":             "#111132",
        "FONT_COLOR_1":           "#111132",
        "TRANSITION_LINE_COLOR":  "#464650",
    }

    PLOT_DARK = {
        "NAME":                     "DARK",
        "BACKGROUND":             "#1E1E1E",
        "FOREGROUND":             "#888888",
        "AXIS_COLOR":             "#EBEBEB",
        "FONT_COLOR_1":           "#EBEBEB",
        "TRANSITION_LINE_COLOR":  "#909092",
    }

    plot_theme: dict = PLOT_LIGHT
    label_colors: dict = {} # format: {label: {"LIGHT": hex, "DARK": hex}}
    data_line_color: QColor = QColor("#4A82E2")
    data_line_width: int = 2

    show_h_grid: bool = True
    show_v_grid: bool = False
    show_labels: bool = True
    show_durations: bool = False
    show_comments: bool = True

    default_recording_directory: str = os.getcwd()
    backup_recording_directory: str = os.getcwd()

    saved_settings: dict = {  # name/type map of all settings to be actually saved 
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


    @staticmethod
    def get_label_color(label: str) -> QColor:
        """
        Retrieve the QColor for a given label in the current theme.
        If not yet defined, generates a light and dark variant automatically.
        """
        label = label.upper()
        #theme_labels = Settings.label_colors[Settings.theme_name]
        if label not in Settings.label_colors:
            # Generate new color for label
            hue = QRandomGenerator.global_().bounded(0, 360)
            saturation_light = QRandomGenerator.global_().bounded(150, 256)
            saturation_dark = saturation_light - 60
            lightness_light = 220
            lightness_dark = 30

            light_color = QColor()
            light_color.setHsl(hue, saturation_light, lightness_light)

            dark_color = QColor()
            dark_color.setHsl(hue, saturation_dark, lightness_dark)

            Settings.label_colors[label] = {
                "LIGHT": light_color.name(),
                "DARK":  dark_color.name(),
            }

        return QColor(Settings.label_colors[label][Settings.plot_theme["NAME"]])

    @staticmethod
    def set_label_color(label: str, color: QColor) -> None:
        """
        Set the QColor for a label in the current theme.
        Automatically generates the corresponding color for the other theme
        by brightening or darkening the base RGB values.
        """
        label = label.upper()
        base_rgb = color.getRgb()[:3]
        inverted_rgb = [
            max(0, min(255, c - 80)) if Settings.plot_theme["NAME"] == "LIGHT"
            else min(255, c + 80) for c in base_rgb
        ]

        Settings.label_colors[label] = {
            "LIGHT": color.name() if Settings.plot_theme["NAME"] == "LIGHT" else QColor(*inverted_rgb).name(),
            "DARK":  color.name() if Settings.plot_theme["NAME"] == "DARK" else QColor(*inverted_rgb).name(),
        }

    @staticmethod
    def rename_label(old: str, new: str):
        Settings.label_colors[new] = Settings.label_colors.pop(old)

    @staticmethod
    def delete_label(label: str):
        Settings.label_colors.pop(label, None)


