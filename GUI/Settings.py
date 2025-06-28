from PyQt6.QtGui import QColor
from PyQt6.QtCore import QRandomGenerator


# === Theme Definitions ===
LIGHT = {
    "NAME":                     "LIGHT",
    "BACKGROUND_COLOR_1":     "#F5F5F6",
    "BACKGROUND_COLOR_2":     "#E6E9ED",
    "POPUP_BACKGROUND_COLOR": "#F5F5F6",
    "ACCENT_COLOR":           "#70AFEA",
    "FONT_COLOR_1":           "#111132",
    "FONT_COLOR_2":           "#B1B1BD",
    "TEXT_FIELD_COLOR":       "#0A0808",
    "BUTTON_COLOR":           "#E1E4E9",
    "HYPERLINK COLOR":        "#6EB9FF", 
    "WHITE_COLOR":            "#FFFFFF",
    "BLACK_COLOR":            "#000000",
}

DARK = {
    "NAME":                     "DARK",
    "BACKGROUND_COLOR_1":     "#2D2D30",
    "BACKGROUND_COLOR_2":     "#363638",
    "POPUP_BACKGROUND_COLOR": "#39393c",
    "ACCENT_COLOR":           "#2093FE",
    "FONT_COLOR_1":           "#EBEBEB",
    "FONT_COLOR_2":           "#BDBDBD",
    "TEXT_FIELD_COLOR":       "#242428",
    "BUTTON_COLOR":           "#424244",
    "HYPERLINK COLOR":        "#8CC9FF", 
    "WHITE_COLOR":            "#FFFFFF",
    "BLACK_COLOR":            "#000000",
}

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

ACCENT_MAP = {
    "RED":    {"LIGHT": "#F28585", "DARK": "#F25555"},
    "ORANGE": {"LIGHT": "#EDB17A", "DARK": "#E1720B"},
    "YELLOW": {"LIGHT": "#E0CC87", "DARK": "#AC8C1A"},
    "GREEN":  {"LIGHT": "#8BC9C5", "DARK": "#27A341"},
    "BLUE":   {"LIGHT": "#70AFEA", "DARK": "#2093FE"},
    "PURPLE": {"LIGHT": "#A09EEF", "DARK": "#926BFF"},
    "PINK":   {"LIGHT": "#DBA0C7", "DARK": "#E454C4"},
}



class Settings:
    """
    Global settings container for theme, label color mappings, and display options. \n
    Includes persistent theme selection, dynamic label color generation,
    and runtime-configurable display toggles.
    """
    # === Static Variables ===
    theme: dict = DARK             # the main theme for the UI
    plot_theme: dict = PLOT_LIGHT  # the theme for the PlotWidgets
    accent_color_name: str = "BLUE"

    label_colors: dict = {}  # format: {label: {"LIGHT": hex, "DARK": hex}}

    data_line_color: QColor = QColor("#4A82E2")

    show_h_grid: bool = True
    show_v_grid: bool = False
    show_labels: bool = True
    show_durations: bool = False
    show_comments: bool = True

    # === Static Functions ===

    @staticmethod
    def set_active_theme(name: str) -> None:
        """
        Update the active theme palette reference (LIGHT or DARK). 
        """
        name_upper = name.upper()
        if name_upper not in ["LIGHT", "DARK"]:
            raise KeyError(f"Invalid color '{name}'. Expected one of ['LIGHT', 'DARK]")
        #Settings.theme_name = name.lower()
        Settings.theme = LIGHT if name_upper == "LIGHT" else DARK
        #change_accent_color(Settings.accent_color_name)

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
            "LIGHT": color.name() if Settings.plot_theme["NAME"] == "light" else QColor(*inverted_rgb).name(),
            "DARK":  color.name() if Settings.plot_theme["NAME"] == "dark" else QColor(*inverted_rgb).name(),
        }

# === Accent Color Application ===
def change_accent_color(color: str) -> None:
    """
    Change the accent color by name (e.g. 'RED', 'GREEN'),
    updating both LIGHT and DARK theme palettes accordingly.
    """
    color_upper = color.upper()
    if color_upper not in ACCENT_MAP:
        raise KeyError(f"Invalid color '{color}'. Expected one of {list(ACCENT_MAP.keys())}")
    Settings.accent_color_name = color_upper
    LIGHT["ACCENT_COLOR"] = ACCENT_MAP[color_upper]['LIGHT']
    DARK["ACCENT_COLOR"] = ACCENT_MAP[color_upper]['DARK']



# class Settings:
#     def __init__(self):
#         Settings.alpha = 30
#         Settings.label_to_color = {
# 			'NP'    : QColor(  0, 255,   0, Settings.alpha),
#             'A'     : QColor(  255, 0,   255, Settings.alpha),
#             'B'     : QColor(  120, 120,   255, Settings.alpha),
#             'B1'    : QColor(  120, 120,   255, Settings.alpha),
#             'B1S'   : QColor(  128, 200,   255, Settings.alpha),
#             'B2'    : QColor(  80, 170,   40, Settings.alpha),
#             'C'     : QColor(  120, 120,   80, Settings.alpha),
#             'C1'    : QColor(  120, 120,   80, Settings.alpha),
#             'C2'    : QColor(  255, 128,   255, Settings.alpha),
#             'D'     : QColor(  200, 0,   255, Settings.alpha), 
#             'F1'    : QColor(  0, 0,   120, Settings.alpha),
#             'FB'    : QColor(  120, 0,   90, Settings.alpha),
#             'FB1W'  : QColor(  255, 125,   255, Settings.alpha),
#             'F2'    : QColor(  190, 60,   0, Settings.alpha),
#             'G'     : QColor(  40, 255,   0, Settings.alpha),
#             'P'     : QColor(255,   0,   0, Settings.alpha),
# 			'J'     : QColor(  0,   0, 255, Settings.alpha),
# 			'K'     : QColor(  0, 255,   0, Settings.alpha),
# 			'L'     : QColor(128,   0, 128, Settings.alpha),
# 			'M'     : QColor(255, 192, 203, Settings.alpha),
# 			'N'     : QColor(  0, 255, 255, Settings.alpha),
# 			'W'     : QColor(255, 215,   0, Settings.alpha),
# 			'Z'     : QColor(255, 215,   0, Settings.alpha),
#             'END AREA' : QColor(  0,   0,   0, 0)
# 		}
#         Settings.labels_to_show = {
# 			'NP' : True,
#             'P'  : True,
# 			'J'  : True,
# 			'K'  : True,
# 			'L'  : True,
# 			'M'  : True,
# 			'N'  : True,
#             'W'  : True,
# 			'Z'  : True
# 		}
#         Settings.line_color: QColor = QColor("blue")
#         Settings.unset_color = QColor(255, 255, 255, Settings.alpha)
#         Settings.unset_label = 'NONE'
#         Settings.show_grid = True
#         Settings.show_labels = True
#         Settings.show_durations = False
#         Settings.show_comments = True