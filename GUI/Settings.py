from PyQt6.QtGui import QColor

class Settings:
    def __init__(self):
        Settings.alpha = 30
        Settings.label_to_color = {
			'NP' : QColor(  0, 255,   0, Settings.alpha),
            'P'  : QColor(255,   0,   0, Settings.alpha),
			'J'  : QColor(  0,   0, 255, Settings.alpha),
			'K'  : QColor(  0, 255,   0, Settings.alpha),
			'L'  : QColor(128,   0, 128, Settings.alpha),
			'M'  : QColor(255, 192, 203, Settings.alpha),
			'N'  : QColor(  0, 255, 255, Settings.alpha),
			'W'  : QColor(255, 215,   0, Settings.alpha),
			'Z'  : QColor(255, 215,   0, Settings.alpha)
		}
        Settings.labels_to_show = {
			'NP' : True,
            'P'  : True,
			'J'  : True,
			'K'  : True,
			'L'  : True,
			'M'  : True,
			'N'  : True,
            'W'  : True,
			'Z'  : True
		}
        Settings.line_color: QColor = QColor("blue")
        Settings.unset_color = QColor(255, 255, 255, Settings.alpha)
        Settings.unset_label = 'NONE'
        Settings.show_grid = True
        Settings.show_all_text = True
        Settings.show_durations = False
        Settings.show_comments = True
        Settings.h_maj_gridline_spacing = 100
        Settings.v_maj_gridline_spacing = 5
        Settings.h_tick_anchor = 0
        Settings.v_tick_anchor = 0
