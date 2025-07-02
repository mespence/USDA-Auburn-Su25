from PyQt6.QtGui import QColor

class Settings:
    def __init__(self):
        Settings.alpha = 30
        Settings.label_to_color = {
			'NP'    : QColor(  0,   255,     0, Settings.alpha),
            'A'     : QColor(  255,   0,   255, Settings.alpha),
            'B'     : QColor(  120, 120,   255, Settings.alpha),
            'B1'    : QColor(  120, 120,   255, Settings.alpha),
            'B1S'   : QColor(  128, 200,   255, Settings.alpha),
            'B2'    : QColor(  80,  170,    40, Settings.alpha),
            'C'     : QColor(  120, 120,    80, Settings.alpha),
            'C1'    : QColor(  120, 120,    80, Settings.alpha),
            'C2'    : QColor(  255, 128,   255, Settings.alpha),
            'CG'    : QColor(60, 80,   255, Settings.alpha),
            'D'     : QColor(  200,   0,   255, Settings.alpha), 
            'DG'    : QColor(  200,   120,  60, Settings.alpha), 
            'F1'    : QColor(  0, 0,   120, Settings.alpha),
            'F3'    : QColor( 80, 140, 200, Settings.alpha),
            'F4'    : QColor( 80, 20, 200, Settings.alpha),
            'FB'    : QColor(  120, 0,   90, Settings.alpha),
            'FB1W'  : QColor(  255, 125,   255, Settings.alpha),
            'F2'    : QColor(  190, 60,   0, Settings.alpha),
            'G'     : QColor(  40, 255,   0, Settings.alpha),
            'P'     : QColor(255,   0,   0, Settings.alpha),
			'J'     : QColor(  0,   0, 255, Settings.alpha),
			'K'     : QColor(  0, 255,   0, Settings.alpha),
			'L'     : QColor(128,   0, 128, Settings.alpha),
			'M'     : QColor(255, 192, 203, Settings.alpha),
			'N'     : QColor(  0, 255, 255, Settings.alpha),
			'W'     : QColor(255, 215,   0, Settings.alpha),
			'Z'     : QColor(255, 215,   0, Settings.alpha),
            '2'     : QColor(100, 60,   60, Settings.alpha),
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
        Settings.show_labels = True
        Settings.show_durations = False
        Settings.show_comments = True