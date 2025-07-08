from PyQt6.QtWidgets import (
   QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QFrame, QLabel, QToolButton, QComboBox, QPushButton, QSizePolicy,
    QSpinBox, QCheckBox, QColorDialog, QLineEdit, QMessageBox, 
    QSpacerItem
)
from PyQt6.QtGui import (
    QColor, QBrush, QPen, QPainter,
    QIcon, QMouseEvent,
)
from PyQt6.QtCore import Qt, QSize, QSettings

from settings.Settings import Settings

class AppearanceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(15)

        # === Plot Theme ComboBox ===
        theme_label = QLabel("Plot Theme")
        font = theme_label.font()
        font.setBold(True)
        theme_label.setFont(font)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentText(Settings.plot_theme["NAME"])
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)

        theme_row = QHBoxLayout()
        theme_row.setSpacing(10)
        theme_row.addSpacing(20)
        theme_row.addWidget(self.theme_combo)
        theme_row.addSpacing(1000)

        layout.addWidget(theme_label)
        layout.addLayout(theme_row)

        # === Data Line Appearance ===
        data_line_label = QLabel("Data Line")
        font = data_line_label.font()
        font.setBold(True)
        data_line_label.setFont(font)

        # Color Picker
        color_label = QLabel("Color:")
        color_label.setFixedWidth(40)

        self.data_line_color_button = QPushButton()
        self.data_line_color_button.setFixedWidth(80)
        self.data_line_color_button.clicked.connect(self.pick_line_color)

        self.set_data_line_color_button(Settings.data_line_color)

        # Width Spinner
        width_label = QLabel("Width:")
        width_label.setFixedWidth(40)

        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 5)
        self.width_spinbox.setValue(Settings.data_line_width)
        self.width_spinbox.setSuffix(" px")
        self.width_spinbox.setFixedWidth(80)
        self.width_spinbox.valueChanged.connect(self.on_width_changed)

        # Horizontal layout with spacing
        line_appearance_row = QHBoxLayout()
        line_appearance_row.setSpacing(10)

        line_appearance_row.addSpacing(20)
        line_appearance_row.addWidget(color_label)
        line_appearance_row.addWidget(self.data_line_color_button)
        line_appearance_row.addSpacing(20)
        line_appearance_row.addWidget(width_label)
        line_appearance_row.addWidget(self.width_spinbox)
        line_appearance_row.addStretch()

        layout.addWidget(data_line_label)
        layout.addLayout(line_appearance_row)

        # === Boolean Toggles ===
        toggles_label = QLabel("Display Options")
        font = toggles_label.font()
        font.setBold(True)
        toggles_label.setFont(font)

        layout.addWidget(toggles_label)

        self.checkboxes = {}
        bool_settings = {
            "Show Horizontal Grid Lines": "show_h_grid",
            "Show Vertical Grid Lines": "show_v_grid",
            "Show Waveform Labels": "show_labels",
            "Show Waveform Durations": "show_durations",
            "Show Comments": "show_comments"
        }

        for label_text, attr in bool_settings.items():
            checkbox = QCheckBox(label_text)
            checkbox.setChecked(getattr(Settings, attr))
            checkbox.toggled.connect(lambda state, a=attr: self.on_checkbox_toggled(a, state))
            self.checkboxes[attr] = checkbox

            row = QHBoxLayout()
            row.addSpacing(20)
            row.addWidget(checkbox)
            row.addStretch()
            layout.addLayout(row)

        # Horizontal Rule
        hrule = QFrame()
        hrule.setFrameShape(QFrame.Shape.HLine)
        hrule.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(hrule)


        # === Waveform Labels ===
        waveform_label = QLabel("Waveform Labels")
        font = waveform_label.font()
        font.setBold(True)
        waveform_label.setFont(font)
        layout.addWidget(waveform_label)

        # Select Label row (label + combobox side by side)
        select_row = QHBoxLayout()
        select_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        select_label = QLabel("Select Waveform:")
        select_label.setFixedWidth(100)

        self.label_selector = QComboBox()
        self.label_selector.setFixedWidth(80)
        self.label_selector.addItems(sorted(Settings.label_colors.keys()))
        self.label_selector.currentTextChanged.connect(self.load_label_info)

        self.delete_button = QPushButton("Delete All Instances of Waveform")
        self.delete_button.setFixedWidth(240)
        self.delete_button.clicked.connect(self.delete_label)

        select_row.addWidget(select_label)
        select_row.addWidget(self.label_selector)
        select_row.addSpacing(95)
        select_row.addWidget(self.delete_button)

        layout.addLayout(select_row)

        # Rename + Color picker row (side-by-side)
        edit_row = QHBoxLayout()

        # Rename section
        name_label = QLabel("Waveform Name:")
        name_label.setFixedWidth(100)
        self.name_edit = QLineEdit()
        self.name_edit.setFixedWidth(80)
        self.name_edit.textChanged.connect(self.update_rename_button_state)
        self.rename_button = QPushButton("Rename")
        self.rename_button.setFixedWidth(80)
        self.rename_button.clicked.connect(self.rename_label)

        rename_layout = QHBoxLayout()
        rename_layout.addWidget(name_label)
        rename_layout.addWidget(self.name_edit)
        rename_layout.addWidget(self.rename_button)

        edit_row.addLayout(rename_layout)

        # Color section
        label_color_label = QLabel("Color:")
        label_color_label.setFixedWidth(50)

        self.label_color_button = QPushButton()
        self.label_color_button.setFixedWidth(100)
        self.label_color_button.clicked.connect(self.pick_label_color)

        self.label_color_wrapper = QFrame()
        self.label_color_wrapper.setObjectName("LabelColorWrapper")
        self.update_label_color_wrapper_style()

        wrapper_layout = QHBoxLayout(self.label_color_wrapper)
        wrapper_layout.setContentsMargins(10, 10, 10, 10)
        wrapper_layout.addWidget(self.label_color_button)

        color_row = QHBoxLayout()
        color_row.addWidget(label_color_label)
        color_row.addWidget(self.label_color_wrapper)

        edit_row.addSpacing(40)
        edit_row.addLayout(color_row)
        edit_row.addStretch()

        layout.addLayout(edit_row)

        # Static interactable widgets
        self.apply_interactive_cursor(self.theme_combo)
        self.apply_interactive_cursor(self.data_line_color_button)
        self.apply_interactive_cursor(self.width_spinbox)
        self.apply_interactive_cursor(self.rename_button)
        self.apply_interactive_cursor(self.delete_button)
        self.apply_interactive_cursor(self.label_selector)
        self.apply_interactive_cursor(self.label_color_button)

        for checkbox in self.checkboxes.values():
            self.apply_interactive_cursor(checkbox)

        self.rename_button.installEventFilter(self)

        self.load_label_info(self.label_selector.currentText())

    def set_data_line_color_button(self, color: QColor):
        text_color = self.get_contrasting_text_color(color)
        self.data_line_color_button.setText(color.name().upper())
        self.data_line_color_button.setStyleSheet(
            f"background-color: {color.name()}; color: {text_color};"
        )

    def on_theme_changed(self, value: str):
        Settings.plot_theme = Settings.PLOT_LIGHT if value == "Light" else Settings.PLOT_DARK
        print("Theme set to:", Settings.plot_theme["NAME"])
        self.update_label_color_wrapper_style()
        self.refresh_label_color_preview()

    def on_width_changed(self, value: int):
        Settings.data_line_width = value
        print("data_line_width set to:", Settings.data_line_width)

    def on_checkbox_toggled(self, attr_name: str, value: bool):
        setattr(Settings, attr_name, value)
        print(f"{attr_name} set to: {getattr(Settings, attr_name)}")

    def load_label_info(self, label: str):
        self.name_edit.setText(label)
        color = Settings.get_label_color(label)
        self.label_color_button.setText(color.name().upper())
        text_color = self.get_contrasting_text_color(color)
        border_color = Settings.plot_theme["TRANSITION_LINE_COLOR"]
        self.label_color_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color.name()};
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                background-color: {color.lighter(120).name()};
            }}
            QPushButton:pressed {{
                background-color: {color.darker(120).name()};
            }}
        """)
        self.update_rename_button_state()

    def update_label_color_wrapper_style(self):
        theme_bg = Settings.plot_theme["BACKGROUND"]
        theme_fg = Settings.plot_theme["FOREGROUND"]
        self.label_color_wrapper.setStyleSheet(f"""
            QFrame#LabelColorWrapper {{
                background-color: {theme_bg};
                border: 1px solid {theme_fg};
                border-radius: 6px;
                padding: 6px;
            }}
        """)

    def pick_label_color(self):
        label = self.label_selector.currentText()
        current_color = Settings.get_label_color(label)
        new_color = QColorDialog.getColor(current_color, self)
        if new_color.isValid():
            Settings.set_label_color(label, new_color)
            self.load_label_info(label)

    def pick_line_color(self):
        color = QColorDialog.getColor(initial=Settings.data_line_color, parent=self)
        if color.isValid():
            Settings.data_line_color = color
            self.set_data_line_color_button(color)
            print("data_line_color set to:", color.name())

    def rename_label(self):
        old_name = self.label_selector.currentText()
        new_name = self.name_edit.text().strip().upper()

        if not new_name or new_name == old_name:
            return

        if new_name in Settings.label_colors:
            QMessageBox.warning(self, "Rename Failed", "Label name already exists.")
            return

        reply = QMessageBox.question(
            self,
            "Rename Waveform",
            f"Are you sure you want to change waveform label '{old_name}' to '{new_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # Proceed with renaming
        Settings.rename_label(old_name, new_name)
        self.label_selector.blockSignals(True)
        self.label_selector.clear()
        self.label_selector.addItems(sorted(Settings.label_colors.keys()))
        self.label_selector.setCurrentText(new_name)
        self.label_selector.blockSignals(False)
        self.load_label_info(new_name)

    def update_rename_button_state(self):
        current_name = self.label_selector.currentText().strip().upper()
        typed_name = self.name_edit.text().strip().upper()
        self.rename_button.setEnabled(bool(typed_name and typed_name != current_name))

    def delete_label(self):
        label = self.label_selector.currentText()
        reply = QMessageBox.question(
            self, "Delete Waveform",
            f"Are you sure you want to delete all instances of waveform '{label}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            Settings.delete_label(label)
            self.label_selector.clear()
            self.label_selector.addItems(sorted(Settings.label_colors.keys()))
            if Settings.label_colors:
                self.label_selector.setCurrentIndex(0)
                self.load_label_info(self.label_selector.currentText())
            else:
                self.name_edit.clear()
                self.label_color_button.setText("")
                self.label_color_button.setStyleSheet("")

    def refresh_label_color_preview(self):
        current_label = self.label_selector.currentText()
        if current_label:
            self.load_label_info(current_label)

    def get_contrasting_text_color(self, bg_color: QColor) -> str:
        r, g, b = bg_color.red(), bg_color.green(), bg_color.blue()
        brightness = (0.299 * r + 0.587 * g + 0.114 * b)
        return "#000000" if brightness > 186 else "#FFFFFF"
    
    def apply_interactive_cursor(self, widget):
        widget.setCursor(Qt.CursorShape.PointingHandCursor)

    def eventFilter(self, obj, event):
        if obj == self.rename_button:
            if event.type() == event.Type.Enter:
                if not self.rename_button.isEnabled():
                    self.rename_button.setCursor(Qt.CursorShape.ArrowCursor)
                else:
                    self.rename_button.setCursor(Qt.CursorShape.PointingHandCursor)
        return super().eventFilter(obj, event)




class EPGSettingsTab(QWidget):
    pass



class SidebarButton(QToolButton):
    def __init__(self, text: str, index: int, icon_path: str = None, parent=None):
        super().__init__(parent)
        self.index = index
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        if icon_path:
            self.setIcon(QIcon(icon_path))
            self.setIconSize(QSize(32, 32))
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        else:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        self.setStyleSheet("""
            QToolButton {
                background-color: #2b2b2b;
                color: white;
                padding: 6px 10px 6px 10px;
                text-align: left;
                border: none;
                font-size: 10pt;
                font-weight: normal;
            }
            QToolButton:hover {
                background-color: rgba(32, 147, 254, 0.15);
            }
            QToolButton:pressed {
                background-color: rgba(32, 147, 254, 0.35);
            }
            QToolButton:checked {
                background-color: rgba(32, 147, 254, 0.25);
                border-left: 4px solid #2093FE;
                padding-left: 6px;
                font-weight: 600;
            }
        """)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setMaximumSize(600, 600)

        self.settings = QSettings("USDA", "SCIDO")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar_frame = QFrame()
        self.sidebar_frame.setFixedWidth(180)
        self.sidebar_frame.setStyleSheet("background-color: #2b2b2b;")
        self.sidebar_layout = QVBoxLayout(self.sidebar_frame)
        self.sidebar_layout.setContentsMargins(0, 12, 0, 0)
        self.sidebar_layout.setSpacing(0)

        button_info = [
            ("Appearance", None),  # Use valid icon paths or leave empty
            ("EPG Settings", None)
        ]

        self.buttons = []
        for index, (label, icon) in enumerate(button_info):
            btn = SidebarButton(label, index, icon)
            btn.clicked.connect(lambda _, i=index: self.switch_tab(i))
            self.sidebar_layout.addWidget(btn)
            self.buttons.append(btn)

        self.sidebar_layout.addStretch()
        self.buttons[0].setChecked(True)

        self.stack = QStackedWidget()

        self.appearance_tab = AppearanceTab(self.stack)
        self.stack.addWidget(self.appearance_tab)

        self.stack.addWidget(self._create_test_tab())

        main_layout.addWidget(self.sidebar_frame)
        main_layout.addWidget(self.stack)

    def switch_tab(self, index: int):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)

    def _create_test_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel("EPG Settings")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return tab
    
    def save_setting(self, key: str, value):
        if isinstance(value, QColor):
            value = value.name()  # store as hex string
        print(f"Setting {key} to {value}.")
        self.settings.setValue(key, value)

    def save_settings(self):
        for key in Settings.saved_settings.keys():
            self.save_setting(key, getattr(Settings, key))
        self.settings.sync()   
        
    def load_settings(self):
        if not self.settings.allKeys():
            print("No settings file found, using default values.")
            return
        
        for key, expected_type in Settings.saved_settings.items():
            if expected_type is QColor:
                color_str = self.settings.value(key, getattr(Settings, key).name())
                value = QColor(color_str)
            else:
                value = self.settings.value(key, getattr(Settings, key), type=expected_type)
            setattr(Settings, key, value)
        
    def reset_settings(self):
        self.settings.clear()
        self.settings.sync()

    