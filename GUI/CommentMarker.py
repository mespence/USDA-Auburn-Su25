from pyqtgraph import (
    PlotWidget, InfiniteLine, mkPen
)
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QLabel, QDialog, QTextEdit
from PyQt6.QtSvgWidgets import QGraphicsSvgItem
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtCore import Qt, QPointF, QTimer

class CommentMarker():
    """
    A class for creating comments with a
    vertical dashed line and comment icon
    """
    def __init__(self, time: float, text: str, datawindow: PlotWidget, icon_path: str = "icons/message.svg"):
        self.time = time
        self.text = text
        self.datawindow = datawindow
        self.scene = self.datawindow.scene()
        self.viewbox = self.datawindow.getPlotItem().getViewBox()
        self.icon_path = icon_path

        self.marker = InfiniteLine(
            pos = self.time,
            angle = 90,
            pen = mkPen('black', style = Qt.PenStyle.DashLine, width=3),
            movable = False,
        )
        self.viewbox.addItem(self.marker)
        
        self.icon_item = QGraphicsSvgItem(self.icon_path)
        self.icon_item.setScale(1)
        self.icon_item.setZValue(10)
        self.scene.addItem(self.icon_item)

        self.update_position()
        self.viewbox.sigTransformChanged.connect(self.update_position) 
        # self.viewbox.sigXRangeChanged.connect(self.update_position)
        self.icon_item.mousePressEvent = self.show_comment_editor

    def update_position(self):
        line_scene_x = self.viewbox.mapViewToScene(QPointF(self.time, 0))
        scene_rect = self.viewbox.sceneBoundingRect()

        icon_x = line_scene_x.x()
        icon_y = scene_rect.bottom() - 25

        self.icon_item.setPos(icon_x, icon_y)
        
        # don't show past viewbox
        icon_scene_rect = self.icon_item.mapRectToScene(self.icon_item.boundingRect())
        icon_scene_x = icon_scene_rect.right()
        icon_right_x = self.viewbox.mapSceneToView(QPointF(icon_scene_x, 0)).x()
        x_min, x_max = self.viewbox.viewRange()[0]
        self.icon_item.setVisible(x_min <= self.time <= x_max and icon_right_x <= x_max)


    def show_comment_editor(self, event: None):
        """
        should be able to edit, delete, and move the comment
        """

        self.datawindow.comment_editing = True

        dialog = QDialog()
        dialog.setWindowTitle(f"Edit Comment @ {self.time:.2f}s")

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Edit Comment:"))

        text_edit = QTextEdit()
        text_edit.setText(self.text)

        cursor = text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        text_edit.setTextCursor(cursor)
        
        layout.addWidget(text_edit)

        save_button = QPushButton("Save")
        move_button = QPushButton("Move")
        delete_button = QPushButton("Delete")
        cancel_button = QPushButton("Cancel")
        layout.addWidget(save_button)
        layout.addWidget(move_button)
        layout.addWidget(delete_button)
        layout.addWidget(cancel_button)

        def save():
            self.set_text(text_edit.toPlainText())
            dialog.accept()

        def move():
            self.set_visible(False)
            dialog.accept()
            # delay move comment so that it doesn't register the dialog mouse press event
            QTimer.singleShot(0, lambda: self.datawindow.move_comment_helper(self))

        def delete():
            dialog.accept()
            self.datawindow.delete_comment(self.time)
        
        save_button.clicked.connect(save)
        move_button.clicked.connect(move)
        delete_button.clicked.connect(delete)
        cancel_button.clicked.connect(dialog.reject)
        dialog.setModal(True)
        dialog.exec()

    def set_text(self, new_text: str):
        self.text = new_text
        df = self.datawindow.epgdata.dfs[self.datawindow.file]
        nearest_idx = (df['time'] - self.time).abs().idxmin()
        df.at[nearest_idx, 'comments'] = new_text

    def set_visible(self, visible: bool):
        self.marker.setVisible(visible)
        self.icon_item.setVisible(visible)

    def remove(self):
        self.viewbox.removeItem(self.marker)
        self.viewbox.removeItem(self.icon_item)

