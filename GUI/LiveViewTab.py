import numpy as np
import json
import threading
from queue import Empty


from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QToolButton, QHBoxLayout, QVBoxLayout
)


from LiveDataWindow import LiveDataWindow
from SliderPanel import SliderPanel
from EPGSocket import SocketClient, SocketServer


class LiveViewTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.socket_server = SocketServer()
        self.socket_server.start()

        self.socket_client = SocketClient(client_id='CS')
        self.socket_client.start()


        self.datawindow = LiveDataWindow()
        self.datawindow.getPlotItem().hideButtons()

        self.slider_panel = SliderPanel(parent=self)
        sliderButton = QToolButton(parent=self)
        sliderButton.setText("EPG Controls")
        sliderButton.setIcon(QIcon("icons/sliders.svg"))
        sliderButton.setIconSize(QSize(24, 24))
        sliderButton.setToolTip("Open control sliders")
        sliderButton.setAutoRaise(True)
        sliderButton.setCursor(Qt.CursorShape.PointingHandCursor)
        sliderButton.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        sliderButton.clicked.connect(self.toggleSliders)
        self.slider_panel.hide()

        top_controls = QHBoxLayout()
        top_controls.addStretch()  # push slider button to right
        top_controls.addWidget(sliderButton)

        left_layout = QVBoxLayout()
        left_layout.addLayout(top_controls)
        left_layout.addWidget(self.datawindow)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 4)
        main_layout.addWidget(self.slider_panel, 1)

        self.setLayout(main_layout)


        self.recieve_loop = threading.Thread(target=self._socket_recv_loop, daemon=True)
        self.recieve_loop.start()


    def _socket_recv_loop(self):
        while self.socket_client.running:

            try:
                # can include multiple commands/data in one message
                raw_message = self.socket_client.recv_queue.get(timeout=1.0)
            except Empty:
                continue  # restart the loop

            try:
                # parse message
                message_list = raw_message.split("\n")
                messages = [json.loads(s) for s in message_list if s.strip()]

                if '' in message_list:
                    message_list.remove('')

                for message in messages:
                    if message['type'] != 'data':
                        # message is for sliders
                        continue

                    time = float(message['value'][0])
                    volt = float(message['value'][1])

                    # this copies the np array each time to append so O(n)
                    xy_data = self.datawindow.xy_data
                    xy_data[0] = np.append(xy_data[0], time)
                    xy_data[1] = np.append(xy_data[1], volt)   

                    # update latest time input
                    self.datawindow.current_time = time
                self.datawindow.update_plot()

            except Exception as e:
                print("[RECIEVE LOOP ERROR]", e)

    def toggleSliders(self):
        self.slider_panel.setVisible(not self.slider_panel.isVisible())



