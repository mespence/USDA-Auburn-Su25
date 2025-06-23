import numpy as np
import json
import threading
from queue import Empty


from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QToolButton, QHBoxLayout, QVBoxLayout, QLabel
)


from LiveDataWindow import LiveDataWindow
from ConnectionIndicator import ConnectionIndicator
from SliderPanel import SliderPanel
from EPGSocket import SocketClient, SocketServer


class LiveViewTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.connection_indicator = ConnectionIndicator()

        self.socket_server = SocketServer()
        self.socket_server.engrConnected.connect(self.connection_indicator.set_connected)
        self.socket_server.start()

        self.socket_client = SocketClient(client_id='CS')
        self.socket_client.start()


        self.datawindow = LiveDataWindow()
        self.datawindow.getPlotItem().hideButtons()

        self.pause_button = QPushButton("Pause Live View", self)
        self.pause_button.setStyleSheet("""
            background-color: gray;
            color: white;
            border-radius: 3px;
            padding: 5px;
        """)
        self.pause_button.setCheckable(True)
        self.pause_button.setChecked(True)
        self.pause_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_button.clicked.connect(self.toggleLive)


        


        self.slider_panel = SliderPanel(parent=self)
        self.slider_button = QToolButton(parent=self)
        self.slider_button.setText("EPG Controls")
        self.slider_button.setIcon(QIcon("icons/sliders.svg"))
        self.slider_button.setIconSize(QSize(24, 24))
        self.slider_button.setToolTip("Open control sliders")
        self.slider_button.setAutoRaise(True)
        self.slider_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.slider_button.clicked.connect(self.toggleSliders)
        self.slider_panel.hide()

        top_controls = QHBoxLayout()
        top_controls.addWidget(self.pause_button)
        top_controls.addStretch()  # push slider button to right
        top_controls.addWidget(self.connection_indicator)
        top_controls.addWidget(self.slider_button)

        left_layout = QVBoxLayout()
        left_layout.addLayout(top_controls)
        left_layout.addWidget(self.datawindow)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 4)
        main_layout.addWidget(self.slider_panel, 1)

        self.setLayout(main_layout)


        self.recieve_loop = threading.Thread(target=self._socket_recv_loop, daemon=True)
        self.recieve_loop.start()

    def toggleSliders(self):
        is_visible = self.slider_panel.isVisible()
        self.slider_panel.setVisible(not is_visible)

        if is_visible:
            self.slider_button.setToolTip("Open control sliders")
        else:
            self.slider_button.setToolTip("Hide control sliders")


    def toggleLive(self):
        live_mode = self.pause_button.isChecked()

        self.pause_button.setText("Pause Live View" if live_mode else "Live View")

        if live_mode:
            self.pause_button.setStyleSheet("""background-color: gray;
                                        color: white;
                                        border-radius: 3px;
                                        padding: 5px;""")
        else:
            self.pause_button.setStyleSheet("""background-color: #379acc;
                                        color: white;
                                        border-radius: 3px;
                                        padding: 5px;""")
        
        self.datawindow.set_live_mode(live_mode)


    def _socket_recv_loop(self):
        while self.socket_client.running:

            try:
                # NOTE: message can include multiple commands/data, i.e. "{<command1>}\n{<command2>}\n"
                raw_message = self.socket_client.recv_queue.get(timeout=1.0)
            except Empty:
                continue  # restart the loop

            try:
                # parse message into individual commands
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

   

