import numpy as np
import json
import threading
from queue import Empty


from PyQt6.QtCore import Qt, QSize, QMetaObject, Q_ARG
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QToolButton, QHBoxLayout, QVBoxLayout, QLabel
)

from live_view.LiveDataWindow import LiveDataWindow
from live_view.SliderPanel import SliderPanel
from live_view.socket.ConnectionIndicator import ConnectionIndicator
from live_view.socket.EPGSocket import SocketClient, SocketServer


class LiveViewTab(QWidget):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.connection_indicator = ConnectionIndicator()
        # === Socket ===
        self.socket_server = SocketServer()
        self.socket_server.start()

        self.socket_client = SocketClient(client_id='CS', parent=self)
        self.socket_client.peerConnectionChanged.connect(self.connection_indicator.set_connected)
        self.socket_client.peerConnectionChanged.connect(self.update_button_state)
        self.socket_client.connect()

        self.receive_loop = threading.Thread(target=self._socket_recv_loop, daemon=True)
        self.receive_loop.start()

        self.datawindow = LiveDataWindow(self, settings=settings)
        self.datawindow.getPlotItem().hideButtons()

        self.pause_button = QPushButton("Pause Live View", self)
        self.pause_button.setCheckable(True)
        self.pause_button.setChecked(True)
        self.pause_button.setCursor(Qt.CursorShape.PointingHandCursor)
        # TODO: CHANGE COLOR OF BUTTONS
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #379acc;
                color: white;
                border-radius: 3px;
                padding: 5px;
                outline: none;
            } QPushButton:checked {
                background-color: gray;
                color: white;
                border-radius: 3px;
                padding: 5px;
                outline: none;
            } QPushButton:disabled {
                background-color: gray;
                color: white;
                border-radius: 3px;
                padding: 5px;
                outline: none;
            } QPushButton:focus {
                border: 3px solid #4aa8ff;
                padding: 2px;
            }
        """)
        self.pause_button.clicked.connect(self.toggle_live)

        self.add_comment_button = QPushButton("Add Comment", self)
        self.add_comment_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_comment_button.setToolTip("Add Comment at Current Time")
        # TODO: CHANGE COLOR OF BUTTONS
        self.add_comment_button.setStyleSheet("""
            QPushButton {
                background-color: gray;
                color: white;
                border-radius: 3px;
                padding: 5px;
                outline: none;
            } QPushButton:disabled {
                background-color: gray;
                color: white;
                border-radius: 3px;
                padding: 5px;
                outline: none;
            } QPushButton:focus {
                border: 3px solid #4aa8ff;
                padding: 2px;
            }
        """)
        self.add_comment_button.clicked.connect(self.call_add_comment)
        
        self.pause_button.setCheckable(True)
        self.pause_button.setChecked(True)
        self.pause_button.setToolTip("Pause Live View")
        self.pause_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_button.clicked.connect(self.toggle_live)


        self.slider_panel = SliderPanel(parent=self)
        self.slider_panel.off_button.clicked.connect(self.end_recording)
        self.slider_button = QToolButton(parent=self)
        self.slider_button.setText("EPG Controls")
        self.slider_button.setIcon(QIcon("icons/sliders.svg"))
        self.slider_button.setIconSize(QSize(24, 24))
        self.slider_button.setToolTip("Open control sliders")
        self.slider_button.setAutoRaise(True)
        self.slider_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.slider_button.clicked.connect(self.toggleSliders)
        # TODO: CHANGE COLOR OF BUTTONS
        self.slider_button.setStyleSheet("""
            QToolButton {
                outline: none;
            } QToolButton:disabled {
                background-color: gray;
            }
            QToolButton:focus {
                outline 3px solid #4aa8ff;
            }
        """)
        self.slider_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.update_button_state(False)
        

        top_controls = QHBoxLayout()
        top_controls.addWidget(self.pause_button)
        top_controls.addWidget(self.add_comment_button)
        top_controls.addStretch()  # push slider button to right
        top_controls.addWidget(self.connection_indicator)
        top_controls.addWidget(self.slider_button)

        left_layout = QVBoxLayout()
        left_layout.addLayout(top_controls)
        left_layout.addWidget(self.datawindow)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 4)
        main_layout.addWidget(self.slider_panel, 1)

        # can't figure out the 2 random tabs --> this logic below doesnt work either
        # self.setTabOrder(self.pause_button, self.add_comment_button)
        # self.setTabOrder(self.add_comment_button, self.slider_button)
        # self.setTabOrder(self.slider_button, self.pause_button)

        self.setLayout(main_layout)

    def toggleSliders(self):
        if not self.slider_button.isEnabled():
            return
        is_visible = self.slider_panel.isVisible()
        self.slider_panel.setVisible(not is_visible)

        if is_visible:
            self.slider_button.setToolTip("Open control sliders")
        else:
            self.slider_button.setToolTip("Hide control sliders")

    def update_button_state(self, is_connected: bool):
        """
        Handles disabling the slider, live view, and add comment button
        when the EPG is not connected.
        """      
        self.pause_button.setEnabled(is_connected)
        self.add_comment_button.setEnabled(is_connected)
        self.slider_button.setEnabled(is_connected)

        if is_connected:
            self.pause_button.setToolTip("Pause Live View")
            self.add_comment_button.setToolTip("Add Comment at Current Time")
            self.slider_button.setToolTip("Open control sliders")
        else:
            self.pause_button.setToolTip("Connect to EPG to enable live mode")
            self.add_comment_button.setToolTip("Connect to EPG to enable commenting")
            self.slider_button.setToolTip("Connect to EPG to enable controls")

        self.slider_panel.hide()

    def toggle_live(self):
        live_mode = self.pause_button.isChecked()

        self.pause_button.setText("Pause Live View" if live_mode else "Live View")
        self.datawindow.set_live_mode(live_mode)

    def call_add_comment(self):
        self.datawindow.add_comment_live()

    def end_recording(self):
        self.datawindow.live_mode = False
        self.pause_button.setToolTip("Recording has ended")
        self.pause_button.setChecked(False)
        self.pause_button.setEnabled(False)

        self.add_comment_button.setToolTip("Recording has ended")
        self.add_comment_button.setEnabled(False)

    def _socket_recv_loop(self):

        acknowledged = False # whether the client has been acknowledged by the server
        while self.socket_client.connected:
            try:
                # NOTE: message can include multiple commands/data, i.e. "{<command1>}\n{<command2>}\n"
                raw_message = self.socket_client.recv_queue.get(timeout=1.0)

                if not acknowledged:
                    if raw_message.strip() == "ack":
                        acknowledged = True
                    continue

                self.datawindow.live_mode = True
                
                # parse message into individual commands
                if isinstance(raw_message, dict):
                    messages = [raw_message]
                else:
                    # Multiple newline-separated JSON strings
                    message_list = raw_message.strip().split("\n")
                    messages = [
                        json.loads(s) for s in message_list if s.strip()
                    ]
                
                for message in messages:
                    if message["source"] == self.socket_client.client_id:
                        continue
                    
                    message_type = message['type']

                    if message_type == 'data':
                        time = float(message['value'][0])
                        volt = float(message['value'][1])

                        with self.datawindow.buffer_lock:
                            self.datawindow.buffer_data.append((time, volt))
                            
                        # update latest time input
                        self.datawindow.current_time = time

                    elif message_type == "control":
                        name = message["name"]
                        value = message["value"]
                        source = message.get("source")

                        # Workaround to get set_control_value to run in the GUI thread
                        # Might be cleaner to use signals, but this works for now
                        QMetaObject.invokeMethod(
                            self.slider_panel,
                            "set_control_value",
                            Qt.ConnectionType.QueuedConnection,
                            Q_ARG(str, name),
                            Q_ARG(object, value),
                            Q_ARG(str, source)
                        )

                    elif message_type == "state_sync":
                        value = message["value"]

                        QMetaObject.invokeMethod(
                            self.slider_panel,
                            "set_all_controls",
                            Qt.ConnectionType.QueuedConnection,
                            Q_ARG(dict, value),
                        )
            except Empty:
                self.datawindow.live_mode = False
                continue  # restart the loop

            except Exception as e:
                self.datawindow.live_mode = False
                print("[CS RECIEVE LOOP ERROR]", e)

   

