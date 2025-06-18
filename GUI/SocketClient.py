import socket
import threading
import queue
import json


import sys
import logging

# Log output to console even if running in background thread
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%Y-%m-%d | %H:%M:%S]",
    #format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)


class SocketClient:
    def __init__(self, client_id, host="localhost", port=16671):
        self.host: str = host                   # use "localhost" for interal socket
        self.port: int = port                   # arbitrary port
        self.client_id: str = client_id         # identifying string for this client (e.g., CS, ENGR) 
        self.send_queue: queue = queue.Queue()  # queue to send data to other client
        self.recv_queue: queue = queue.Queue()  # queue to receive data from other client
        self.running: bool = False              # whether the socket is running
        self._sock: socket.socket = None        # the socket connection

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))

        # send initial message with client ID to socket
        self._sock.sendall(f"client_id={self.client_id}\n".encode('utf-8'))
        self.running = True

        threading.Thread(target=self._send_loop, daemon=True).start()
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def send(self, data: dict):
        self.send_queue.put_nowait(data)

    def receive(self):
        try:
            return self.recv_queue.get_nowait()
        except queue.Empty:
            return None

    def _send_loop(self):
        while self.running:
            try:
                msg = self.send_queue.get(timeout=0.1)
                json_str = json.dumps(msg) + "\n"
                self._sock.sendall(json_str.encode("utf-8"))
            except queue.Empty:
                continue
            except Exception as e:
                logging.info("[SocketClient SEND ERROR]", e)
                self.running = False
                break

    def _recv_loop(self):
        while self.running:
            try:
                data = self._sock.recv(1024)
                if data:
                    self.recv_queue.put_nowait(data.decode("utf-8"))
            except Exception as e:
                logging.info("[SocketClient RECV ERROR]", e)  
                self.running = False  
                break