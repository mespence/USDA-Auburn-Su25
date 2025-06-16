import socket
import threading
import queue
import json

class SocketClient:
    def __init__(self, host="localhost", port=16671):
        self.host: str = host  # use "localhost" for interal socket
        self.port: int = port  # arbitrary port
        self.send_queue: queue = queue.Queue()  # queue to send data to CS
        self.recv_queue: queue = queue.Queue()  # queue to receive data from CS
        self._sock: socket.socket = None  # the socket connection
        self._running: bool = False  # whether the socket is running

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        self._sock.sendall(b"client_id=ENGR\n")
        self._running = True

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
        while self._running:
            try:
                msg = self.send_queue.get(timeout=0.1)
                json_str = json.dumps(msg) + "\n"
                self._sock.sendall(json_str.encode("utf-8"))
            except queue.Empty:
                continue
            except Exception as e:
                print("[SocketClient SEND ERROR]", e)
                break

    def _recv_loop(self):
        while self._running:
            try:
                data = self._sock.recv(1024)
                if data:
                    self.recv_queue.put_nowait(data.decode("utf-8"))
            except Exception as e:
                print("[SocketClient RECV ERROR]", e)
                break