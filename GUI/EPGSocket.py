import socket
import threading
import json
import queue
import time
import sys
import logging

# Log output to console even if running in background thread
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%Y-%m-%d | %H:%M:%S]",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

class SocketServer:
    """
    A bidirectional socket to connect the CS and ENGR UIs.
    Forwards EPG data and slider control events beteween the clients. 
    """
    def __init__(self, host = "localhost", port=16671):
        self.host: str = host                           # use "localhost" for interal socket
        self.port: int = port                           # arbitrary port
        self.clients: dict[str, socket.socket] = {}     # map of client IDs to their connection objects
        self.running = False                            # whether the server is running
        self.ready_event = threading.Event()            # event to signal that the server is ready to receive connections
        self._server_socket: socket.socket = None       # the socket connection
        self._current_time:float = time.perf_counter()  # time tracker used in logging

    def start(self):
        """
        Starts the server in a background thread, listening for incoming client connections.
        """
        if self.running:
            return
        self.running = True
        threading.Thread(target = self._listen, daemon = True).start()
        self.ready_event.wait()  # wait for server to fully initialize

    def stop(self):
        """
        Stops the server, closes all sockets, and disconnects any clients.
        """
        self.running = False
        logging.info("[SOCKET] Shutting down...")

        # Close client connections
        for client_id, conn in self.clients.items():
            if conn:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                    conn.close()
                    logging.info(f"[SOCKET] Disconnected {client_id}")
                except Exception as e:
                    logging.info(f"[SOCKET] Error closing {client_id}: {e}")

        self.clients = {}

        # Close server socket
        if self._server_socket:
            try:
                self._server_socket.close()
                logging.info("[SOCKET] Socket closed")
            except Exception as e:
                logging.info(f"[SOCKET] Error closing socket: {e}")

        logging.info("[SOCKET] Shutdown complete")

    def _listen(self):
        """
        Internal loop that binds the server socket and accepts new connections.
        Spawns a new thread for each connected client.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            self._server_socket = server_sock
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(2)
            self.ready_event.set()
            logging.info(f"[SOCKET] Listening on {self.host}:{self.port}")

            while self.running:
                try:
                    conn, addr = self._server_socket.accept()
                    threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
                except OSError:
                    break  # socket closed
       
    def _handle_client(self, conn: socket.socket, addr):
        """
        Processes new client connections and begins reading messages from it.
        """
        try:
            initial_message = conn.recv(1024).decode().strip()
            client_id = initial_message.split("=")[1]  
            self.clients[client_id] = conn
            logging.info(f"[SOCKET] Client \"{client_id}\" connected from {addr}")
            self._receive_loop(conn, client_id)
        except:
            pass
        # except Exception as e:
        #     name = self.clients.get(addr, addr) # default to addr if no client_id
        #     logging.info(f"[SERVER ERROR] \"{name}\": {e}")
        finally:
            conn.close()
            # name = self.clients.get(addr, addr) # default to addr if no client_id
            # logging.info(f"[SERVER] {name} disconnected")

    def _receive_loop(self, conn: socket.socket, client_id: str):
        """
        Backgroung loop to read newline-delimited JSON messages from the given client connection
        and dispatch them to the appropriate handler.
        """
        buffer = ""
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                break  # client disconnected
            buffer += chunk.decode()
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                self._process_message(line.strip(), client_id)


    def _process_message(self, message: str, client_id: str):
        """
        Processes a single JSON-formatted message from a client.
        Delegates to control or data handlers based on message type.
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logging.info(f"[SOCKET] Invalid JSON from {client_id}: {message}")
            return
        
        if data["type"] == "data":
            self._forward_data(data)
        elif data["type"] == "control":
            logging.info(f"[{client_id}] Control: {data['control_type']} = {data['value']}")
        else:
            logging.info(f"[{client_id}] Unknown message type: {data['type']}")
    
    def _forward_data(self, data: dict):
        """
        Forwards a simplified (timestamp, voltage) message from ENGR to CS.
        If CS is not connected, logs a warning.
        """
        cs_conn = self.clients.get("CS")
        if not cs_conn:
            if (time.perf_counter() - self._current_time) > 1: # only send every 1s
                logging.info("[SOCKET] CS not connected, can't forward data.")
                self._current_time = time.perf_counter()
            return

        data_list = data["value"].split(",")
        msg = {
            "type": "data",
            "value": (data_list[0], data_list[2])  # (timestamp, voltage)
        }
        cs_conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))


class SocketClient:
    """
    A client class to connect to the socket and handle sending/receiving data it.
    Incoming messages are pulled from the recieve queue, and outgoing messages are placed in the send queue.
    """
    def __init__(self, client_id, host="localhost", port=16671):
        self.host: str = host                   # use "localhost" for interal socket
        self.port: int = port                   # arbitrary port
        self.client_id: str = client_id         # identifying string for this client (e.g., CS, ENGR) 
        self.send_queue: queue = queue.Queue()  # queue to send data to other client
        self.recv_queue: queue = queue.Queue()  # queue to receive data from other client
        self.running: bool = False              # whether the socket is running
        self._sock: socket.socket = None        # the socket connection

    def start(self):
        """
        Starts the client and initializes the connection to the socket.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))

        # send initial message with client ID to socket
        self._sock.sendall(f"client_id={self.client_id}\n".encode('utf-8'))
        self.running = True

        threading.Thread(target=self._send_loop, daemon=True).start()
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def send(self, data: dict):
        """
        Places a JSON-formatted dictionary in the send queue.
        """
        self.send_queue.put_nowait(data)

    def receive(self):
        """
        Gets a JSON-formatted dictionary from the receive queue if it's non-empty.
        """
        try:
            return self.recv_queue.get_nowait()
        except queue.Empty:
            return None

    def _send_loop(self):
        """
        Background loop to handle outgoing messages.
        """
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
        """
        Background loop to handle incoming messages.
        """
        while self.running:
            try:
                data = self._sock.recv(1024)
                if data:
                    self.recv_queue.put_nowait(data.decode("utf-8"))
            except Exception as e:
                logging.info("[SocketClient RECV ERROR]", e)  
                self.running = False  
                break