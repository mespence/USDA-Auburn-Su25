import socket
import threading
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

class SocketServer:
    def __init__(self, host = "localhost", port=16671):
        self.host: str = host                        # use "localhost" for interal socket
        self.port: int = port                        # arbitrary port
        self.clients: dict[str, socket.socket] = {}  # map of client IDs to their connection objects
        self.running = False                         # whether the server is running
        self.ready_event = threading.Event()         # event to signal that the server is ready to receive connections
        self._server_socket: socket.socket = None    # the socket connection

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
        logging.info("[SERVER] Shutting down...")

        # Close client connections
        for client_id, conn in self.clients.items():
            if conn:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                    conn.close()
                    logging.info(f"[SERVER] Disconnected {client_id}")
                except Exception as e:
                    logging.info(f"[SERVER] Error closing {client_id}: {e}")

        self.clients = {}

        # Close server socket
        if self._server_socket:
            try:
                self._server_socket.close()
                logging.info("[SERVER] Socket closed")
            except Exception as e:
                logging.info(f"[SERVER] Error closing server socket: {e}")

        logging.info("[SERVER] Shutdown complete.")

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
            logging.info(f"[SERVER] Listening on {self.host}:{self.port}")

            while self.running:
                try:
                    conn, addr = self._server_socket.accept()
                except OSError:
                    break  # socket closed
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
       
    def _handle_client(self, conn: socket.socket, addr):
        """
        Processes new client connections and begins reading messages from it.
        """
        try:
            initial_message = conn.recv(1024).decode().strip()
            client_id = initial_message.split("=")[1]  
            self.clients[client_id] = conn
            logging.info(f"[SOCKET] Client [{client_id}] connected from {addr}")
            self._receive_loop(conn, client_id)
        except Exception as e:
            logging.info(f"[SERVER ERROR] {addr}: {e}")
        finally:
            conn.close()
            logging.info(f"[SERVER] {addr} disconnected")

    def _receive_loop(self, conn: socket.socket, client_id: str):
        """
        Reads newline-delimited JSON messages from the given client connection
        and dispatches them to the appropriate handler.
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
            logging.info("[SERVER] CS not connected, can't forward data.")
            return

        data_list = data["value"].split(",")
        msg = {
            "type": "data",
            "value": (data_list[0], data_list[2])  # (timestamp, voltage)
        }
        cs_conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))


if __name__ == "__main__":
    socket_server = SocketServer()
    socket_server.start()

    input("Press Enter to quit...\n")




# HOST = 'localhost'
# PORT = 16671


# clients: dict[str, socket.socket] = {
#     "ENGR": None,
#     "CS": None,
# }

# def slider_listener(conn):
#     """
#     Background thread: receive control messages (e.g., slider updates).
#     """
#     while True:
#         try:
#             data = conn.recv(1024)
#             if not data:
#                 print("Sliders disconnected.")
#                 break
#             print("[CONTROL]", data.decode().strip())
#         except ConnectionResetError:
#             print("Client forcibly closed the connection.")
#             break
#         except Exception as e:
#             print("Error in control thread:", e)
#             break     


# def handle_client(conn, addr):

#     initial_data = conn.recv(1024).decode().strip()
#     client_id = initial_data.split("=")[1]

#     clients[client_id] = conn
#     print(f"[SERVER] Connected to {client_id} from {addr}")

#     buffer = ""

#     try:
#         while True:
#             chunk = conn.recv(1024)
#             if not chunk:
#                 break  # client disconnected

#             buffer += chunk.decode()

#             # Split buffer into complete lines
#             while "\n" in buffer:
#                 line, buffer = buffer.split("\n", 1)
#                 process_message(line.strip(), client_id)

#     except Exception as e:
#         print(f"[ERROR] {addr}: {e}")
#     finally:
#         conn.close()
#         print(f"[DISCONNECTED] {addr}")

# def process_message(message: str, client_id: str):
#     try:
#         msg = json.loads(message)

#         if msg["type"] == "data":
#             send_stream(msg)

#         elif msg["type"] == "control":
#             print(f"[{client_id}] control: {msg["control_type"]} = {msg["value"]}")

#         else:
#             print(f"[{client_id}] UNKNOWN TYPE: {msg}")

#     except json.JSONDecodeError:
#         print(f"[{client_id}] INVALID JSON: {message}")



# def send_stream(message_dict: dict):
#     """
#     Forwards data from the engr client to the cs client.
#     """
#     cs_conn = clients["CS"]

#     if cs_conn is None:
#         print("[SERVER] CS client not connected, cannot forward data.")
#         return  # CS client not connected
    
#     if message_dict["type"] == "data":
#         data_list = message_dict["value"].split(",") # [timestamp, DATA, voltage, channel]
#         message = {"type": "data", "value": (data_list[0], data_list[2])}
#         json_str = json.dumps(message) + "\n"

#         cs_conn.sendall(json_str.encode("utf-8"))
    

# def main():

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen(2)
#         print(f"[SERVER] Listening on {HOST}:{PORT}...")

#         while True:
#             conn, addr = s.accept()

#             threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start() # client handler

#             # slider_thread = threading.Thread(target=slider_listener, args=(conn,), daemon = True)
#             # slider_thread.start()

# if __name__ == "__main__":
#     main()
