import socket
import time
import threading
import json


HOST = "localhost"  
PORT = 16671     

clients: dict[str, socket.socket] = {
    "ENGR": None,
    "CS": None,
}

def slider_listener(conn):
    """
    Background thread: receive control messages (e.g., slider updates).
    """
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                print("Sliders disconnected.")
                break
            print("[CONTROL]", data.decode().strip())
        except ConnectionResetError:
            print("Client forcibly closed the connection.")
            break
        except Exception as e:
            print("Error in control thread:", e)
            break     


def handle_client(conn, addr):

    initial_data = conn.recv(1024).decode().strip()
    client_id = initial_data.split("=")[1]

    clients[client_id] = conn
    print(f"[SERVER] Connected to {client_id} from {addr}")

    buffer = ""

    try:
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                break  # client disconnected

            buffer += chunk.decode()

            # Split buffer into complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                process_message(line.strip(), client_id)

    except Exception as e:
        print(f"[ERROR] {addr}: {e}")
    finally:
        conn.close()
        print(f"[DISCONNECTED] {addr}")

def process_message(message: str, client_id: str):
    try:
        msg = json.loads(message)

        if msg["type"] == "data":
            send_stream(msg)

        elif msg["type"] == "control":
            print(f"[{client_id}] control: {msg["control_type"]} = {msg["value"]}")

        else:
            print(f"[{client_id}] UNKNOWN TYPE: {msg}")

    except json.JSONDecodeError:
        print(f"[{client_id}] INVALID JSON: {message}")



def send_stream(message_dict: dict):
    """
    Forwards data from the engr client to the cs client.
    """
    cs_conn = clients["CS"]

    if cs_conn is None:
        print("[SERVER] CS client not connected, cannot forward data.")
        return  # CS client not connected
    
    if message_dict["type"] == "data":
        data_list = message_dict["value"].split(",") # [timestamp, DATA, voltage, channel]
        message = {"type": "data", "value": (data_list[0], data_list[2])}
        json_str = json.dumps(message) + "\n"

        cs_conn.sendall(json_str.encode("utf-8"))
    

def main():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(2)
        print(f"[SERVER] Listening on {HOST}:{PORT}...")

        while True:
            conn, addr = s.accept()

            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start() # client handler

            # slider_thread = threading.Thread(target=slider_listener, args=(conn,), daemon = True)
            # slider_thread.start()

if __name__ == "__main__":
    main()
