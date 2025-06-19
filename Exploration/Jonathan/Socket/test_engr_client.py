import socket
import threading
import queue
import time
import json

import csv


recv_queue = queue.Queue()
send_queue = queue.Queue()

running = True


def device_simulation():
    """
    Simulates a device reading (time, voltage) data every 0.01s
    """
    #DATA_FILE = r"C:\EPG-Project\Summer\CS-Repository\Exploration\Jonathan\Data\smooth_18mil.csv"
    DATA_FILE = r"C:\EPG-Project\Summer\CS-Repository\Exploration\Jonathan\Data\sharpshooter_label2.csv"
    with open(DATA_FILE, newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip header row
        interval = 0.01
        next_time = time.perf_counter()

        # simulate sending data every 0.01s, 
        # taking into account execution time
        for row in reader:
            data_row = f"{float(row[4]):.4f},DATA,{float(row[5]):.4f},0\n"
            data_dict = {"type":"data", "value":data_row}
            send_queue.put_nowait(data_dict)
            next_time += interval
            sleep_time = max(0, next_time - time.perf_counter())
            time.sleep(sleep_time)



send_count = 0
start_time = time.time()

def send_queue_to_socket(sock: socket.socket):
    """
    Passes data in the send queue to the socket.
    """
    global send_count, start_time, running

    while running:
        try:
            data_dict = send_queue.get(timeout=0.1)
            json_str = json.dumps(data_dict) + "\n"
            sock.sendall(json_str.encode("utf-8"))

            send_count += 1
            now = time.time()
            if now - start_time >= 1.0:
                print(f"[ENGR] Send Rate: {send_count}/sec")
                send_count = 0
                start_time = now

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ENGR ERROR] Failed to send data: {e}")
            running = False
            break

def socket_to_recv_queue(sock: socket.socket):
    """
    Passes received data in the socket to the receive queue.
    """
    global running
    while running:
        try:
            data = sock.recv(1024)
            if data:
                recv_queue.put_nowait(data.decode("utf-8"))
        except Exception as e:
            print(f"[SOCKET ERROR] {e}")
            running = False

            

def socket_client():
    global running
    HOST = "localhost"
    PORT = 16671  # arbitrary
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print("Connecting...")
        sock.connect((HOST, PORT))
        sock.sendall(b"client_id=ENGR\n")
        print(f"Connected to {HOST}:{PORT}")


        device_sim_thread = threading.Thread(target=device_simulation, daemon=True)
        device_sim_thread.start()

        send_queue_thread = threading.Thread(target=send_queue_to_socket, args=(sock,), daemon=True)
        send_queue_thread.start()

        recv_queue_thread = threading.Thread(target=socket_to_recv_queue, args=(sock,), daemon=True)
        recv_queue_thread.start()

        try:
            while running:
                time.sleep(0.1)  # keep client open
        except KeyboardInterrupt:
            print("[ENGR] Interrupted by user.")
            running = False
        


if __name__ == "__main__":
    socket_client()
    

# socket_thread = threading.Thread(target=socket_client)
# socket_thread.start()