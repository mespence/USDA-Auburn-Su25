import socket
import threading
import queue
import time

recv_queue = queue.Queue()
send_queue = queue.Queue()

recv_count = 0
start_time = time.time()

def socket_client():
    global recv_count, start_time 

    HOST = 'localhost'
    PORT = 16671
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print('Connecting...')
        sock.connect((HOST, PORT))
        sock.sendall(b"client_id=CS\n")
        print(f'Connected to {HOST}:{PORT}')

        while True:
            # print(f"Send Queue: {send_queue.qsize()}")
            # print(f"Receive Queue: {recv_queue.qsize()}")


            # Read incoming data
            try:
                data = sock.recv(1024)
                if data:
                    decoded = data.decode('utf-8')
                    recv_count += decoded.count('\n')

                    recv_queue.put_nowait(decoded)

                    now = time.time()
                    if now - start_time >= 1.0:
                        print(f"[CS] Receive rate: {recv_count}/sec")
                        recv_count = 0
                        start_time = now
            except BlockingIOError:
                pass

            #TODO:  once data has been plotted to the screen, remove from recv_queue

            # Send queued messages
            try:
                msg = send_queue.get_nowait()
                sock.sendall(msg.encode('utf-8'))
            except queue.Empty:
                pass

            #time.sleep(0.005)  # Small sleep to avoid CPU overuse


if __name__ == "__main__":
    socket_client()
    

# socket_thread = threading.Thread(target=socket_client)
# socket_thread.start()