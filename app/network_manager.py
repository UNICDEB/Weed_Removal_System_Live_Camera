import socket
import json

# SERVER_IP = "10.110.33.154"
SERVER_IP = "172.18.120.154"
PORT = 5000

class NetworkManager:
    def __init__(self):
        self.client = None
        self.connected = False

    def connect(self):
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((SERVER_IP, PORT))
            self.connected = True
            print("Receiver Connected")
        except:
            self.connected = False

    # def send(self, x, y, z):
    #     if not self.connected:
    #         self.connect()
    #     if self.connected:
    #         try:
    #             data = {"X": x, "Y": y, "Z": z}
    #             self.client.sendall(json.dumps(data).encode())
    #         except:
    #             self.connected = False
    

    def send(self, x, y, z):

        if not self.connected:
            print("Not connected to RPI")
            return

        try:
            data = {"X": x, "Y": y, "Z": z}
            message = json.dumps(data) + "\n"
            self.client.sendall(message.encode())

        except Exception as e:
            print("Send error:", e)
            self.connected = False

network_manager = NetworkManager()
