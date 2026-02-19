import serial
import threading
import time
import socket

class ArduinoManager:

    def __init__(self):
        self.ser = None
        self.connected = False
        self.port = "COM6"      # In my case, if needed then change the com port
        self.baudrate = 9600
        self.last_response = ""

    # ----------------------------------

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.connected = True
            print("RPI Connected")
        except:
            print("Not Connected")
            self.connected = False



    # ----------------------------------

    def send_coordinate(self, x, y, z):

                if not self.connected:
                    self.connect()

                if self.connected:
                    try:
                        msg = f"{x:.2f},{y:.2f},{z:.2f}\n"
                        self.sock.send(msg.encode())
                    except:
                        self.connected = False


    # ----------------------------------

    def send_command(self, cmd):

        if not self.connected:
            self.connect()

        if self.connected:
            try:
                self.ser.write((cmd + "\n").encode())
                print("Command Sent:", cmd)
            except:
                self.connected = False

    # ----------------------------------

    def read_feedback(self):
        if self.connected and self.ser.in_waiting:
            self.last_response = self.ser.readline().decode().strip()
        return self.last_response


arduino_manager = ArduinoManager()
