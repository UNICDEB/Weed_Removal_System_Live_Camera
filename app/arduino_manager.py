import serial
import threading
import time

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
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.connected = True
            print("Arduino Connected")
        except Exception as e:
            print("Arduino Connection Failed:", e)
            self.connected = False

    # ----------------------------------

    def send_coordinate(self, x, y, z):

        if not self.connected:
            self.connect()

        if self.connected:
            try:
                msg = f"{x:.2f},{y:.2f},{z:.2f}\n"
                self.ser.write(msg.encode())
                print("Sent to Arduino:", msg.strip())
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
