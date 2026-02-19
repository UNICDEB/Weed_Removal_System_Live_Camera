import serial
import serial.tools.list_ports
import threading
import time


class ArduinoManager:

    def __init__(self):
        self.ser = None
        self.connected = False
        self.port = None
        self.baudrate = 9600
        self.last_response = ""
        self.lock = threading.Lock()

    # ------------------------------

    def auto_detect(self):
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if "Arduino" in p.description:
                return p.device
        return None

    # ------------------------------

    def connect(self):

        if self.connected:
            return

        try:
            self.port = self.auto_detect()

            if self.port is None:
                print("Arduino not found")
                return

            self.ser = serial.Serial(self.port, self.baudrate, timeout=0)
            time.sleep(2)

            self.connected = True
            print("Arduino Connected:", self.port)

        except Exception as e:
            print("Arduino connection failed:", e)
            self.connected = False

    # ------------------------------

    def send_coordinate(self, x, y, z):

        if not self.connected:
            return  # 🔥 DO NOT auto connect here

        try:
            msg = f"{x:.2f},{y:.2f},{z:.2f}\n"
            with self.lock:
                self.ser.write(msg.encode())
        except:
            self.connected = False

    # ------------------------------

    def send_command(self, cmd):

        if not self.connected:
            return

        try:
            with self.lock:
                self.ser.write((cmd + "\n").encode())
        except:
            self.connected = False

    # ------------------------------

    def read_feedback(self):

        if not self.connected:
            return ""

        try:
            if self.ser.in_waiting:
                self.last_response = self.ser.readline().decode().strip()
        except:
            self.connected = False

        return self.last_response


arduino_manager = ArduinoManager()
