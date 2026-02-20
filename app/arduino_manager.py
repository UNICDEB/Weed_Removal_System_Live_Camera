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

        # If already connected and port open → do nothing
        if self.connected and self.ser and self.ser.is_open:
            return

        try:
            # Auto detect Arduino port
            self.port = self.auto_detect()

            if self.port is None:
                print("❌ Arduino not found")
                self.connected = False
                return

            # Close old connection safely
            if self.ser and self.ser.is_open:
                self.ser.close()

            # Open serial
            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=0.1   # 🔥 small timeout (important)
            )

            time.sleep(2)  # Allow Arduino reset

            self.connected = True
            print("✅ Arduino Connected:", self.port)

            # 🔥 Start background reader
            self.start_reader_thread()

        except Exception as e:
            print("❌ Arduino connection failed:", e)
            self.connected = False

    # --------------------
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

    # ---------------
    def start_reader_thread(self):

        def read_loop():
            print("🔄 Arduino reader thread started")

            while self.connected:
                try:
                    if self.ser.in_waiting:
                        self.last_response = self.ser.readline().decode().strip()
                        print("📥 Arduino:", self.last_response)
                except:
                    self.connected = False
                    print("⚠ Arduino disconnected")
                    break

                time.sleep(0.05)

        threading.Thread(target=read_loop, daemon=True).start()

    def send_raw(self, command):

        if self.connected:
            try:
                self.ser.write((command + "\n").encode())
                print("Sent to Arduino:", command)
            except Exception as e:
                print("Arduino Send Error:", e)
                self.connected = False




arduino_manager = ArduinoManager()
