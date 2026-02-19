import cv2
import threading
import time
from app.detection_manager import DetectionManager

class CameraController:

    def __init__(self):
        self.running = False
        self.detector = None
        self.thread = None

    def start(self, conf):

        if self.running:
            return

        self.running = True

        self.detector = DetectionManager()
        self.detector.set_confidence(conf)

        self.thread = threading.Thread(
            target=self.detector.run,
            daemon=True
        )
        self.thread.start()

        print("Camera Started")

    def stop(self):

        self.running = False

        if self.detector:
            self.detector.stop()

        print("Camera Stopped")

    def generate(self):

        while True:

            if not self.running or self.detector is None:
                time.sleep(0.1)
                continue

            frame = self.detector.frame

            if frame is None:
                time.sleep(0.01)
                continue

            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   frame_bytes +
                   b"\r\n")
            

camera_controller = CameraController()

