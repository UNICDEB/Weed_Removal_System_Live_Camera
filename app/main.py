from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.camera_manager import camera_controller
import cv2
from app.network_manager import network_manager


app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start")
def start_camera(conf: float):
    camera_controller.start(conf)
    return {"status": "Camera Started"}

@app.post("/stop")
def stop_camera():
    camera_controller.stop()
    return {"status": "Camera Stopped"}

@app.post("/exit")
def exit_system():
    camera_controller.stop()
    import os
    os._exit(0)

@app.get("/video")
def video_feed():
    return StreamingResponse(
        camera_controller.generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
def get_status():

    detector = camera_controller.detector

    if detector is None:
        return {
            "device": "N/A",
            "detection_count": 0,
            "receiver": False,
            "latest": None,
            "log": []
        }

    return {
        "device": detector.device.upper(),
        "detection_count": detector.detection_count,
        "receiver": network_manager.connected,
        "latest": detector.latest_result,
        "log": detector.log
    }

@app.post("/toggle_zone")
def toggle_zone():
    if camera_controller.detector:
        camera_controller.detector.toggle_zone_mode()
        return {"zone": camera_controller.detector.zone_mode}
    return {"zone": False}


