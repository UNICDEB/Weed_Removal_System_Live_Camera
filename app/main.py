from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.camera_manager import camera_controller
import cv2
from app.network_manager import network_manager
from app.arduino_manager import arduino_manager
import time
from fastapi import Body
from app.motion_calculator import update_config, load_config
from fastapi import Body
import json
import os
from fastapi.responses import HTMLResponse
from fastapi import Request



app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/start")
# def start_camera(conf: float):
#     camera_controller.start(conf)
#     return {"status": "Camera Started"}

@app.post("/start")
def start_camera(conf: float):
    camera_controller.stop()
    time.sleep(0.2)
    camera_controller.start(conf)
    return {"status": "Camera Started"}

@app.post("/stop")
def stop_camera():
    camera_controller.stop()
    return {"status": "Camera Stopped"}

# @app.post("/exit")
# def exit_system():
#     camera_controller.stop()
#     import os
#     os._exit(0)

@app.get("/video")
def video_feed():
    return StreamingResponse(
        camera_controller.generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# @app.get("/status")
# def get_status():

#     detector = camera_controller.detector

#     if detector is None:
#         return {
#             "device": "N/A",
#             "detection_count": 0,
#             "receiver": False,
#             "latest": None,
#             "log": []
#         }

#     return {
#         "device": detector.device.upper(),
#         "detection_count": detector.detection_count,
#         "receiver": network_manager.connected,
#         "latest": detector.latest_result,
#         "log": detector.log
#     }



@app.get("/status")
def get_status():

    detector = camera_controller.detector

    # 🔴 If detector not initialized
    if detector is None:
        return {
            "device": "NOT INITIALIZED",
            "detection_count": 0,
            "target": "None",
            "receiver": "Disconnected",
            "latest": None,
            "log": [],
            "arduino_connected": arduino_manager.connected,
            "rpi_connected": network_manager.connected
        }

    # 🔥 Determine receiver status properly
    receiver_status = "Disconnected"

    if detector.target_mode == "arduino":
        receiver_status = (
            "Arduino Connected" if arduino_manager.connected
            else "Arduino Not Connected"
        )

    elif detector.target_mode == "rpi":
        receiver_status = (
            "RPI Connected" if network_manager.connected
            else "RPI Not Connected"
        )

    return {
        "device": detector.device.upper(),
        "detection_count": detector.detection_count,
        "target": detector.target_mode,
        "receiver": receiver_status,
        "latest": detector.latest_result,
        "log": detector.log,
        "arduino_connected": arduino_manager.connected,
        "rpi_connected": network_manager.connected
    }


@app.post("/toggle_zone")
def toggle_zone():
    if camera_controller.detector:
        camera_controller.detector.toggle_zone_mode()
        return {"zone": camera_controller.detector.zone_mode}
    return {"zone": False}

@app.post("/set_target")
def set_target(mode: str):

    if camera_controller.detector:

        camera_controller.detector.set_target(mode)

        if mode == "arduino":
            arduino_manager.connect()

        return {"target": mode}

    return {"target": "none"}





@app.post("/arduino_command")
def arduino_command(cmd: str):

    arduino_manager.send_command(cmd)
    time.sleep(0.1)

    feedback = arduino_manager.read_feedback()

    return {
        "sent": cmd,
        "feedback": feedback
    }


@app.post("/exit")
def exit_system():
    camera_controller.stop()

    if arduino_manager.ser and arduino_manager.ser.is_open:
        arduino_manager.ser.close()

    import os
    os._exit(0)



@app.post("/update_config")
def update_config_api(data: dict = Body(...)):
    config = update_config(data)
    return {"status":"updated", "config": config}


CONFIG_FILE = "motion_config.json"

@app.get("/value_input", response_class=HTMLResponse)
async def value_input_page(request: Request):
    return templates.TemplateResponse("value_input.html", {"request": request})


@app.post("/save_motion_config")
async def save_motion_config(data: dict = Body(...)):

    # If file exists → load old values
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            old_data = json.load(f)
    else:
        old_data = {}

    # Update only provided values
    for key, value in data.items():
        if value is not None:
            old_data[key] = value

    # Save updated file
    with open(CONFIG_FILE, "w") as f:
        json.dump(old_data, f, indent=4)

    return {"status": "saved"}


@app.post("/weeder/up")
def manual_weeder_up():

    detector = camera_controller.detector
    if detector is None:
        return {"status": "error", "msg": "Camera not started"}

    if detector.weeder_position == "UP":
        return {"status": "ignored", "msg": "Weeder already UP"}

    detector.send_command("xU0350")
    detector.weeder_position = "UP"
    detector.log.append("MANUAL → UP")

    return {"status": "ok", "msg": "Weeder moved UP"}


@app.post("/weeder/down")
def manual_weeder_down():

    detector = camera_controller.detector
    if detector is None:
        return {"status": "error", "msg": "Camera not started"}

    if detector.weeder_position == "DOWN":
        return {"status": "ignored", "msg": "Weeder already DOWN"}

    detector.send_command("xD0350")
    detector.weeder_position = "DOWN"
    detector.log.append("MANUAL → DOWN")

    return {"status": "ok", "msg": "Weeder moved DOWN"}





