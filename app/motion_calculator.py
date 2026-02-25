# import math
# import json
# import os


# CONFIG_FILE = "motion_config.json"


# # ---------------------------------------------------
# # CONFIG LOAD
# # ---------------------------------------------------
# def load_config():
#     if not os.path.exists(CONFIG_FILE):
#         default = {
#             "camera_height": 1.0,
#             "camera_angle": 0.0,
#             "tool_distance": 0.2,
#             "speed": 0.5,
#             "calibration_time": 0
#         }
#         save_config(default)
#         return default

#     with open(CONFIG_FILE, "r") as f:
#         return json.load(f)


# # ---------------------------------------------------
# # CONFIG SAVE
# # ---------------------------------------------------
# def save_config(data):
#     with open(CONFIG_FILE, "w") as f:
#         json.dump(data, f, indent=4)


# # ---------------------------------------------------
# # UPDATE CONFIG (ONLY IF USER PROVIDES VALUE)
# # ---------------------------------------------------
# def update_config(user_data):
#     config = load_config()

#     for key in user_data:
#         if user_data[key] is not None:
#             config[key] = float(user_data[key])

#     save_config(config)
#     return config


# # ---------------------------------------------------
# # CORE TIME CALCULATION
# # ---------------------------------------------------
# def calculate_time(z_value, config):

#     camera_height = config["camera_height"]
#     tool_distance = config["tool_distance"]
#     speed = config["speed"]
#     calibration_time = config["calibration_time"]

#     # b = sqrt(z^2 - h^2)
#     b = math.sqrt(abs((z_value ** 2) - (camera_height ** 2)))

#     # d = b + tool_distance
#     d = b + tool_distance

#     # t = ((d/speed)*1000) + calibration_time
#     t = ((d / speed) * 1000) + calibration_time

#     return int(t)


# # ---------------------------------------------------
# # FORMAT TO 4 DIGIT STRING
# # ---------------------------------------------------
# def format_4digit(value):
#     return str(value).zfill(4)


# # ---------------------------------------------------
# # MAIN FUNCTION
# # ---------------------------------------------------
# def generate_motion_commands(start_z, end_z):

#     config = load_config()

#     t_start = calculate_time(start_z, config)
#     t_end = calculate_time(end_z, config)

#     t_start_str = format_4digit(t_start)
#     t_end_str = format_4digit(t_end)

#     cmd_up = "xU" + t_start_str
#     cmd_down = "xD" + t_end_str

#     return cmd_up, cmd_down

##########################
# ## Date: 24/02/2026

import math
import json
import os

CONFIG_FILE = "motion_config.json"


# ---------------------------------------------------
# CONFIG LOAD
# ---------------------------------------------------
def load_config():
    if not os.path.exists(CONFIG_FILE):
        default = {
            "camera_height": 0.62,      # meters
            "camera_angle": 38.0,       # degrees downward tilt
            "tool_distance": 1.10,      # meters (98 cm)
            "speed": 0.70,              # m/s
            "calibration_time": 0       # milliseconds
        }
        save_config(default)
        return default

    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


# ---------------------------------------------------
# CONFIG SAVE
# ---------------------------------------------------
def save_config(data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------
# UPDATE CONFIG
# ---------------------------------------------------
def update_config(user_data):
    config = load_config()

    for key in user_data:
        if user_data[key] is not None:
            config[key] = float(user_data[key])

    save_config(config)
    return config


# ---------------------------------------------------
# CORE TIME CALCULATION (TILTED CAMERA VERSION)
# ---------------------------------------------------
def calculate_time(z_value, config):

    h = config["camera_height"]          # meters
    theta = config["camera_angle"]       # degrees
    tool_distance = config["tool_distance"]
    speed = config["speed"]
    calibration_time = config["calibration_time"]

    theta_rad = math.radians(theta)

    # 🔥 Ground distance considering tilt
    ground_distance = (z_value * math.cos(theta_rad)) - (h * math.sin(theta_rad))

    # Total forward travel until tool reaches weed
    total_distance = ground_distance + tool_distance

    if total_distance < 0:
        total_distance = 0

    # Time in milliseconds
    time_ms = ((total_distance / speed) * 1000) + calibration_time

    return int(time_ms)


# ---------------------------------------------------
# FORMAT TO 4 DIGIT STRING
# ---------------------------------------------------
def format_4digit(value):
    return str(value).zfill(4)


# ---------------------------------------------------
# GENERATE MOTION COMMANDS
# ---------------------------------------------------
def generate_motion_commands(start_z, end_z):

    config = load_config()

    t_start = calculate_time(start_z, config)
    t_end = calculate_time(end_z, config)

    cmd_up = "xU" + format_4digit(t_start)
    cmd_down = "xD" + format_4digit(t_end)

    return cmd_up, cmd_down