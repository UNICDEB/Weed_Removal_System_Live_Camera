# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import torch
# from ultralytics import YOLO
# from app.depth_filters import apply_depth_filters
# from app.network_manager import network_manager

# MODEL_PATH = "E:/Debabrata/Weed/Weed_Removal_Syatem_Using_AI/Weight/yolo11s.pt"

# class DetectionManager:

#     def __init__(self):
#         self.running = False
#         self.frame = None
#         self.confidence = 0.5

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print("Running on:", self.device)

#         self.model = YOLO(MODEL_PATH)
#         self.model.to(self.device)
#         self.detection_count = 0
#         self.latest_result = None
#         self.log = []

#     def set_confidence(self, conf):
#         self.confidence = conf

#     def stop(self):
#         self.running = False

#     def run(self):
#         self.running = True

#         pipeline = rs.pipeline()
#         config = rs.config()

#         config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
#         config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

#         pipeline.start(config)
#         align = rs.align(rs.stream.color)

#         while self.running:
#             frames = pipeline.wait_for_frames()
#             aligned = align.process(frames)

#             color_frame = aligned.get_color_frame()
#             depth_frame = aligned.get_depth_frame()

#             if not color_frame or not depth_frame:
#                 continue

#             depth_frame = apply_depth_filters(depth_frame)

#             color_image = np.asanyarray(color_frame.get_data()).copy()

#             results = self.model(color_image, conf=self.confidence,
#                                  imgsz=640, device=self.device)

#             for box in results[0].boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x = int((x1 + x2) / 2)
#                 center_y = int((y1 + y2) / 2)

#                 depth = depth_frame.get_distance(center_x, center_y)
#                 if depth == 0:
#                     continue

#                 intrin = depth_frame.profile.as_video_stream_profile().intrinsics
#                 point_3d = rs.rs2_deproject_pixel_to_point(
#                     intrin, [center_x, center_y], depth)

#                 X_cm = point_3d[0] * 100
#                 Y_cm = point_3d[1] * 100
#                 Z_cm = point_3d[2] * 100

#                 cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,255,0), 2)
#                 cv2.putText(color_image,
#                             f"X:{X_cm:.1f} Y:{Y_cm:.1f} Z:{Z_cm:.1f}",
#                             (x1, y1-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

#                 network_manager.send(X_cm, Y_cm, Z_cm)

#             self.frame = color_image

#         pipeline.stop()

###################

# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import torch
# import time
# from ultralytics import YOLO
# from app.depth_filters import apply_depth_filters
# from app.network_manager import network_manager
# from app.arduino_manager import arduino_manager
# from app.motion_calculator import generate_motion_commands


# # MODEL_PATH = "E:/Debabrata/Weed/Weed_Removal_Syatem_Using_AI/Weight/best_nitu_001(ep-50).pt"
# MODEL_PATH = "E:/Debabrata/Weed/Weed_Removal_Syatem_Using_AI/Weight/yolo11s.pt"

# class DetectionManager:

#     def __init__(self):

#         self.running = False
#         self.frame = None
#         self.confidence = 0.5
#         self.zone_mode = False
#         self.target_mode = "rpi"   # default
        



#         # 🔥 GPU / CPU Detection
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print("Running on:", self.device.upper())

#         self.model = YOLO(MODEL_PATH)
#         self.model.to(self.device)

#         # 🔥 Dashboard Variables
#         self.detection_count = 0
#         self.latest_result = None
#         self.log = []

#     # --------------------------------------------------

#     def set_confidence(self, conf):
#         self.confidence = float(conf)

#     # --------------------------------------------------

#     def stop(self):
#         self.running = False

#     # --------------------------------------------------

#     def run(self):

#         self.running = True

#         pipeline = rs.pipeline()
#         config = rs.config()

#         config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
#         config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

#         profile = pipeline.start(config)
#         align = rs.align(rs.stream.color)

#         print("Camera + Detection Started")

#         try:
#             # while self.running:

#             #     frames = pipeline.wait_for_frames()
#             #     aligned = align.process(frames)

#             #     color_frame = aligned.get_color_frame()
#             #     depth_frame = aligned.get_depth_frame()

#             #     if not color_frame or not depth_frame:
#             #         continue

#             #     depth_frame = apply_depth_filters(depth_frame)

#             #     color_image = np.asanyarray(color_frame.get_data()).copy()

#             #     # 🔥 YOLO Inference
#             #     results = self.model(
#             #         color_image,
#             #         conf=self.confidence,
#             #         imgsz=640,
#             #         device=self.device,
#             #         verbose=False
#             #     )

#             #     detection_found = False

#             #     for box in results[0].boxes:

#             #         detection_found = True

#             #         x1, y1, x2, y2 = map(int, box.xyxy[0])
#             #         center_x = int((x1 + x2) / 2)
#             #         center_y = int((y1 + y2) / 2)

#             #         depth = depth_frame.get_distance(center_x, center_y)
#             #         if depth == 0:
#             #             continue

#             #         intrin = depth_frame.profile.as_video_stream_profile().intrinsics
#             #         point_3d = rs.rs2_deproject_pixel_to_point(
#             #             intrin, [center_x, center_y], depth)

#             #         X_cm = point_3d[0] * 100
#             #         Y_cm = point_3d[1] * 100
#             #         Z_cm = point_3d[2] * 100

#             #         # 🔥 Draw Box
#             #         cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             #         cv2.circle(color_image, (center_x, center_y), 4, (0, 0, 255), -1)

#             #         cv2.putText(
#             #             color_image,
#             #             f"X:{X_cm:.1f} Y:{Y_cm:.1f} Z:{Z_cm:.1f}",
#             #             (x1, y1 - 10),
#             #             cv2.FONT_HERSHEY_SIMPLEX,
#             #             0.6,
#             #             (255, 0, 0),
#             #             2
#             #         )

#             #         # 🔥 Update Dashboard Values
#             #         self.detection_count += 1

#             #         self.latest_result = {
#             #             "X": round(X_cm, 2),
#             #             "Y": round(Y_cm, 2),
#             #             "Z": round(Z_cm, 2)
#             #         }

#             #         self.log.append(self.latest_result)

#             #         # Keep only last 100 entries
#             #         if len(self.log) > 100:
#             #             self.log.pop(0)

#             #         print("Detected:", self.latest_result)

#             #         # 🔥 Send to Receiver
#             #         network_manager.send(X_cm, Y_cm, Z_cm)

#             #     # Even if no detection, keep frame updating
#             #     self.frame = color_image

#             #     time.sleep(0.001)

#             while self.running:

#                 frames = pipeline.wait_for_frames()
#                 aligned = align.process(frames)

#                 color_frame = aligned.get_color_frame()
#                 depth_frame = aligned.get_depth_frame()

#                 if not color_frame or not depth_frame:
#                     continue

#                 depth_frame = apply_depth_filters(depth_frame)

#                 color_image = np.asanyarray(color_frame.get_data()).copy()
#                 height, width, _ = color_image.shape

#                 results = self.model(
#                     color_image,
#                     conf=self.confidence,
#                     imgsz=640,
#                     device=self.device,
#                     verbose=False
#                 )

#                 intrin = depth_frame.profile.as_video_stream_profile().intrinsics

#                 # ----------------------------------------------------
#                 # 🔥 ZONE LOGIC
#                 # ----------------------------------------------------
#                 left_pixel = None
#                 right_pixel = None

#                 if self.zone_mode:

#                     Z_ref = 0.5  # 50 cm reference forward

#                     left_point = [-0.15, 0, Z_ref]
#                     right_point = [0.15, 0, Z_ref]

#                     left_pixel = rs.rs2_project_point_to_pixel(intrin, left_point)
#                     right_pixel = rs.rs2_project_point_to_pixel(intrin, right_point)

#                     left_x = int(left_pixel[0])
#                     right_x = int(right_pixel[0])

#                     # Draw vertical lines
#                     cv2.line(color_image, (left_x, 0), (left_x, height), (255, 0, 0), 2)
#                     cv2.line(color_image, (right_x, 0), (right_x, height), (255, 0, 0), 2)

#                 # ----------------------------------------------------

#                 for box in results[0].boxes:

#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     center_x = int((x1 + x2) / 2)
#                     center_y = int((y1 + y2) / 2)

#                     depth = depth_frame.get_distance(center_x, center_y)
#                     if depth == 0:
#                         continue

#                     point_3d = rs.rs2_deproject_pixel_to_point(
#                         intrin, [center_x, center_y], depth)

#                     X_cm = point_3d[0] * 100
#                     Y_cm = point_3d[1] * 100
#                     Z_cm = point_3d[2] * 100

#                     # Draw bounding box
#                     cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.circle(color_image, (center_x, center_y), 4, (0, 0, 255), -1)

#                     send_allowed = True

#                     # ----------------------------------------------------
#                     # 🔥 APPLY ZONE FILTER
#                     # ----------------------------------------------------
#                     if self.zone_mode:

#                         if center_x < left_x or center_x > right_x:
#                             send_allowed = False

#                             cv2.putText(
#                                 color_image,
#                                 "Outside the Box",
#                                 (x1, y1 - 25),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.7,
#                                 (0, 0, 255),
#                                 2
#                             )

#                     # ----------------------------------------------------

#                     if send_allowed:

#                         # ---------------------------------------------------
#                         # 🔥 START & END BOUNDING BOX 3D CALCULATION
#                         # ---------------------------------------------------

#                         # Starting corner (x1, y1)
#                         start_depth = depth_frame.get_distance(x1, y1)

#                         # Ending corner (x2, y2)
#                         end_depth = depth_frame.get_distance(x2, y2)

#                         if start_depth == 0 or end_depth == 0:
#                             continue

#                         start_point = rs.rs2_deproject_pixel_to_point(
#                             intrin, [x1, y1], start_depth)

#                         end_point = rs.rs2_deproject_pixel_to_point(
#                             intrin, [x2, y2], end_depth)

#                         # Z in meters
#                         start_z = start_point[2]
#                         end_z = end_point[2]

#                         # ---------------------------------------------------
#                         # 🔥 GENERATE MOTION COMMANDS
#                         # ---------------------------------------------------
#                         cmd_up, cmd_down = generate_motion_commands(start_z, end_z)

#                         print("Generated Commands:", cmd_up, cmd_down)

#                         # ---------------------------------------------------
#                         # 🔥 SEND TO ARDUINO
#                         # ---------------------------------------------------
#                         if self.target_mode == "arduino" and arduino_manager.connected:

#                             arduino_manager.send_raw(cmd_up)
#                             time.sleep(0.05)
#                             arduino_manager.send_raw(cmd_down)

#                         elif self.target_mode == "rpi":
#                             if not network_manager.connected:
#                                 network_manager.connect()

#                             if network_manager.connected:
#                                 network_manager.send(start_z, end_z, 0)

#                         # ---------------------------------------------------
#                         # 🔥 DASHBOARD UPDATE (Optional)
#                         # ---------------------------------------------------
#                         self.detection_count += 1

#                         self.latest_result = {
#                             "Start_Z(m)": round(start_z, 3),
#                             "End_Z(m)": round(end_z, 3),
#                             "CMD_UP": cmd_up,
#                             "CMD_DOWN": cmd_down
#                         }

#                         self.log.append(self.latest_result)

#                         if len(self.log) > 100:
#                             self.log.pop(0)



#                         cv2.putText(
#                             color_image,
#                             f"X:{X_cm:.1f} Y:{Y_cm:.1f} Z:{Z_cm:.1f}",
#                             (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.6,
#                             (255, 0, 0),
#                             2
#                         )

#                 self.frame = color_image


#         finally:
#             pipeline.stop()
#             print("Camera Stopped Safely")


#     def toggle_zone_mode(self):
#         self.zone_mode = not self.zone_mode
#         print("Zone Mode:", self.zone_mode)

#     ### Target Mode selection---------------
#     def set_target(self, mode):

#         self.target_mode = mode
#         print("Target Mode:", mode)

#         # ---------- Arduino Mode ----------
#         if mode == "arduino":

#             # Disconnect RPI if connected
#             if network_manager.connected:
#                 try:
#                     network_manager.client.close()
#                 except:
#                     pass
#                 network_manager.connected = False

#             arduino_manager.connect()

#         # ---------- RPI Mode ----------
#         elif mode == "rpi":

#             # Disconnect Arduino if connected
#             if arduino_manager.connected:
#                 try:
#                     arduino_manager.ser.close()
#                 except:
#                     pass
#                 arduino_manager.connected = False

#             network_manager.connect()

#############################
# ## Date: 24.02.2026

# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import torch
# import time
# from ultralytics import YOLO

# from app.depth_filters import apply_depth_filters
# from app.network_manager import network_manager
# from app.arduino_manager import arduino_manager
# from app.motion_calculator import generate_motion_commands

# #  MODEL_PATH = "E:/Debabrata/Weed/Weed_Removal_Syatem_Using_AI/Weight/best_nitu_001(ep-50).pt"
# #  MODEL_PATH = "E:/Debabrata/Weed/Weed_Removal_Syatem_Using_AI/Weight/best_trl_fld_debu(25_02_26).pt"
# MODEL_PATH = "E:/Debabrata/Weed/Weed_Removal_Syatem_Using_AI/Weight/yolo11s.pt"


# class DetectionManager:

#     def __init__(self):

#         self.running = False
#         self.frame = None
#         self.confidence = 0.5
#         self.zone_mode = False
#         self.target_mode = "rpi"

#         # -----------------------------------
#         # 🔥 DEVICE SETUP
#         # -----------------------------------
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print("Running on:", self.device.upper())

#         self.model = YOLO(MODEL_PATH)
#         self.model.to(self.device)

#         # -----------------------------------
#         # 🔥 DUPLICATE REMOVAL PARAMETERS
#         # -----------------------------------

#         # Forward trigger window (meters)
#         self.trigger_min_z = 0.6
#         self.trigger_max_z = 1.2

#         # Lock threshold (meters)
#         self.z_lock_threshold = 0.15

#         # Unlock distance (meters)
#         self.unlock_z_threshold = 0.30

#         self.last_trigger_z = None
#         self.trigger_lock = False

#         # -----------------------------------
#         # Dashboard
#         # -----------------------------------
#         self.detection_count = 0
#         self.latest_result = None
#         self.log = []

#     # --------------------------------------------------

#     def set_confidence(self, conf):
#         self.confidence = float(conf)

#     # --------------------------------------------------

#     def stop(self):
#         self.running = False

#     # --------------------------------------------------

#     def run(self):

#         self.running = True

#         pipeline = rs.pipeline()
#         config = rs.config()

#         config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
#         config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

#         profile = pipeline.start(config)
#         align = rs.align(rs.stream.color)

#         print("Camera + Detection Started")

#         try:

#             while self.running:

#                 frames = pipeline.wait_for_frames()
#                 aligned = align.process(frames)

#                 color_frame = aligned.get_color_frame()
#                 depth_frame = aligned.get_depth_frame()

#                 if not color_frame or not depth_frame:
#                     continue

#                 depth_frame = apply_depth_filters(depth_frame)

#                 color_image = np.asanyarray(color_frame.get_data()).copy()
#                 height, width, _ = color_image.shape

#                 results = self.model(
#                     color_image,
#                     conf=self.confidence,
#                     imgsz=640,
#                     device=self.device,
#                     verbose=False
#                 )

#                 intrin = depth_frame.profile.as_video_stream_profile().intrinsics

#                 # ----------------------------------------------------
#                 # 🔥 ZONE LOGIC
#                 # ----------------------------------------------------

#                 left_x = None
#                 right_x = None

#                 if self.zone_mode:

#                     Z_ref = 0.8  # reference depth
#                     left_point = [-0.15, 0, Z_ref]
#                     right_point = [0.15, 0, Z_ref]

#                     left_pixel = rs.rs2_project_point_to_pixel(intrin, left_point)
#                     right_pixel = rs.rs2_project_point_to_pixel(intrin, right_point)

#                     left_x = int(left_pixel[0])
#                     right_x = int(right_pixel[0])

#                     cv2.line(color_image, (left_x, 0), (left_x, height), (255, 0, 0), 2)
#                     cv2.line(color_image, (right_x, 0), (right_x, height), (255, 0, 0), 2)

#                 # ----------------------------------------------------

#                 for box in results[0].boxes:

#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     center_x = int((x1 + x2) / 2)
#                     center_y = int((y1 + y2) / 2)

#                     depth = depth_frame.get_distance(center_x, center_y)
#                     if depth == 0:
#                         continue

#                     point_3d = rs.rs2_deproject_pixel_to_point(
#                         intrin, [center_x, center_y], depth)

#                     X = point_3d[0]
#                     Y = point_3d[1]
#                     Z = point_3d[2]

#                     # Draw bounding box
#                     cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.circle(color_image, (center_x, center_y), 4, (0, 0, 255), -1)

#                     # ----------------------------------------------------
#                     # 🔥 ZONE FILTER
#                     # ----------------------------------------------------

#                     if self.zone_mode:
#                         if center_x < left_x or center_x > right_x:
#                             continue

#                     # ----------------------------------------------------
#                     # 🔥 FORWARD DEPTH WINDOW FILTER
#                     # ----------------------------------------------------

#                     if not (self.trigger_min_z <= Z <= self.trigger_max_z):
#                         continue

#                     # ----------------------------------------------------
#                     # 🔥 DUPLICATE REMOVAL LOGIC
#                     # ----------------------------------------------------

#                     allow_send = False

#                     if not self.trigger_lock:
#                         allow_send = True

#                     else:
#                         # Allow if new object far from previous
#                         if abs(Z - self.last_trigger_z) > self.z_lock_threshold:
#                             allow_send = True

#                     # ----------------------------------------------------

#                     if allow_send:

#                         print("Triggering at Z:", round(Z, 3))

#                         self.trigger_lock = True
#                         self.last_trigger_z = Z

#                         # Calculate start & end 3D for motion
#                         start_depth = depth_frame.get_distance(x1, y1)
#                         end_depth = depth_frame.get_distance(x2, y2)

#                         if start_depth == 0 or end_depth == 0:
#                             continue

#                         start_point = rs.rs2_deproject_pixel_to_point(
#                             intrin, [x1, y1], start_depth)

#                         end_point = rs.rs2_deproject_pixel_to_point(
#                             intrin, [x2, y2], end_depth)

#                         start_z = start_point[2]
#                         end_z = end_point[2]

#                         cmd_up, cmd_down = generate_motion_commands(start_z, end_z)

#                         # Only proceed if valid commands generated
#                         if cmd_up is not None and cmd_down is not None:

#                             if self.target_mode == "arduino" and arduino_manager.connected:
#                                 arduino_manager.send_raw(cmd_up)
#                                 time.sleep(0.05)
#                                 arduino_manager.send_raw(cmd_down)

#                             elif self.target_mode == "rpi":
#                                 if network_manager.connected:
#                                     network_manager.send(start_z, end_z, 0)

#                             # Update dashboard ONLY when valid trigger
#                             self.detection_count += 1
#                             self.latest_result = {
#                                 "Start_Z(m)": round(start_z, 3),
#                                 "End_Z(m)": round(end_z, 3),
#                                 "CMD_UP": cmd_up,
#                                 "CMD_DOWN": cmd_down
#                             }

#                             self.log.append(self.latest_result)
#                             if len(self.log) > 100:
#                                 self.log.pop(0)

#                     # ----------------------------------------------------
#                     # 🔥 UNLOCK LOGIC
#                     # ----------------------------------------------------

#                     if self.trigger_lock and self.last_trigger_z is not None:
#                         if Z > (self.trigger_max_z + 0.2):
#                             print("Object passed. Unlocking.")
#                             self.trigger_lock = False
#                             self.last_trigger_z = None

#                     cv2.putText(
#                         color_image,
#                         f"Z:{Z:.2f}m",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (255, 0, 0),
#                         2
#                     )

#                 self.frame = color_image
#                 time.sleep(0.001)

#         finally:
#             pipeline.stop()
#             print("Camera Stopped Safely")

#     # --------------------------------------------------

#     def toggle_zone_mode(self):
#         self.zone_mode = not self.zone_mode
#         print("Zone Mode:", self.zone_mode)

#     # --------------------------------------------------

#     def set_target(self, mode):

#         self.target_mode = mode
#         print("Target Mode:", mode)

#         if mode == "arduino":
#             if network_manager.connected:
#                 network_manager.client.close()
#                 network_manager.connected = False
#             arduino_manager.connect()

#         elif mode == "rpi":
#             if arduino_manager.connected:
#                 arduino_manager.ser.close()
#                 arduino_manager.connected = False
#             network_manager.connect()

#####################
# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import torch
# import time
# from ultralytics import YOLO

# from app.depth_filters import apply_depth_filters
# from app.network_manager import network_manager
# from app.arduino_manager import arduino_manager
# from app.motion_calculator import generate_motion_commands


# MODEL_PATH = "E:/Debabrata/Weed/Weed_Removal_Syatem_Using_AI/Weight/yolo11s.pt"


# class DetectionManager:

#     def __init__(self):

#         # ----------------------------
#         # Runtime Flags
#         # ----------------------------
#         self.running = False
#         self.frame = None
#         self.confidence = 0.5
#         self.zone_mode = False
#         self.target_mode = "rpi"

#         # ----------------------------
#         # Device Setup
#         # ----------------------------
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print("Running on:", self.device.upper())

#         self.model = YOLO(MODEL_PATH)
#         self.model.to(self.device)

#         # ----------------------------
#         # Trigger System
#         # ----------------------------
#         self.trigger_lock = False
#         self.last_trigger_z = None

#         # ----------------------------
#         # Dashboard
#         # ----------------------------
#         self.detection_count = 0
#         self.latest_result = None
#         self.log = []

#     # ------------------------------------------

#     def set_confidence(self, conf):
#         self.confidence = float(conf)

#     # ------------------------------------------

#     def stop(self):
#         self.running = False

#     # ------------------------------------------

#     def run(self):

#         self.running = True

#         pipeline = rs.pipeline()
#         config = rs.config()

#         config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
#         config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

#         profile = pipeline.start(config)
#         align = rs.align(rs.stream.color)

#         print("Camera + Detection Started")

#         try:

#             while self.running:

#                 frames = pipeline.wait_for_frames()
#                 aligned = align.process(frames)

#                 color_frame = aligned.get_color_frame()
#                 depth_frame = aligned.get_depth_frame()

#                 if not color_frame or not depth_frame:
#                     continue

#                 depth_frame = apply_depth_filters(depth_frame)

#                 color_image = np.asanyarray(color_frame.get_data()).copy()
#                 height, width, _ = color_image.shape

#                 # YOLO Inference
#                 results = self.model(
#                     color_image,
#                     conf=self.confidence,
#                     imgsz=640,
#                     device=self.device,
#                     verbose=False
#                 )

#                 intrin = depth_frame.profile.as_video_stream_profile().intrinsics

#                 # --------------------------------------------------------
#                 # ZONE CALCULATION (15cm left-right from center)
#                 # --------------------------------------------------------

#                 left_x = None
#                 right_x = None

#                 if self.zone_mode:

#                     Z_ref = 1.0  # reference depth (1 meter)

#                     # 15 cm left & right in real world
#                     left_point = [-0.15, 0, Z_ref]
#                     right_point = [0.15, 0, Z_ref]

#                     left_pixel = rs.rs2_project_point_to_pixel(intrin, left_point)
#                     right_pixel = rs.rs2_project_point_to_pixel(intrin, right_point)

#                     left_x = int(left_pixel[0])
#                     right_x = int(right_pixel[0])

#                     # Draw vertical lines
#                     cv2.line(color_image, (left_x, 0), (left_x, height), (255, 0, 0), 2)
#                     cv2.line(color_image, (right_x, 0), (right_x, height), (255, 0, 0), 2)

#                 # --------------------------------------------------------
#                 # Process detections
#                 # --------------------------------------------------------

#                 for box in results[0].boxes:

#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     center_x = int((x1 + x2) / 2)
#                     center_y = int((y1 + y2) / 2)

#                     # --------------------------------------------------------
#                     # ZONE FILTER
#                     # --------------------------------------------------------

#                     if self.zone_mode:
#                         if center_x < left_x or center_x > right_x:
#                             continue

#                     # Get CENTER depth (stable)
#                     depth = depth_frame.get_distance(center_x, center_y)

#                     if depth <= 0:
#                         continue

#                     point_3d = rs.rs2_deproject_pixel_to_point(
#                         intrin, [center_x, center_y], depth)

#                     X, Y, Z = point_3d

#                     # Optional safety depth window
#                     if not (0.3 <= Z <= 2.0):
#                         continue

#                     # Draw visuals
#                     cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.circle(color_image, (center_x, center_y), 4, (0, 0, 255), -1)

#                     cv2.putText(
#                         color_image,
#                         f"Z:{Z:.2f}m",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (255, 0, 0),
#                         2
#                     )

#                     # --------------------------------------
#                     # Lock System
#                     # --------------------------------------

#                     allow_trigger = False

#                     if not self.trigger_lock:
#                         allow_trigger = True
#                     else:
#                         # New object far from previous
#                         if abs(Z - self.last_trigger_z) > 0.25:
#                             allow_trigger = True

#                     if not allow_trigger:
#                         continue

#                     # --------------------------------------
#                     # Generate Motion Command
#                     # --------------------------------------

#                     cmd_up, cmd_down = generate_motion_commands(Z, Z)

#                     if cmd_up is None or cmd_down is None:
#                         continue

#                     # --------------------------------------
#                     # Trigger Execution
#                     # --------------------------------------

#                     print(f"Triggering at Z: {round(Z,3)}")

#                     self.trigger_lock = True
#                     self.last_trigger_z = Z

#                     if self.target_mode == "arduino" and arduino_manager.connected:
#                         arduino_manager.send_raw(cmd_up)
#                         time.sleep(0.05)
#                         arduino_manager.send_raw(cmd_down)

#                     elif self.target_mode == "rpi":
#                         if network_manager.connected:
#                             network_manager.send(Z, Z, 0)

#                     # Dashboard update
#                     self.detection_count += 1
#                     self.latest_result = {
#                         "Start_Z(m)": round(Z, 3),
#                         "End_Z(m)": round(Z, 3),
#                         "CMD_UP": cmd_up,
#                         "CMD_DOWN": cmd_down
#                     }

#                     self.log.append(self.latest_result)
#                     if len(self.log) > 100:
#                         self.log.pop(0)

#                 # --------------------------------------
#                 # Unlock condition (movement based)
#                 # --------------------------------------

#                 if self.trigger_lock and self.last_trigger_z is not None:

#                     # If no object near previous Z anymore
#                     # Reset after sufficient change
#                     if results[0].boxes is None or len(results[0].boxes) == 0:
#                         self.trigger_lock = False
#                         self.last_trigger_z = None

#                 self.frame = color_image

#         finally:
#             pipeline.stop()
#             print("Camera Stopped Safely")

#     # ------------------------------------------

#     def toggle_zone_mode(self):
#         self.zone_mode = not self.zone_mode
#         print("Zone Mode:", self.zone_mode)

#     # ------------------------------------------

#     def set_target(self, mode):

#         self.target_mode = mode
#         print("Target Mode:", mode)

#         if mode == "arduino":
#             if network_manager.connected:
#                 network_manager.client.close()
#                 network_manager.connected = False
#             arduino_manager.connect()

#         elif mode == "rpi":
#             if arduino_manager.connected:
#                 arduino_manager.ser.close()
#                 arduino_manager.connected = False
#             network_manager.connect()

#####################
##  Updated Version

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time
from ultralytics import YOLO

from app.arduino_manager import arduino_manager
from app.network_manager import network_manager


MODEL_PATH = "E:/Debabrata/Weed/Weed_Removal_Syatem_Using_AI/Weight/yolo11s.pt"


class DetectionManager:

    def __init__(self):

        # =============================
        # SYSTEM FLAGS
        # =============================
        self.running = False
        self.frame = None
        self.confidence = 0.5

        # =============================
        # DEVICE + MODEL
        # =============================
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(MODEL_PATH)
        self.model.to(self.device)

        # =============================
        # TARGET OUTPUT
        # =============================
        self.target_mode = "none"
        self.detection_count = 0
        self.latest_result = None

        # =============================
        # LOGGING (for status API)
        # =============================
        self.log = []

        # =============================
        # ZONE SETTINGS
        # =============================
        self.zone_mode = True
        self.zone_half_width = 120

        # Horizontal lines (rectangle)
        self.top_line_y = 200
        self.bottom_line_y = 520

        # =============================
        # STABLE STATE MACHINE
        # =============================
        self.zone_active = False
        self.inside_counter = 0
        self.outside_counter = 0
        self.stable_frames = 5

        # =============================
        # OBJECT TRACKING MEMORY
        # =============================
        self.tracked_objects = {}
        self.next_object_id = 0
        self.max_tracking_distance = 80  # pixels

        # =============================
        # COMMAND LOCK
        # =============================
        self.last_command_time = 0
        self.command_delay = 0.4  # seconds

        # =============================
        # WEEDER STATE (IMPORTANT)
        # =============================
        # Change to "UP" if your system starts physically UP
        self.weeder_position = "DOWN"


    # =====================================================
    # PUBLIC FUNCTIONS
    # =====================================================

    def set_target(self, mode):
        self.target_mode = mode

    def set_confidence(self, conf):
        self.confidence = float(conf)

    def toggle_zone_mode(self):
        self.zone_mode = not self.zone_mode
        self.reset_state()
        print("Zone Mode:", self.zone_mode)

    def stop(self):
        self.running = False

    def reset_state(self):
        self.zone_active = False
        self.inside_counter = 0
        self.outside_counter = 0
        self.tracked_objects = {}
        self.next_object_id = 0
        # ❗ DO NOT reset weeder_position here


    # =====================================================
    # SAFE COMMAND SENDER
    # =====================================================

    def send_command(self, cmd):

        current_time = time.time()

        if current_time - self.last_command_time < self.command_delay:
            return

        self.last_command_time = current_time

        print("SEND:", cmd)

        if self.target_mode == "arduino":
            if arduino_manager.connected:
                arduino_manager.send_raw(cmd)

        elif self.target_mode == "rpi":
            if network_manager.connected:
                network_manager.send(cmd)


    # =====================================================
    # SIMPLE CENTER TRACKING
    # =====================================================

    def track_objects(self, detections):

        updated_objects = {}

        for cx, cy in detections:
            matched_id = None

            for obj_id, (px, py) in self.tracked_objects.items():
                dist = np.hypot(cx - px, cy - py)
                if dist < self.max_tracking_distance:
                    matched_id = obj_id
                    break

            if matched_id is None:
                matched_id = self.next_object_id
                self.next_object_id += 1

            updated_objects[matched_id] = (cx, cy)

        self.tracked_objects = updated_objects


    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run(self):

        self.running = True

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        pipeline.start(config)

        print("Camera Started")

        try:
            while self.running:

                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                height, width, _ = frame.shape

                center_x = width // 2
                left_zone = center_x - self.zone_half_width
                right_zone = center_x + self.zone_half_width

                # ==========================================
                # DRAW ZONE (VERTICAL + HORIZONTAL)
                # ==========================================
                if self.zone_mode:
                    # Vertical lines
                    cv2.line(frame, (left_zone, 0),
                             (left_zone, height), (255, 0, 0), 2)
                    cv2.line(frame, (right_zone, 0),
                             (right_zone, height), (255, 0, 0), 2)

                    # Horizontal lines
                    cv2.line(frame, (0, self.top_line_y),
                             (width, self.top_line_y), (0, 255, 255), 2)
                    cv2.line(frame, (0, self.bottom_line_y),
                             (width, self.bottom_line_y), (0, 0, 255), 2)

                    # Rectangle
                    cv2.rectangle(
                        frame,
                        (left_zone, self.top_line_y),
                        (right_zone, self.bottom_line_y),
                        (0, 255, 0),
                        2
                    )

                # ==========================================
                # YOLO DETECTION
                # ==========================================
                results = self.model(
                    frame,
                    conf=self.confidence,
                    imgsz=640,
                    device=self.device,
                    verbose=False
                )

                detections = []

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    detections.append((cx, cy))

                    cv2.rectangle(frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy),
                               4, (0, 0, 255), -1)

                # ==========================================
                # TRACK OBJECTS
                # ==========================================
                self.track_objects(detections)

                # ==========================================
                # ZONE LOGIC (RECTANGLE BASED)
                # ==========================================
                object_inside = False

                for cx, cy in self.tracked_objects.values():
                    if (left_zone < cx < right_zone and
                        self.top_line_y < cy < self.bottom_line_y):
                        object_inside = True
                        break

                if self.zone_mode:

                    if object_inside:
                        self.inside_counter += 1
                        self.outside_counter = 0
                    else:
                        self.outside_counter += 1
                        self.inside_counter = 0

                    # =============================
                    # ENTER ZONE → UP (STATE AWARE)
                    # =============================
                    if (self.inside_counter >= self.stable_frames
                            and not self.zone_active):

                        if self.weeder_position == "DOWN":
                            msg = "ZONE ENTER → UP"
                            print(msg)
                            self.log.append(msg)

                            self.send_command("xU0350")
                            self.weeder_position = "UP"
                            self.detection_count += 1
                        else:
                            msg = "ZONE ENTER → already UP (ignored)"
                            print(msg)
                            self.log.append(msg)

                        self.zone_active = True

                    # =============================
                    # EXIT ZONE → DOWN (STATE AWARE)
                    # =============================
                    if (self.outside_counter >= self.stable_frames
                            and self.zone_active):

                        if self.weeder_position == "UP":
                            msg = "ZONE EXIT → DOWN"
                            print(msg)
                            self.log.append(msg)

                            self.send_command("xD0350")
                            self.weeder_position = "DOWN"
                        else:
                            msg = "ZONE EXIT → already DOWN (ignored)"
                            print(msg)
                            self.log.append(msg)

                        self.zone_active = False

                self.frame = frame

        finally:
            pipeline.stop()
            print("Camera Stopped")