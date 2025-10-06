#!/usr/bin/env python3
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import datetime
import blobconverter
import requests, datetime, os
from datetime import timezone
from utils import ARUCO_DICT
import math
import socket
import requests
from collections import deque
from gpiozero import Servo
from time import sleep
import threading,queue
import time
from PIL import Image, ImageDraw, ImageFont, ImageOps
import st7735
from smbus2 import SMBus
from bme280 import BME280
from enviroplus import gas
from ltr559 import LTR559
import socketio

# -------------------- Global Variables --------------------
SERVER_URL = "http://10.88.42.96:5000" # replace
SOCKETIO_URL = "http://10.88.42.96:5000"  # SocketIO server URL
DEVICE_ID = "pi_device_001"  # Unique device identifier


# -------------------- Sensor Code --------------------
ltr559 = LTR559()
SERVER_URL_SENSORS = "http://192.168.1.237:5000"
LCD_ROTATION = 270
LCD_SPI_SPEED_HZ = 4000000
LCD_FPS = 15
API_KEY = None

# shared state
state = {
    "ip" : " 0.0.0.0",
    "air_c": 0.0,
    "cpu_c": 0.0,
    "display_mode": "default",  # default, ip, targets, temp, sensors
    "detections": [],  # Current detections for targets mode
}
state_lock = threading.Lock()

# SocketIO client
sio = socketio.Client()

# SocketIO Event Handlers
@sio.event
def connect():
    print(f"[SocketIO] Connected to server with device_id: {DEVICE_ID}")
    sio.emit('join_room', {'device_id': DEVICE_ID})

@sio.event
def disconnect():
    print("[SocketIO] Disconnected from server")

@sio.event
def set_display(data):
    """Handle display mode change from GCS"""
    display_mode = data.get('mode', 'default')
    with state_lock:
        state['display_mode'] = display_mode
    print(f"[SocketIO] Display mode changed to: {display_mode}")

@sio.event
def connect_error(data):
    print(f"[SocketIO] Connection error: {data}")

# Motor threading primitives
motor_queue = queue.Queue(maxsize=1)   # only one command at a time
motor_busy = threading.Event()         # indicates motor is running
motor_flag = 0                         # flag to prevent continuous triggering
# Init LCD

lcd = st7735.ST7735(
    port=0,
    cs=1,
    dc="GPIO9",
    backlight="GPIO12",
    rotation=LCD_ROTATION,
    spi_speed_hz=LCD_SPI_SPEED_HZ
)
lcd.begin()
LCD_W,LCD_H = lcd.width, lcd.height
font = ImageFont.load_default()

# Init sensors

# Values for MX + B straight line from datasheet
red_m = -0.867 # log(0.01)- log(4) / log(1000) - log(1) using CO line
ox_m = 1.015 #  log(0.05) -log(42) / log(0.03) - log(6.5) using NO2 line
nh3_m = -0.567 # log(0.045) - log(0.8) / log(160) - log(1) using NH3 line

# Values are from calibration data
red_R0 = 435816.37
ox_R0 = 6081.13
nh3_R0 = 220543.21

# Y intercepts for the lines
red_b = 0.602 # log(4) - m*log(1)
ox_b = 0.797 # log(0.3) - m*log(0.05)
nh3_b = -0.097 # log(0.8) - m*log(1)

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        
    except Exception as e:
        print("Error getting IP address:", e)
        return None
    return ip_address

def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.read()
            temp_c = float(temp_str) / 1000.0
    except Exception as e:
        return 0.0
    return temp_c

def post_json(payload):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    r = requests.post(SERVER_URL + "/api/sensors", headers=headers, data=json.dumps(payload), timeout=5)
    r.raise_for_status()

def calculate_ppm(rs, r0, m, b):
    if rs <= 0 or r0 <= 0:
        return 0.0
    ratio = rs / r0
    # Rearrange formula to calculate ppm from rs/r0
    log_ppm = (math.log10(ratio) - b) / m
    ppm = 10 ** log_ppm
    return ppm

def read_lux():
    lux = ltr559.get_lux()
    if lux is None or lux < 0:
        return 0.0
    return float(lux)

def touch_screen():
    """Cycle display modes on touch"""
    with state_lock:
        mode = state.get("display_mode", "default")

        if mode == "ip":
            state["display_mode"] = "targets"
        elif mode == "targets":
            state["display_mode"] = "temp"
        elif mode == "temp":
            state["display_mode"] = "default"
        else:
            state["display_mode"] = "ip"

bus = SMBus(1)
bme280 = BME280(i2c_dev=bus)

# ---------------- Sensor thread --------------------
def sensor_post_loop():
    
    ip = get_ip_address()
    with state_lock:
        state["ip"] = ip

    while True:
        try:
            g = gas.read_all()

            # Get readings

            red_RS = g.reducing
            ox_RS = g.oxidising
            nh3_RS = g.nh3

            # Calculate PPM

            ppm_CO = calculate_ppm(red_RS, red_R0, red_m, red_b)
            ppm_NO2 = calculate_ppm(ox_RS, ox_R0, ox_m, ox_b)
            ppm_NH3 = calculate_ppm(nh3_RS, nh3_R0, nh3_m, nh3_b)

            # Get light level
            lux = read_lux()

            # Get temperature, pressure, humidity
            temperature = float(bme280.get_temperature())
            pressure = float(bme280.get_pressure())
            humidity = float(bme280.get_humidity())

            # Update LCD display
            cpu_c = read_cpu_temp()
            with state_lock:
                state["air_c"] = temperature
                state["cpu_c"] = cpu_c
                # Store sensor data for sensors display mode
                state["sensor_data"] = {
                    "co_ppm": ppm_CO,
                    "no2_ppm": ppm_NO2,
                    "nh3_ppm": ppm_NH3,
                    "light_lux": lux,
                    "pressure_hpa": pressure,
                    "humidity_pct": humidity
                }
            payload = {
                "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
                "co_ppm": ppm_CO,
                "no2_ppm": ppm_NO2,
                "nh3_ppm": ppm_NH3,
                "light_lux": lux,
                "temp_c": temperature,
                "pressure_hpa": pressure,
                "humidity_pct": humidity,
            }
            post_json(payload)
        except Exception:
            # Dont crash, just try again in a second
            pass
        time.sleep(1.0) #why is this here? - To give the CPU some rest. Dont want to hammer the sensors too much.

# ---------------- Motor thread --------------------

def _motor_worker():
    while True:
        cmd = motor_queue.get()  # blocks until a command arrives
        motor_busy.set()  # Set immediately after dequeuing
        print(f"[motor] Sequence started - command: {cmd}")
        try:
            if cmd == "gauge_correction":
                # Extend phase
                print("[motor] Extend phase: start")
                rotate_servo(10.0, True)
                print("[motor] Extend phase: end")
                
                # Pause phase
                print("[motor] Pause phase: 0.5s")
                time.sleep(0.5)
                
                # Retract phase
                print("[motor] Retract phase: start")
                rotate_servo(10.0, False)
                print("[motor] Retract phase: end")
                
                print("[motor] Sequence completed successfully")
        except Exception as e:
            print(f"[motor] Sequence error: {e}")
        finally:
            # Guaranteed final stop
            _force_motor_stop()
            motor_busy.clear()
            motor_flag = 0  # Reset flag to allow future triggers
            print("[motor] Motor flag reset - ready for next trigger")
            motor_queue.task_done()

# ---------------- Display thread --------------------

def display_loop():
        last_touch = 0
        armed = True
        PROX_HIGH = 10
        PROX_LOW = 5
        PROX_COOLDOWN = 0.5  # seconds

        while True:

            try:
                ltr559.update_sensor()
                prox = ltr559.get_proximity() or 0
            except Exception:
                prox = 0
            
            print (f"Proximity: {prox}")
            now = time.time()
            if armed and prox > PROX_HIGH and (now - last_touch) > PROX_COOLDOWN:
                touch_screen()
                armed = False
                last_touch = now
            elif not armed and prox < PROX_LOW:
                armed = True


            try:
                frame = lcd_queue.get(timeout=1.0)
            except queue.Empty:
                frame = None
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil = ImageOps.fit(pil, (LCD_W, LCD_H), Image.BILINEAR)
            draw = ImageDraw.Draw(pil)
            
            with state_lock:
                ip = state["ip"]
                air_c = state["air_c"]
                cpu_c = state["cpu_c"]
                display_mode = state["display_mode"]
                detections = state["detections"]
            
            # Display based on mode
            if display_mode == "ip":
                draw.text((2, 2), f"IP: {ip}", font=font, fill=(255, 255, 255))
                draw.text((2, 16), f"Mode: IP", font=font, fill=(0, 255, 0))
                
            elif display_mode == "targets":
                draw.text((2, 2), f"TARGETS: {len(detections)}", font=font, fill=(255, 255, 0))
                draw.text((2, 16), f"Mode: TARGETS", font=font, fill=(0, 255, 0))
                # Show detection details
                y_offset = 30
                for i, det in enumerate(detections[:3]):  # Show max 3 detections
                    label = det.get('label', 'Unknown')
                    conf = det.get('confidence', 0)
                    draw.text((2, y_offset), f"{label}: {conf:.1f}%", font=font, fill=(255, 255, 255))
                    y_offset += 14
                    
            elif display_mode == "temp":
                draw.text((2, 2), f"Air: {air_c:.1f}C", font=font, fill=(255, 255, 255))
                draw.text((2, 16), f"CPU: {cpu_c:.1f}C", font=font, fill=(255, 255, 255))
                draw.text((2, 30), f"Mode: TEMP", font=font, fill=(0, 255, 0))
                
                
            else:  # default mode
                draw.rectangle((0,0,LCD_W,14), fill =(0,0,0))
                draw.text((2, 2), f"IP:{ip}", font=font, fill=(255,255,255))
                draw.rectangle((0,LCD_H-14,LCD_W,LCD_H), fill =(0,0,0))
                draw.text((2, LCD_H-12), f"Air:{air_c:.1f}C CPU:{cpu_c:.1f}C", font=font, fill=(255,255,255))
            
            lcd.display(pil)
# SocketIO connection management
def socketio_connect():
    """Connect to SocketIO server with retry logic"""
    while True:
        try:
            sio.connect(SOCKETIO_URL)
            break
        except Exception as e:
            print(f"[SocketIO] Connection failed: {e}, retrying in 5 seconds...")
            time.sleep(5)

# Start threads
lcd_queue = queue.Queue(maxsize=3)
threading.Thread(target=sensor_post_loop, daemon=True).start()
threading.Thread(target=display_loop, daemon=True).start()
threading.Thread(target=_motor_worker, daemon=True).start()
threading.Thread(target=socketio_connect, daemon=True).start()


# -------------------- End Sensor Code -------------------

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='yolov4_tiny_coco_416x416', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='json/yolov4-tiny.json', type=str)
args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(W, H)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# deque to handle auto popping, change len for larger moving avg
readings = deque(maxlen=30)  # 10 sample window
angles   = deque(maxlen=30)
MOTOR_HEADER2_GPIO_PIN = 13

def rotate_servo(seconds: float, clockwise: bool = True) -> None:
    servo = Servo(
        MOTOR_HEADER2_GPIO_PIN,
        frame_width=0.02,        # 50 Hz
        min_pulse_width=0.0010,  # 1.0 ms
        max_pulse_width=0.0020   # 2.0 ms
    )
    try:
        if clockwise:
            servo.max()   # run one direction
        else:
            servo.min()   # run the other direction
        sleep(seconds)    # KEEP PWM ON for the whole duration
        servo.mid()       # stop
    finally:
        servo.close()

def _force_motor_stop():
    """Force motor to stop completely - called on any error or completion"""
    try:
        servo = Servo(MOTOR_HEADER2_GPIO_PIN)
        servo.mid()  # Neutral position
        servo.close()  # Release GPIO
        print("[motor] Force stop completed")
    except Exception as e:
        print(f"[motor] Force stop error: {e}")

def update_gauge(center, tip, tail):

    reading_tt, angle_tt = gauge_reading(tail, tip) # tip to tail readings
    reading_ct, angle_ct = gauge_reading(center, tip) # center to tip readings
    reading_tc, angle_tc = gauge_reading(tail, center) # tail to center readings
    #print(f"RAW Readings: {reading_tt:.2f}, {reading_ct:.2f}, {reading_tc:.2f}")
    # append to deque
    readings.append(reading_tt)
    readings.append(reading_ct)
    readings.append(reading_tc)

    angles.append(angle_tt)
    angles.append(angle_ct)
    angles.append(angle_tc)
    
    return np.median(readings), np.median(angles) # send 

    

# Linking
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    """
    frame - Frame from the OAK-D Lite stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return: frame with axis drawn
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
    gray, aruco_dict, parameters=parameters)


    marker_id = None
    pose = None

    if ids is not None and len(corners) > 0:
        # take the first (and only expected) marker
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners[0], 0.20, matrix_coefficients, distortion_coefficients)

        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)

        marker_id = int(ids[0][0])  # ensure plain int, not numpy type
        pose = tvec.flatten().tolist()

        #print(f"[Pose] ID={marker_id}, rvec={pose['rvec']}, tvec={pose['tvec']}")

    return frame, marker_id, pose

def send_detection(target_type, details, frame):
    # Encode frame as JPEG in memory, not saving to disk here.
    _, buffer = cv2.imencode(".jpg", frame)

    # Put image bytes in the "file" field
    files = {
        "file": ("frame.jpg", buffer.tobytes(), "image/jpeg")
    }

    data = {
        "ts": datetime.datetime.now().isoformat(),
        "target_type": target_type,
        "details": json.dumps(details)
    }
    try:
        resp = requests.post(
            SERVER_URL + "/api/targets",
            files=files,
            data=data,
            timeout=1
        )
        #print("Status:", resp.status_code)
        #print("Response:", resp.json())
    except Exception as e:
        print("POST failed::", e)

def bbox_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return (cx, cy)
def gauge_reading(center, tip, min_val=0, max_val=10, theta_min=225, theta_max=315):
    """
    Calculate gauge reading based on detected needle points.
    """

    dx = tip[0] - center[0]
    dy = tip[1] - center[1]

    # Get angle in degrees (Desmos-style atan2)
    t1 = math.degrees(math.atan2(-dy, dx))  

    # Match Desmos: transform to gauge-relative angle
    t2 = -t1 + 225
    if t2 >= 360:
        t2 -= 360
    
    # Map to gauge value (0 to 10 over 270Â°)
    value = (t2 / 270) * 10

    #print(f"Gauge reading: {value:.2f} bar (angle {t2:.1f}Â°)")
    return value, t2

# ARUCO CODE:
aruco_dict_type = ARUCO_DICT["DICT_5X5_100"]
calibration_matrix_path = "./calibration_matrix.npy"
distortion_coefficients_path = "./distortion_coefficients.npy"

k = np.load(calibration_matrix_path)
d = np.load(distortion_coefficients_path)

points = {"Center": (0,0), "Tip": (0,0), "Tail": (0,0)}
# Connect to device and start pipeline
while True:
    try:
        with dai.Device(pipeline) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            startTime = time.monotonic()
            counter = 0
            color2 = (255, 255, 255)

            # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
            def frameNorm(frame, bbox):
                normVals = np.full(len(bbox), frame.shape[0])
                normVals[::2] = frame.shape[1]
                return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

            def displayFrame(name, frame, detections):
                color = (255, 0, 0)
                for detection in detections:
                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                # Show the frame
                cv2.imshow(name, frame)

            while True:
                inRgb = qRgb.get()
                inDet = qDet.get()

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                    cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                                (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

                detections = []
                if inDet is not None:
                    detections = inDet.detections
                    counter += 1
                
                if frame is not None:
                    # Update detections in shared state for targets display mode
                    current_detections = []
                    for detection in detections:
                        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        detection_info = {
                            'label': labels[detection.label],
                            'confidence': detection.confidence,
                            'bbox': bbox.tolist()
                        }
                        current_detections.append(detection_info)
                        
                        cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
                        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    
                    # Update shared state with current detections
                    with state_lock:
                        state["detections"] = current_detections

                    try:
                        lcd_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass
                    
                    # Draw bboxes for human operator
                    
                    displayFrame("rgb", frame, detections)
                    if not detections:
                        detail= {
                            "id": "livedata",
                        }
                        send_detection("livedata", detail, frame)
                    else:
                        # Send detections to server
                        for det in detections:
                            bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
                            label = labels[det.label]
                            
                            if label in ["open-valve", "closed-valve"]:
                                #displayFrame("rgb", output, detections)

                                details = {
                                    "state": "open" if "open" in label else "closed",
                                    "confidence": round(det.confidence, 2),
                                    "bbox": bbox.tolist()
                                }
                                send_detection("valve", details, frame)
                            elif label == "Gauge":
                                for det in detections:
                                    #displayFrame("rgb", output, detections)
                                    bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
                                    label = labels[det.label]
                                    if label in ["Center", "Tip", "Tail"]:
                                        points[label] = bbox_center(bbox.tolist())
                                    
                                    if labels != "Gauge":
                                        if (points["Center"] != (0,0) and
                                            points["Tip"] != (0,0) and
                                            points["Tail"] != (0,0)):

                                            center = points["Center"]
                                            tip = points["Tip"]
                                            tail = points["Tail"]
                                            
                                            # reading, angle = gauge_reading(center, tip,
                                            #     min_val=0, max_val=10, 
                                            #     theta_min=270, theta_max=0
                                            # )
                                            reading, angle = update_gauge(center, tip, tail)
                                            #print(f"Avg reading: {reading:.2f} bar (angle {angle:.1f}Â°)")
                                            details = {
                                                "id": "guage",
                                                "reading_bar": round(reading, 2),
                                                "confidence": round(det.confidence, 2),
                                                "bbox": bbox.tolist()
                                            }
                                            
                                            send_detection("gauge", details, frame)

                                            if reading < 2.0 and motor_flag == 0:
                                                # Simple gating: check flag and enqueue
                                                try:
                                                    motor_queue.put_nowait("gauge_correction")
                                                    motor_flag = 1  # set flag to prevent continuous triggering
                                                    print(f"[motor] Command enqueued - reading: {reading:.2f}")
                                                except queue.Full:
                                                    print("[motor] Command rejected - queue full")
                                            #print(f"Gauge reading: {reading:.2f} bar (angle {angle:.1f}Â°)")

                            elif label == "ARUCO":
                                output, marker_id, pose = pose_estimation(frame, aruco_dict_type, k, d)
                                #displayFrame("rgb", output, detections)
                                if marker_id is not None:
                                    details = {
                                        "id": marker_id,
                                        "pose": pose,
                                        "confidence": round(det.confidence, 2),
                                        "bbox": bbox.tolist()
                                    }
                                    send_detection("aruco", details, frame)
                                    #print(f"[Pose] ID={marker_id}, position={pose}")
                if cv2.waitKey(1) == ord('q'):
                    break
    except Exception as e:
        print("Detection processing error:", e)
        # Just wait a second
        time.sleep(1.0)
        # Continue - device will be reconnected in the next loop
                        
                        

                        


        
