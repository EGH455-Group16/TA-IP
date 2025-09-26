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
import blobconverter
import requests, datetime, os
from utils import ARUCO_DICT
import math
from collections import deque
from gpiozero import Servo
from time import sleep
SERVER_URL = "http://10.88.60.164:5001/api/target-data"   # replace <server-ip>
FRAME_PATH = "/static/frame.jpg"                        # or ./static/frame.jpg

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='yolov4_tiny_coco_416x416', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='json/yolov4-tiny.json', type=str)
args = parser.parse_args()
motor_flag = 0

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
def test_sensor_api_valid_data():
    """Test sensor API with valid data"""
    payload = {
        "timestamp": "2025-01-15T10:30:00Z",
        "co_ppm": 1.5,
        "no2_ppm": 0.8,
        "nh3_ppm": 0.3,
        "light_lux": 500,
        "temp_c": 22.5,
        "pressure_hpa": 1013.25,
        "humidity_pct": 60.0,
        "source": "ur_gay"
    }
    try:
        # Best way: requests sets Content-Type automatically
        resp = requests.post(
            "http://10.88.60.164:5001/api/sensors",
            json=payload
        )
        print("Status:", resp.status_code)
        print("Response:", resp.json())
    except Exception as e:
        print("POST failed:", e)

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
            "http://10.88.60.164:5000/api/targets",
            files=files,
            data=data,
            timeout=1
        )
        print("Status:", resp.status_code)
        print("Response:", resp.json())
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
    
    # Map to gauge value (0 to 10 over 270°)
    value = (t2 / 270) * 10

    #print(f"Gauge reading: {value:.2f} bar (angle {t2:.1f}°)")
    return value, t2

# ARUCO CODE:
aruco_dict_type = ARUCO_DICT["DICT_5X5_100"]
calibration_matrix_path = "./calibration_matrix.npy"
distortion_coefficients_path = "./distortion_coefficients.npy"

k = np.load(calibration_matrix_path)
d = np.load(distortion_coefficients_path)

points = {"Center": (0,0), "Tip": (0,0), "Tail": (0,0)}
# Connect to device and start pipeline
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
                                    #print(f"Avg reading: {reading:.2f} bar (angle {angle:.1f}°)")
                                    details = {
                                        "id": "guage",
                                        "reading_bar": round(reading, 2),
                                        "confidence": round(det.confidence, 2),
                                        "bbox": bbox.tolist()
                                    }
                                    
                                    send_detection("gauge", details, frame)

                                    if reading < 2.0 and motor_flag == 0:
                                        rotate_servo(10.0, True)
                                        sleep(0.5)
                                        rotate_servo(10.0, False)
                                        motor_flag = 1
                                    #print(f"Gauge reading: {reading:.2f} bar (angle {angle:.1f}°)")

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