import cv2
from ultralytics import YOLO

# Load your trained helmet model
model = YOLO("runsHelmet/detect/Helmet_training/helmet_yolov8_1129/weights/best.pt")  # Change path if needed

# Path to your saved video
import sys

input_path = sys.argv[1]
cap = cv2.VideoCapture("trafficVid.mp4")
 # <-- Replace with your actual path

# Run prediction
model.predict(
    source=cap,
    conf=0.5,
    show=True,     # Show video with detections
    save=False,     # Do NOT save output
    device="mps"    # Use Apple Silicon GPU
)
