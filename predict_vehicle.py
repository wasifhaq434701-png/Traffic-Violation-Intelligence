import cv2
from ultralytics import YOLO

def predict_video_live():
    """
    Live vehicle detection on video
    - Shows video with detections
    - Does NOT save anything
    """

    MODEL_PATH = "runsVehicle/detect/Vehicle_training/vehicle_yolov8/weights/best.pt"

    import sys
    input_path = sys.argv[1]
    cap = cv2.VideoCapture(input_path)


    # Load trained model
    model = YOLO(MODEL_PATH)

    # Live prediction (no saving)
    model.predict(
        source=cap,
        conf=0.4,
        iou=0.5,
        device="mps",   # Apple Silicon GPU
        show=True,      # 👈 THIS shows live video
        save=False      # 👈 THIS disables saving
    )

if __name__ == "__main__":
    predict_video_live()
