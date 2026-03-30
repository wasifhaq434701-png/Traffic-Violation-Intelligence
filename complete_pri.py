from ultralytics import YOLO
import cv2

# Load both trained models
vehicle_model = YOLO("runsVehicle/detect/Vehicle_training/vehicle_yolov8/weights/best.pt")
helmet_model = YOLO("runsHelmet/detect/Helmet_training/helmet_yolov8_1129/weights/best.pt")



# Open video
cap = cv2.VideoCapture("trafficVid2.mp4")

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

# Save output video
out = cv2.VideoWriter(
    "output_combined.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run both models
    vehicle_results = vehicle_model(frame, verbose=False)
    helmet_results  = helmet_model(frame, verbose=False)

    # Draw vehicle detections
    combined_frame = vehicle_results[0].plot()

    # Draw helmet detections manually (Red color)
    for box in helmet_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        label = f"Helmet {conf:.2f}"

        cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            combined_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # Show live
    cv2.imshow("Traffic AI System", combined_frame)

    # Save frame
    out.write(combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
