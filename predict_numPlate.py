import cv2
import easyocr
import re
from ultralytics import YOLO
import pandas as pd
# ================= SETTINGS =================
VIDEO_PATH = "trafficVid.mp4"          # video file
MODEL_PATH = "runsNum/best.pt"         # trained YOLO model
CONF_THRESHOLD = 0.4           # YOLO confidence
OCR_THRESHOLD = 0.30           # EasyOCR confidence
# ============================================

# Load YOLO model
model = YOLO(MODEL_PATH)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Storage
detected_plates = {}
current_plate = None

# Open video
import sys

input_path = sys.argv[1]
cap = cv2.VideoCapture(input_path)


if not cap.isOpened():
    print("ERROR: Cannot open video file")
    exit()

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # YOLO inference
    results = model(frame, conf=CONF_THRESHOLD)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            if float(box.conf) < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # Resize for better OCR
            plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2)

            # Mild preprocessing (SAFE for OCR)
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            # OCR (NO thresholding)
            ocr_results = reader.readtext(
                gray,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1
            )

            for (_, text, prob) in ocr_results:
                if prob < OCR_THRESHOLD:
                    continue

                # Clean text
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

                # Indian number plate regex
                pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$'
                if not re.match(pattern, clean_text):
                    continue

                current_plate = clean_text

                conf = float(prob)

# If plate seen before, only update if new confidence is higher
                if clean_text not in detected_plates:
                    detected_plates[clean_text] = {
                        "best_frame": frame_number,
                        "confidence": round(conf, 2)
                    }
                else:
                    if conf > detected_plates[clean_text]["confidence"]:
                        detected_plates[clean_text] = {
                            "best_frame": frame_number,
                            "confidence": round(conf, 2)
                        }


                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(frame, clean_text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

                print("DETECTED PLATE:", clean_text)

    # Display
    cv2.imshow("ANPR - YOLO + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n========= FINAL RESULTS =========")
print("Latest detected plate:", current_plate)
print("All detected plates:")
print(detected_plates)
df = pd.DataFrame.from_dict(detected_plates, orient='index')
df.to_csv('detected_plates.csv')