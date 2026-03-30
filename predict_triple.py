from ultralytics import YOLO
import cv2
import torch

# ==========================================
# SETTINGS
# ==========================================

# IMAGE_PATH = "Images/Image7.jpg"
CONF_THRESHOLD = 0.4

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ==========================================
# LOAD MODEL (COCO PRETRAINED)
# ==========================================

model = YOLO("yolov8m.pt")

print("Using device:", DEVICE)

# ==========================================
# HELPER FUNCTION
# ==========================================

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# ==========================================
# LOAD IMAGE
# ==========================================

import sys

input_path = sys.argv[1]
image = cv2.imread(input_path)


if image is None:
    print("ERROR: Check image path.")
    exit()

output = image.copy()

# ==========================================
# RUN YOLO DETECTION
# ==========================================

results = model(image, device=DEVICE, conf=CONF_THRESHOLD)

persons = []
motorcycles = []
phones = []

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_name == "person":
            persons.append((x1, y1, x2, y2))

        elif class_name == "motorcycle":
            motorcycles.append((x1, y1, x2, y2))

        elif class_name == "cell phone":
            phones.append((x1, y1, x2, y2))

# ==========================================
# CHECK TRIPLE RIDING
# ==========================================

triple_detected = False

for bike in motorcycles:

    mx1, my1, mx2, my2 = bike
    person_count = 0
    matched_persons = []

    for person in persons:
        cx, cy = box_center(person)

        # Person center inside motorcycle box
        if mx1 < cx < mx2 and my1 < cy < my2:
            person_count += 1
            matched_persons.append(person)

    if person_count >= 3:
        triple_detected = True

        # Draw motorcycle box
        cv2.rectangle(output, (mx1, my1), (mx2, my2), (0, 0, 255), 3)

        cv2.putText(output,
                    f"TRIPLE RIDING ({person_count})",
                    (mx1, my1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3)

        # Draw persons
        for px1, py1, px2, py2 in matched_persons:
            cv2.rectangle(output, (px1, py1), (px2, py2),
                          (255, 0, 0), 2)

# ==========================================
# CHECK MOBILE USAGE
# ==========================================

mobile_detected = False

for bike in motorcycles:

    mx1, my1, mx2, my2 = bike

    for person in persons:
        pcx, pcy = box_center(person)

        # Person must be riding bike
        if mx1 < pcx < mx2 and my1 < pcy < my2:

            px1, py1, px2, py2 = person

            for phone in phones:
                fcx, fcy = box_center(phone)

                # Phone center inside rider box
                if px1 < fcx < px2 and py1 < fcy < py2:
                    mobile_detected = True

                    cv2.rectangle(output,
                                  (px1, py1),
                                  (px2, py2),
                                  (0, 255, 255),
                                  3)

                    cv2.putText(output,
                                "MOBILE USAGE",
                                (px1, py1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 255, 255),
                                3)

# ==========================================
# DISPLAY FINAL RESULT
# ==========================================



cv2.imshow("Triple + Mobile Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
