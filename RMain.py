import cv2
import torch
import re
import uuid
import time
from datetime import datetime
import easyocr
from ultralytics import YOLO
from pymongo import MongoClient
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

load_dotenv()
# ================= CLOUDINARY =================
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

def upload_image(path):
    try:
        res = cloudinary.uploader.upload(path)
        return res["secure_url"]
    except Exception as e:
        print("Upload error:", e)
        return None

# ================= MONGODB =================
client = MongoClient(os.getenv("MONGO_URI"))
db = client["traffic_db"]
collection = db["violations"]

def save_violation(image_url, location, violation_type, vehicle_number):
    vid = "VIO_" + str(uuid.uuid4())[:8]

    data = {
        "violation_id": vid,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location,
        "vehicle_number": vehicle_number,
        "violation_type": violation_type,
        "image_url": image_url
    }

    collection.insert_one(data)
    return vid

# ================= DUPLICATE CONTROL =================
last_violation_time = {}
COOLDOWN_SECONDS = 10

# ================= DEVICE =================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ================= MODELS =================
helmet_model = YOLO("runsHelmet/detect/Helmet_training/helmet_yolov8_1129/weights/best.pt")
vehicle_model = YOLO("runsVehicle/detect/Vehicle_training/vehicle_yolov8/weights/best.pt")
coco_model = YOLO("yolov8m.pt")
numberplate_model = YOLO("runsNum/best.pt")

reader = easyocr.Reader(['en'], gpu=False)

# ================= HELPERS =================
def draw_label(img, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, 0.5, 1)
    y1 = max(y1, h + 10)
    cv2.rectangle(img, (x1, y1 - h - 8), (x1 + w + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 4), font, 0.5, (255,255,255), 1)

def box_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

# ================= CORE =================
def process_frame(frame):
    try:
        output = frame.copy()
        violation_types = []
        detected_plate = "UNKNOWN"
        vid = None

        # VEHICLE
        vehicles = vehicle_model(frame, device=DEVICE, conf=0.4)
        motorcycles = []

        for box in vehicles[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            name = vehicle_model.names[int(box.cls[0])]

            if name.lower() == "motorcycle":
                motorcycles.append((x1,y1,x2,y2))

            cv2.rectangle(output,(x1,y1),(x2,y2),(255,0,0),1)
            draw_label(output,name.upper(),x1,y1,(255,0,0))

        # HELMET
        helmets = helmet_model(frame, device=DEVICE, conf=0.5)
        for box in helmets[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            name = helmet_model.names[int(box.cls[0])].lower()

            if any(x in name for x in ["no", "without"]):
                if "no_helmet" not in violation_types:
                    violation_types.append("no_helmet")
                color=(0,0,255)
                label="NO HELMET"
            else:
                color=(0,200,0)
                label="HELMET"

            cv2.rectangle(output,(x1,y1),(x2,y2),color,1)
            draw_label(output,label,x1,y1,color)

        # COCO
        coco = coco_model(frame, device=DEVICE, conf=0.4)
        persons, phones = [], []

        for r in coco:
            for box in r.boxes:
                name = coco_model.names[int(box.cls[0])]
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                if name=="person": persons.append((x1,y1,x2,y2))
                if name=="cell phone": phones.append((x1,y1,x2,y2))

        # TRIPLE RIDING
        for bike in motorcycles:
            mx1,my1,mx2,my2 = bike
            count = sum(1 for p in persons if mx1 < box_center(p)[0] < mx2 and my1 < box_center(p)[1] < my2)

            if count >= 3:
                if "triple_riding" not in violation_types:
                    violation_types.append("triple_riding")

                cv2.rectangle(output,(mx1,my1),(mx2,my2),(0,0,180),2)
                draw_label(output,"TRIPLE RIDING",mx1,my1,(0,0,180))

        # MOBILE USAGE (FIXED)
        for bike in motorcycles:
            mx1,my1,mx2,my2 = bike

            for person in persons:
                px1,py1,px2,py2 = person
                pcx,pcy = box_center(person)

                if mx1 < pcx < mx2 and my1 < pcy < my2:
                    for phone in phones:
                        fcx,fcy = box_center(phone)

                        if (
                            px1-20 < fcx < px2+20 and
                            py1-20 < fcy < py2+20 and
                            fcy < py2 - (py2-py1)*0.3
                        ):
                            if "mobile_usage" not in violation_types:
                                violation_types.append("mobile_usage")

                            cv2.rectangle(output,(px1,py1),(px2,py2),(0,255,255),2)
                            draw_label(output,"MOBILE USAGE",px1,py1,(0,255,255))

        # NUMBER PLATE
        plates = numberplate_model(frame, device=DEVICE, conf=0.4)
        for r in plates:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)

                if x2<=x1 or y2<=y1: continue
                crop = frame[y1:y2,x1:x2]

                if crop is None or crop.size==0: continue

                try:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                except:
                    continue

                results = reader.readtext(gray)
                for _,text,prob in results:
                    if prob>0.3:
                        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if len(clean)>=6:
                            detected_plate = clean
                            draw_label(output,clean,x1,y1,(0,255,0))

        # SAVE LOGIC
        if len(violation_types) > 0:

            key = f"{detected_plate}_{'_'.join(sorted(violation_types))}"
            now = time.time()

            if key in last_violation_time:
                if now - last_violation_time[key] < COOLDOWN_SECONDS:
                    return output, None

            last_violation_time[key] = now

            cv2.imwrite("violation.jpg", output)
            image_url = upload_image("violation.jpg")

            if image_url:
                vid = save_violation(image_url,"Camera 1",list(set(violation_types)),detected_plate)

        return output, vid

    except Exception as e:
        print("ERROR:", e)
        return frame, None


# ================= VIDEO =================
def process_video(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        yield process_frame(frame)
    cap.release()

# ================= IMAGE =================
def process_image(path):
    frame = cv2.imread(path)
    if frame is None:
        return None, None
    return process_frame(frame)