import uuid
from datetime import datetime
from pymongo import MongoClient

client = MongoClient("YOUR_CONNECTION_STRING")
db = client["traffic_db"]
collection = db["violations"]

def save_violation(image_url, location, violation_type, vehicle_number):
    violation_id = "VIO_" + str(uuid.uuid4())[:8]

    data = {
        "violation_id": violation_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location,
        "vehicle_number": vehicle_number,
        "violation_type": violation_type,
        "image_url": image_url
    }

    collection.insert_one(data)

    return violation_id


# TEST
vid = save_violation(
    image_url="https://dummyimage.com/test.jpg",
    location="Nizamabad",
    violation_type=["no_helmet"],
    vehicle_number="TS08AB1234"
)

print("Saved with ID:", vid)