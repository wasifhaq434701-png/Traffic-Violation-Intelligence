import uuid
from datetime import datetime
from pymongo import MongoClient

# 🔗 MongoDB Connection
client = MongoClient("YOUR_CONNECTION_STRING")
db = client["traffic_db"]
collection = db["violations"]


# 🚨 SAVE VIOLATION FUNCTION
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


# 🔍 GET VIOLATION FUNCTION
def get_violation(violation_id):
    result = collection.find_one({"violation_id": violation_id})

    if result:
        return {
            "id": result["violation_id"],
            "time": result["timestamp"],
            "location": result["location"],
            "vehicle": result["vehicle_number"],
            "type": result["violation_type"],
            "image": result["image_url"]
        }
    else:
        return None


# 🧪 TEST
if __name__ == "__main__":
    vid = save_violation(
        image_url="https://dummyimage.com/test.jpg",
        location="Camera 1",
        violation_type=["no_helmet"],
        vehicle_number="TS08AB1234"
    )

    print("✅ Saved ID:", vid)

    data = get_violation(vid)
    print("🔍 Retrieved:", data)