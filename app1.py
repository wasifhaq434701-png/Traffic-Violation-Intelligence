import streamlit as st
import tempfile
import cv2
from RMain import process_video, process_image
from pymongo import MongoClient

st.set_page_config(page_title="Traffic AI", layout="wide")
st.title("🚦 Traffic Violation Intelligence System")

uploaded_file = st.file_uploader("Upload Image or Video",
                                 type=["mp4","avi","mov","jpg","jpeg","png"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    path = tfile.name

    st.success("File uploaded")

    if st.button("Run Detection"):

        if "video" in uploaded_file.type:
            frame_placeholder = st.empty()

            for frame, vid in process_video(path):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame)

                if vid is not None:
                    st.warning(f"🚨 Violation: {vid}")

        else:
            result, vid = process_image(path)

            if result is not None:
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result)

                if vid is not None:
                    st.success(f"🚨 Violation ID: {vid}")
                else:
                    st.info("No violation detected")

# SEARCH
st.markdown("---")
st.subheader("🔍 Search Violation")

search_id = st.text_input("Enter ID")

if st.button("Search"):
    client = MongoClient("YOUR_CONNECTION_STRING")
    db = client["traffic_db"]
    col = db["violations"]

    res = col.find_one({"violation_id": search_id})

    if res:
        st.success("Found")
        st.write("📍 Location:", res["location"])
        st.write("🚗 Vehicle:", res["vehicle_number"])
        st.write("⚠️ Type:", res["violation_type"])
        st.write("🕒 Time:", res["timestamp"])
        st.image(res["image_url"])
    else:
        st.error("Not found")