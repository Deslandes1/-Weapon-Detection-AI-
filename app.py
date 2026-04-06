import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import os
import requests
import base64

st.set_page_config(page_title="Weapon Detection AI - GlobalInternet.py", layout="wide")

# ----------------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------------
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.image("https://flagcdn.com/w320/ht.png", width=100)
        with col2:
            st.markdown("<h1 style='text-align: center;'>WEAPON DETECTION AI</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'><em>Real-time gun detection for public safety</em></p>", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='text-align: right;'>
                <b>GlobalInternet.py</b><br>
                Gesner Deslandes<br>
                Python Developer
            </div>
            """, unsafe_allow_html=True)
        st.divider()
        pwd = st.text_input("🔐 Enter password to unlock", type="password")
        if pwd == "20082010":
            st.session_state.authenticated = True
            st.rerun()
        elif pwd:
            st.error("Wrong password. Access denied.")
        return False
    return True

# ----------------------------------------------------------------------
# Load YOLO model (weapon detection)
# ----------------------------------------------------------------------
@st.cache_resource
def load_weapon_model():
    # Try to download a weapon detection model from a public URL
    model_path = "weapon_model.pt"
    if not os.path.exists(model_path):
        # Attempt to download from a reliable source (this is a placeholder, replace with actual URL)
        # For this example, we use a standard YOLOv8 model and filter for knives (demo)
        st.warning("Weapon model not found. Using fallback: detecting knives and scissors as potential weapons.")
        # Fallback: use standard YOLOv8 and filter for knife (class 43) and scissors (class 76)
        model = YOLO("yolov8n.pt")
        return model, True
    else:
        model = YOLO(model_path)
        return model, False

model, is_fallback = load_weapon_model()

# COCO class names for fallback mode
coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
weapon_keywords = ["knife", "scissors", "baseball bat", "hammer", "axe", "gun", "pistol", "rifle", "weapon"]

# ----------------------------------------------------------------------
# Video transformer for real-time detection
# ----------------------------------------------------------------------
class WeaponDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.is_fallback = is_fallback

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Run detection
        results = self.model(img)
        weapon_detected = False
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if self.is_fallback:
                        # In fallback mode, check if the detected class is a potential weapon
                        class_name = coco_names[cls] if cls < len(coco_names) else "unknown"
                        if any(keyword in class_name.lower() for keyword in weapon_keywords):
                            weapon_detected = True
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"{class_name} {conf:.2f}"
                            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    else:
                        # Custom weapon model – assume class 0 is weapon
                        if conf > 0.5:
                            weapon_detected = True
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"Weapon {conf:.2f}"
                            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        if weapon_detected:
            cv2.putText(img, "⚠️ WEAPON DETECTED ⚠️", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------------------------------------------------
# Main app after authentication
# ----------------------------------------------------------------------
if not check_password():
    st.stop()

# Display after login
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("https://flagcdn.com/w320/ht.png", width=100)
with col2:
    st.markdown("<h1 style='text-align: center;'>WEAPON DETECTION AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><em>Real-time gun detection for public safety</em></p>", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style='text-align: right;'>
        <b>GlobalInternet.py</b><br>
        Gesner Deslandes<br>
        Python Developer
    </div>
    """, unsafe_allow_html=True)
st.divider()

st.sidebar.image("https://flagcdn.com/w320/ht.png", width=100)
st.sidebar.title("Weapon Detection AI")
st.sidebar.markdown("**GlobalInternet.py**")
st.sidebar.markdown("Owner: Gesner Deslandes")
st.sidebar.markdown("📧 deslndes78@gmail.com | 📞 (509) 4738-5663")
st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Price")
st.sidebar.markdown("**$299 USD** – One‑time purchase (lifetime license)")
st.sidebar.markdown("---")
st.sidebar.info("How it works:\n- Click 'Start' below.\n- Grant camera permission.\n- The AI will highlight any weapon detected.\n- A red warning appears on screen.\n- Works on phones and computers.")

if is_fallback:
    st.warning("⚠️ Demo mode: Using standard model (detects knives, scissors, bats as potential weapons). For full accuracy, please provide a custom weapon detection model.")

st.markdown("### 📷 Live Camera Feed")
webrtc_ctx = webrtc_streamer(
    key="weapon-detection",
    video_transformer_factory=WeaponDetector,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.state.playing:
    st.success("✅ Camera is active. AI is watching for weapons.")
else:
    st.warning("⚠️ Camera is not started. Click 'Start' above.")

st.markdown("---")
st.markdown("© 2026 GlobalInternet.py – All rights reserved")
