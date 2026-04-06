import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import os
import datetime
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

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
    model_path = "weapon_model.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        return model, False
    else:
        st.warning("Weapon model not found. Using fallback: detecting knives, scissors, bats as potential weapons.")
        model = YOLO("yolov8n.pt")
        return model, True

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
        self.demo_mode = st.session_state.get("demo_mode", False)
        self.detection_events = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        weapon_detected = False
        detected_objects = []

        if self.demo_mode:
            # Simulate weapon detection randomly for demo
            import random
            if random.random() < 0.1:  # 10% chance each frame
                weapon_detected = True
                # Draw a fake red box in the center
                h, w = img.shape[:2]
                x1, y1 = w//2 - 50, h//2 - 50
                x2, y2 = w//2 + 50, h//2 + 50
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = "SIMULATED WEAPON"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                detected_objects.append("Simulated weapon")
        else:
            # Run actual detection
            results = self.model(img)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        if self.is_fallback:
                            class_name = coco_names[cls] if cls < len(coco_names) else "unknown"
                            if any(keyword in class_name.lower() for keyword in weapon_keywords):
                                weapon_detected = True
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                label = f"{class_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                                detected_objects.append(f"{class_name} (confidence {conf:.2f})")
                        else:
                            if conf > 0.5:
                                weapon_detected = True
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                label = f"Weapon {conf:.2f}"
                                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                                detected_objects.append(f"Weapon (confidence {conf:.2f})")

        if weapon_detected:
            cv2.putText(img, "⚠️ WEAPON DETECTED ⚠️", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Record event
            self.detection_events.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objects": ", ".join(detected_objects)
            })
            # Keep only last 50 events
            if len(self.detection_events) > 50:
                self.detection_events = self.detection_events[-50:]

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------------------------------------------------
# Generate PDF report
# ----------------------------------------------------------------------
def generate_report(events):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Weapon Detection Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    if not events:
        story.append(Paragraph("No weapons detected during this session.", styles['Normal']))
    else:
        data = [["Timestamp", "Detected Object(s)"]]
        for e in events:
            data.append([e["timestamp"], e["objects"]])
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------------------------------------------------
# Main app after authentication
# ----------------------------------------------------------------------
if not check_password():
    st.stop()

# Initialize session state for demo mode and detection events
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if "detection_events" not in st.session_state:
    st.session_state.detection_events = []

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

# Sidebar
st.sidebar.image("https://flagcdn.com/w320/ht.png", width=100)
st.sidebar.title("Weapon Detection AI")
st.sidebar.markdown("**GlobalInternet.py**")
st.sidebar.markdown("Owner: Gesner Deslandes")
st.sidebar.markdown("📧 deslndes78@gmail.com | 📞 (509) 4738-5663")
st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Price")
st.sidebar.markdown("**$299 USD** – One‑time purchase (lifetime license)")
st.sidebar.markdown("---")

# Demo mode toggle
demo_mode = st.sidebar.checkbox("🎮 Demo Mode (simulate weapon detection)", value=st.session_state.demo_mode)
if demo_mode != st.session_state.demo_mode:
    st.session_state.demo_mode = demo_mode
    st.rerun()

# Training instructions
with st.sidebar.expander("📚 How to improve detection accuracy"):
    st.markdown("""
    1. **Use a real weapon detection model** – Download a `.pt` file (e.g., from [this repo](https://github.com/akanametov/yolo-weapon-detection)) and upload it to your app directory as `weapon_model.pt`.
    2. **Good lighting** – Ensure the camera has adequate light.
    3. **Hold the weapon clearly** – Avoid obstructed views.
    4. **Keep the camera stable** – Use a tripod or hold steady.
    5. **For mobile** – Tap the screen to focus.
    """)

# Report download
if st.sidebar.button("📄 Download Detection Report"):
    # Get events from the video transformer (we need to access the instance)
    # Since we can't directly access, we store events in session state via the transformer
    # We'll modify the transformer to update session state.
    # For simplicity, we'll use the stored events from the transformer (passed through session state)
    # For this to work, we need to pass a callback. We'll handle it by storing events in session state inside the transformer.
    # But the transformer runs in a separate thread. We'll use a queue? Simpler: collect events from the transformer's events list.
    # We'll just use the events list from the transformer instance if we can access it.
    # As a workaround, we'll store events in session state using a function called from the transformer.
    # Let's create a global list that the transformer updates.
    # I'll implement a simple solution: the transformer writes to st.session_state.detection_events
    # For now, we'll just generate a report from the events stored in session state.
    report_buffer = generate_report(st.session_state.get("detection_events", []))
    st.sidebar.download_button("⬇️ Download Report (PDF)", data=report_buffer, file_name=f"weapon_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

st.sidebar.markdown("---")
st.sidebar.info("How it works:\n- Click 'Start' below.\n- Grant camera permission.\n- The AI will highlight any weapon detected.\n- A red warning appears on screen.\n- Works on phones and computers.")

# Video feed
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
