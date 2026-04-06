import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import os
import datetime
import io
import random
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title="Weapon Detection AI - GlobalInternet.py", layout="wide")

# ----------------------------------------------------------------------
# Multi-language dictionary
# ----------------------------------------------------------------------
lang_dict = {
    "en": {
        "app_title": "WEAPON DETECTION AI",
        "app_subtitle": "Concealed weapon detection for public safety",
        "login_title": "WEAPON DETECTION AI",
        "login_sub": "Concealed weapon detection for public safety",
        "password_placeholder": "🔐 Enter password to unlock",
        "wrong_password": "Wrong password. Access denied.",
        "logout": "🚪 Logout",
        "price": "💰 Price",
        "price_value": "$299 USD – One‑time purchase (lifetime license)",
        "demo_mode": "🎮 Demo Mode (simulate weapon detection)",
        "training_title": "📚 How to get a real weapon detection model",
        "training_text": """
        1. **Download a pre‑trained model** – Visit [Roboflow Universe](https://universe.roboflow.com/) and search for "weapon detection". Download a YOLOv8 PyTorch model (`.pt` file).
        2. **Train your own** – Use [Ultralytics YOLOv8](https://docs.ultralytics.com/) and a weapon dataset.
        3. **Upload the model** below and the app will instantly use it for real concealed weapon detection.
        """,
        "upload_model": "📤 Upload your weapon detection model (.pt)",
        "upload_success": "✅ Model uploaded and loaded successfully!",
        "upload_error": "❌ Invalid model file. Please upload a valid .pt file.",
        "current_model": "🔍 Current model:",
        "fallback_model": "Standard fallback (knives, scissors, bats)",
        "custom_model": "Custom uploaded weapon model",
        "download_report": "📄 Download Detection Report",
        "report_filename": "weapon_report",
        "how_it_works": "How it works:\n- Click 'Start' below.\n- Grant camera permission.\n- The AI will highlight any weapon near a person as 'CONCEALED'.\n- A red warning appears on screen.\n- Works on phones and computers.",
        "camera_feed": "📷 Live Camera Feed",
        "camera_active": "✅ Camera is active. AI is watching for weapons.",
        "camera_inactive": "⚠️ Camera is not started. Click 'Start' above.",
        "model_warning": "⚠️ Using standard model (detects knives, scissors, bats). Upload a custom weapon model for real concealed detection.",
        "dismiss": "Dismiss",
        "weapon_detected": "⚠️ WEAPON DETECTED ⚠️",
        "concealed_on_person": "CONCEALED ON PERSON",
        "simulated_weapon": "SIMULATED WEAPON",
        "report_title": "Weapon Detection Report",
        "report_generated": "Generated",
        "no_weapons": "No weapons detected during this session.",
        "report_table_headers": ["Timestamp", "Detected Object(s)", "Concealed"],
        "yes": "Yes",
        "no": "No"
    },
    "fr": {
        "app_title": "IA DE DÉTECTION D'ARMES",
        "app_subtitle": "Détection d'armes dissimulées pour la sécurité publique",
        "login_title": "IA DE DÉTECTION D'ARMES",
        "login_sub": "Détection d'armes dissimulées pour la sécurité publique",
        "password_placeholder": "🔐 Entrez le mot de passe pour déverrouiller",
        "wrong_password": "Mot de passe incorrect. Accès refusé.",
        "logout": "🚪 Déconnexion",
        "price": "💰 Prix",
        "price_value": "299 $ USD – Achat unique (licence à vie)",
        "demo_mode": "🎮 Mode Démo (simuler la détection d'armes)",
        "training_title": "📚 Comment obtenir un vrai modèle de détection d'armes",
        "training_text": """
        1. **Téléchargez un modèle pré-entraîné** – Visitez [Roboflow Universe](https://universe.roboflow.com/) et recherchez "weapon detection". Téléchargez un modèle YOLOv8 PyTorch (fichier `.pt`).
        2. **Entraînez le vôtre** – Utilisez [Ultralytics YOLOv8](https://docs.ultralytics.com/) et un jeu de données d'armes.
        3. **Téléchargez le modèle** ci-dessous et l'application l'utilisera immédiatement pour une détection réelle des armes dissimulées.
        """,
        "upload_model": "📤 Téléchargez votre modèle de détection d'armes (.pt)",
        "upload_success": "✅ Modèle téléchargé et chargé avec succès !",
        "upload_error": "❌ Fichier de modèle invalide. Veuillez télécharger un fichier .pt valide.",
        "current_model": "🔍 Modèle actuel :",
        "fallback_model": "Modèle standard de secours (couteaux, ciseaux, battes)",
        "custom_model": "Modèle personnalisé téléchargé",
        "download_report": "📄 Télécharger le rapport de détection",
        "report_filename": "rapport_armes",
        "how_it_works": "Comment ça marche :\n- Cliquez sur 'Démarrer' ci-dessous.\n- Autorisez l'accès à la caméra.\n- L'IA mettra en évidence toute arme près d'une personne comme 'DISSIMULÉE'.\n- Un avertissement rouge apparaît à l'écran.\n- Fonctionne sur téléphones et ordinateurs.",
        "camera_feed": "📷 Flux caméra en direct",
        "camera_active": "✅ Caméra active. L'IA surveille les armes.",
        "camera_inactive": "⚠️ Caméra non démarrée. Cliquez sur 'Démarrer' ci-dessus.",
        "model_warning": "⚠️ Utilisation du modèle standard (détecte couteaux, ciseaux, battes). Téléchargez un modèle personnalisé pour une détection réelle.",
        "dismiss": "Ignorer",
        "weapon_detected": "⚠️ ARME DÉTECTÉE ⚠️",
        "concealed_on_person": "DISSIMULÉE SUR UNE PERSONNE",
        "simulated_weapon": "ARME SIMULÉE",
        "report_title": "Rapport de détection d'armes",
        "report_generated": "Généré le",
        "no_weapons": "Aucune arme détectée pendant cette session.",
        "report_table_headers": ["Horodatage", "Objet(s) détecté(s)", "Dissimulé"],
        "yes": "Oui",
        "no": "Non"
    },
    "es": {
        "app_title": "IA DE DETECCIÓN DE ARMAS",
        "app_subtitle": "Detección de armas ocultas para la seguridad pública",
        "login_title": "IA DE DETECCIÓN DE ARMAS",
        "login_sub": "Detección de armas ocultas para la seguridad pública",
        "password_placeholder": "🔐 Ingrese la contraseña para desbloquear",
        "wrong_password": "Contraseña incorrecta. Acceso denegado.",
        "logout": "🚪 Cerrar sesión",
        "price": "💰 Precio",
        "price_value": "$299 USD – Compra única (licencia de por vida)",
        "demo_mode": "🎮 Modo Demo (simular detección de armas)",
        "training_title": "📚 Cómo obtener un modelo real de detección de armas",
        "training_text": """
        1. **Descargue un modelo preentrenado** – Visite [Roboflow Universe](https://universe.roboflow.com/) y busque "weapon detection". Descargue un modelo YOLOv8 PyTorch (archivo `.pt`).
        2. **Entrene el suyo** – Use [Ultralytics YOLOv8](https://docs.ultralytics.com/) y un conjunto de datos de armas.
        3. **Cargue el modelo** abajo y la aplicación lo usará inmediatamente para detección real de armas ocultas.
        """,
        "upload_model": "📤 Cargue su modelo de detección de armas (.pt)",
        "upload_success": "✅ Modelo cargado exitosamente",
        "upload_error": "❌ Archivo de modelo inválido. Cargue un archivo .pt válido.",
        "current_model": "🔍 Modelo actual:",
        "fallback_model": "Modelo estándar de respaldo (cuchillos, tijeras, bates)",
        "custom_model": "Modelo personalizado cargado",
        "download_report": "📄 Descargar informe de detección",
        "report_filename": "informe_armas",
        "how_it_works": "Cómo funciona:\n- Haga clic en 'Iniciar' abajo.\n- Conceda permiso de cámara.\n- La IA resaltará cualquier arma cerca de una persona como 'OCULTA'.\n- Aparece una advertencia roja en la pantalla.\n- Funciona en teléfonos y computadoras.",
        "camera_feed": "📷 Transmisión de cámara en vivo",
        "camera_active": "✅ Cámara activa. La IA está vigilando armas.",
        "camera_inactive": "⚠️ Cámara no iniciada. Haga clic en 'Iniciar' arriba.",
        "model_warning": "⚠️ Usando modelo estándar (detecta cuchillos, tijeras, bates). Cargue un modelo personalizado para detección real.",
        "dismiss": "Descartar",
        "weapon_detected": "⚠️ ARMA DETECTADA ⚠️",
        "concealed_on_person": "OCULTA EN UNA PERSONA",
        "simulated_weapon": "ARMA SIMULADA",
        "report_title": "Informe de detección de armas",
        "report_generated": "Generado el",
        "no_weapons": "No se detectaron armas durante esta sesión.",
        "report_table_headers": ["Marca de tiempo", "Objeto(s) detectado(s)", "Oculta"],
        "yes": "Sí",
        "no": "No"
    }
}

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
            st.markdown(f"<h1 style='text-align: center;'>{lang_dict[st.session_state.get('lang', 'en')]['login_title']}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'><em>{lang_dict[st.session_state.get('lang', 'en')]['login_sub']}</em></p>", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='text-align: right;'>
                <b>GlobalInternet.py</b><br>
                Gesner Deslandes<br>
                Python Developer
            </div>
            """, unsafe_allow_html=True)
        st.divider()
        pwd = st.text_input(lang_dict[st.session_state.get('lang', 'en')]['password_placeholder'], type="password")
        if pwd == "20082010":
            st.session_state.authenticated = True
            st.rerun()
        elif pwd:
            st.error(lang_dict[st.session_state.get('lang', 'en')]['wrong_password'])
        return False
    return True

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ----------------------------------------------------------------------
# Model management
# ----------------------------------------------------------------------
def load_model_from_file(model_path):
    try:
        model = YOLO(model_path)
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

@st.cache_resource
def get_fallback_model():
    return YOLO("yolov8n.pt")

# ----------------------------------------------------------------------
# Video transformer for real-time detection
# ----------------------------------------------------------------------
class WeaponDetector(VideoTransformerBase):
    def __init__(self, model, is_custom):
        self.model = model
        self.is_custom = is_custom
        self.demo_mode = st.session_state.get("demo_mode", False)
        self.detection_events = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        weapon_detected = False
        concealed_alert = False
        detected_objects = []
        persons = []
        weapons = []

        if self.demo_mode:
            if random.random() < 0.1:
                weapon_detected = True
                concealed_alert = True
                h, w = img.shape[:2]
                x1, y1 = w//2 - 50, h//2 - 50
                x2, y2 = w//2 + 50, h//2 + 50
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, lang_dict[st.session_state.get('lang', 'en')]['simulated_weapon'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                detected_objects.append("Simulated weapon")
        else:
            results = self.model(img)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        if self.is_custom:
                            # Custom model: assume class 0 is weapon (adjust if needed)
                            if conf > 0.5:
                                weapons.append((x1, y1, x2, y2, "Weapon", conf))
                                weapon_detected = True
                                detected_objects.append(f"Weapon (conf {conf:.2f})")
                            # Also detect persons if available (optional)
                        else:
                            # Fallback: COCO model
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
                            class_name = coco_names[cls] if cls < len(coco_names) else "unknown"
                            if class_name == "person":
                                persons.append((x1, y1, x2, y2))
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            elif any(keyword in class_name.lower() for keyword in ["knife", "scissors", "baseball bat", "hammer", "axe", "gun", "pistol", "rifle", "weapon"]):
                                weapons.append((x1, y1, x2, y2, class_name, conf))
                                weapon_detected = True
                                detected_objects.append(f"{class_name} (conf {conf:.2f})")

            # Check for concealed weapons (overlap with persons)
            for wx1, wy1, wx2, wy2, wname, wconf in weapons:
                for px1, py1, px2, py2 in persons:
                    if (wx1 < px2 and wx2 > px1 and wy1 < py2 and wy2 > py1):
                        concealed_alert = True
                        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 3)
                        cv2.putText(img, lang_dict[st.session_state.get('lang', 'en')]['concealed_on_person'], (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        break

            for wx1, wy1, wx2, wy2, wname, wconf in weapons:
                cv2.rectangle(img, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
                cv2.putText(img, f"{wname} {wconf:.2f}", (wx1, wy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        if weapon_detected or concealed_alert:
            cv2.putText(img, lang_dict[st.session_state.get('lang', 'en')]['weapon_detected'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if concealed_alert:
                cv2.putText(img, lang_dict[st.session_state.get('lang', 'en')]['concealed_on_person'], (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.detection_events.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objects": ", ".join(detected_objects),
                "concealed": concealed_alert
            })
            if len(self.detection_events) > 50:
                self.detection_events = self.detection_events[-50:]

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------------------------------------------------
# Generate PDF report in selected language
# ----------------------------------------------------------------------
def generate_report(events, lang):
    t = lang_dict[lang]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(t['report_title'], styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"{t['report_generated']}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    if not events:
        story.append(Paragraph(t['no_weapons'], styles['Normal']))
    else:
        data = [t['report_table_headers']]
        for e in events:
            data.append([e["timestamp"], e["objects"], t['yes'] if e.get("concealed") else t['no']])
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

# Language selection
if "lang" not in st.session_state:
    st.session_state.lang = "en"
lang_options = {"en": "🇺🇸 English", "fr": "🇫🇷 Français", "es": "🇪🇸 Español"}
selected_lang = st.sidebar.selectbox("🌐 Language", options=list(lang_options.keys()), format_func=lambda x: lang_options[x], index=list(lang_options.keys()).index(st.session_state.lang))
if selected_lang != st.session_state.lang:
    st.session_state.lang = selected_lang
    st.rerun()
lang = st.session_state.lang
t = lang_dict[lang]

# Initialize session state
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if "detection_events" not in st.session_state:
    st.session_state.detection_events = []
if "model_warning_dismissed" not in st.session_state:
    st.session_state.model_warning_dismissed = False
if "custom_model" not in st.session_state:
    st.session_state.custom_model = None
if "custom_model_path" not in st.session_state:
    st.session_state.custom_model_path = None

# Display after login
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("https://flagcdn.com/w320/ht.png", width=100)
with col2:
    st.markdown(f"<h1 style='text-align: center;'>{t['app_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'><em>{t['app_subtitle']}</em></p>", unsafe_allow_html=True)
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
st.sidebar.title(t['app_title'])
st.sidebar.markdown("**GlobalInternet.py**")
st.sidebar.markdown("Owner: Gesner Deslandes")
st.sidebar.markdown("📧 deslndes78@gmail.com | 📞 (509) 4738-5663")
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {t['price']}")
st.sidebar.markdown(f"**{t['price_value']}**")
st.sidebar.markdown("---")

if st.sidebar.button(t['logout']):
    logout()

demo_mode = st.sidebar.checkbox(t['demo_mode'], value=st.session_state.demo_mode)
if demo_mode != st.session_state.demo_mode:
    st.session_state.demo_mode = demo_mode
    st.rerun()

# Model upload section
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {t['upload_model']}")
uploaded_model = st.sidebar.file_uploader("", type=["pt"], key="model_uploader")
if uploaded_model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(uploaded_model.read())
        model_path = tmp.name
    model, success = load_model_from_file(model_path)
    if success:
        st.session_state.custom_model = model
        st.session_state.custom_model_path = model_path
        st.sidebar.success(t['upload_success'])
    else:
        st.sidebar.error(t['upload_error'])

# Model selection
if st.session_state.custom_model is not None:
    model_option = st.sidebar.radio(t['current_model'], [t['custom_model'], t['fallback_model']])
    if model_option == t['custom_model']:
        current_model = st.session_state.custom_model
        is_custom = True
    else:
        current_model = get_fallback_model()
        is_custom = False
else:
    current_model = get_fallback_model()
    is_custom = False
    st.sidebar.info(t['model_warning'])

# Training instructions
with st.sidebar.expander(t['training_title']):
    st.markdown(t['training_text'])

if st.sidebar.button(t['download_report']):
    report_buffer = generate_report(st.session_state.get("detection_events", []), lang)
    st.sidebar.download_button("⬇️ PDF", data=report_buffer, file_name=f"{t['report_filename']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

st.sidebar.markdown("---")
st.sidebar.info(t['how_it_works'])

# Video feed
st.markdown(f"### {t['camera_feed']}")
webrtc_ctx = webrtc_streamer(
    key="weapon-detection",
    video_transformer_factory=lambda: WeaponDetector(current_model, is_custom),
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.state.playing:
    st.success(t['camera_active'])
else:
    col_warn, col_btn = st.columns([5,1])
    with col_warn:
        st.warning(t['camera_inactive'])
    with col_btn:
        if st.button(t['dismiss'], key="dismiss_cam_warning"):
            st.rerun()

# Show model warning if using fallback and not dismissed
if not is_custom and not st.session_state.model_warning_dismissed:
    col_warn, col_btn = st.columns([5,1])
    with col_warn:
        st.warning(t['model_warning'])
    with col_btn:
        if st.button(t['dismiss'], key="dismiss_model_warning"):
            st.session_state.model_warning_dismissed = True
            st.rerun()

st.markdown("---")
st.markdown("© 2026 GlobalInternet.py – All rights reserved")
