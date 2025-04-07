import logging
import os
import sys
from dotenv import load_dotenv

# Force reload of environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ],
)
logger = logging.getLogger(__name__)

# Zone identification
ZONE = os.getenv("ZONE", "default_zone")
logger.info(f"Operating in zone: {ZONE}")

# Camera input sources
CAMERA_SOURCES = {
    "cctv": os.getenv("CCTV_URL", "rtsp://192.168.0.3:8554/stream"),
    "webcam": os.getenv("WEBCAM_INDEX", "0"),
    "video_file": os.getenv("VIDEO_FILE", "input/test_video3.mov")
}

# Camera configuration - default to CCTV if no specific camera URL is set
INPUT_SOURCE = os.getenv("CAMERA_URL", CAMERA_SOURCES["cctv"])
ALLOW_FALLBACK = os.getenv("NO_FALLBACK", "false").lower() != "true"
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
PROCESS_EVERY = int(os.getenv("PROCESS_EVERY", "3"))

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://dev-backend.spherex.eglobalsphere.com/api")

# File paths
FONT_PATH = os.getenv("FONT_PATH", "fonts/DejaVuSans.ttf")
LPR_MODEL_PATH = os.getenv("LPR_MODEL_PATH", "models/license_yolo8s_1024.pt")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join("models", "yolo11n.pt"))

# Update interval
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "600"))

# Gate controllers
CONTROLLERS = {
    "entry": {
        "ip": os.getenv("ENTRY_CONTROLLER_IP", os.getenv("CONTROLLER_IP", "127.0.0.1")),
        "user": os.getenv("ENTRY_CONTROLLER_USER", os.getenv("CONTROLLER_USER", "test")),
        "password": os.getenv("ENTRY_CONTROLLER_PASSWORD", os.getenv("CONTROLLER_PASSWORD", "E512512A")),
        "door_id": os.getenv("ENTRY_DOOR_ID", os.getenv("DOOR_ID", 1)),
    },
    "exit": {
        "ip": os.getenv("EXIT_CONTROLLER_IP", ""),
        "user": os.getenv("EXIT_CONTROLLER_USER", "test"),
        "password": os.getenv("EXIT_CONTROLLER_PASSWORD", "E512512A"),
        "door_id": os.getenv("EXIT_DOOR_ID", 1),
    }
}

# Backward compatibility
CONTROLLER_IP = os.getenv("CONTROLLER_IP", "127.0.0.1")
CONTROLLER_USER = os.getenv("CONTROLLER_USER", "test")
CONTROLLER_PASSWORD = os.getenv("CONTROLLER_PASSWORD", "E512512A")
DOOR_ID = os.getenv("DOOR_ID", 1)

# Detection parameters
DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.3"))
DETECTION_IOU = float(os.getenv("DETECTION_IOU", "0.45"))
TARGET_WIDTH = int(os.getenv("TARGET_WIDTH", "1280"))

# Log configuration
logger.info(f"Using camera source: {INPUT_SOURCE}")
logger.info(f"Webcam fallback enabled: {ALLOW_FALLBACK}")
logger.info(f"Configuration loaded:")
logger.info(f"- ZONE: {ZONE}")
logger.info(f"- INPUT_SOURCE: {INPUT_SOURCE}")
logger.info(f"- PROCESS_EVERY: {PROCESS_EVERY} frames")
logger.info(f"- Models: {os.path.basename(LPR_MODEL_PATH)}, {os.path.basename(YOLO_MODEL_PATH)}")
logger.info(f"- Detection settings: conf={DETECTION_CONF}, iou={DETECTION_IOU}")

# Arabic mapping
ARABIC_MAPPING = {
    "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤", "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩",
    "Beh": "ب", "Daad": "ض", "Een": "ع", "F": "ف", "Heeh": "H", "Kaaf": "ك", "Laam": "ل",
    "License Plate": "", "Meem": "م", "Noon": "ن", "Q": "ق", "R": "ر", "Saad": "ص",
    "Seen": "س", "Taa": "ط", "Wow": "و", "Yeeh": "ي", "Zah": "ظ", "Zeen": "ز",
    "alef": "أ", "car": "", "daal": "د", "geem": "ج", "ghayn": "غ", "khaa": "خ",
    "sheen": "ش", "teh": "ت", "theh": "ث", "zaal": "ذ", "7aah": "ح",
}