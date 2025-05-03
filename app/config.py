import logging
import os
import sys
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

# Load environment variables from .env file
load_dotenv()

# Fix for Unicode encoding in Windows command prompt
if sys.platform == "win32":
    # Force UTF-8 encoding for console output
    os.system("chcp 65001 > NUL")
    try:
        # Try to set UTF-8 encoding without changing locale
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer)
    except Exception as e:
        print(f"Warning: Could not set UTF-8 encoding: {e}")

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

# Create a custom logger
logger = logging.getLogger("SpherexLogger")
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = RotatingFileHandler(
    log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

# Create formatters and add to handlers
log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Gate identification
GATE = os.getenv("GATE", "A1")
logger.info(f"Operating in zone: {GATE}")

# Camera input sources
CAMERA_SOURCES = {
    "cctv": "rtsp://test:123456789A@10.0.40.11:554/Streaming/channels/801",  # Default CCTV stream
    "webcam": "0",  # Default webcam
    "video_file": "input/test_video3.mov",  # Test video file
}

# Camera configuration - default to CCTV if no specific camera URL is set
CAMERA_URL = os.getenv(
    "CAMERA_URL", CAMERA_SOURCES["cctv"]
)  # Camera source from .env
logger.info(f"Using camera source: {CAMERA_URL}")

# Load multiple camera URLs from environment variables
CAMERA_URLS = {}
CAMERA_TYPES = {}

# First add the main CAMERA_URL for backward compatibility
# CAMERA_URLS["main"] = CAMERA_URL

# Scan for numbered camera URLs (CAMERA_URL_1, CAMERA_URL_2, etc.)
for i in range(1, 10):  # Support up to 9 cameras
    env_key = f"CAMERA_URL_{i}"
    url = os.getenv(env_key)
    if url:
        CAMERA_URLS[f"camera_{i}"] = url
        logger.info(f"Found additional camera: {env_key} = {url}")
        # Load camera type (Entry/Exit) if specified
        type_key = f"CAMERA_TYPE_{i}"
        cam_type = os.getenv(type_key, "Entry")
        CAMERA_TYPES[f"camera_{i}"] = cam_type
        logger.info(f"Camera {env_key} type: {cam_type}")

# If no numbered cameras found but main camera exists, use just the main camera
if len(CAMERA_URLS) == 1:
    logger.info("Using single camera mode with main camera")
else:
    logger.info(f"Using multi-camera mode with {len(CAMERA_URLS)} cameras")

# Fallback configuration
ALLOW_FALLBACK = (
    os.getenv("ALLOW_FALLBACK", "false").lower() == "true"
)  # Parse string to boolean
logger.info(f"Camera fallback enabled: {ALLOW_FALLBACK}")

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
PROCESS_EVERY = int(os.getenv("PROCESS_EVERY", "4"))  # Process every Nth frame

# API configuration
API_BASE_URL = os.getenv(
    "API_BASE_URL", "https://dev-backend.spherex.eglobalsphere.com/api"
)
SOCKETIO_SERVER = os.getenv("SOCKETIO_SERVER", "http://localhost:9001")
SOCKETIO_NAMESPACE = os.getenv("SOCKETIO_NAMESPACE", "/spherex")

# File paths
FONT_PATH = os.getenv("FONT_PATH", "fonts/NotoSansArabic-Regular.ttf")
LPR_MODEL_PATH = os.getenv(
    "LPR_MODEL_PATH", "models/license_plate_recognition.pt"
)
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo11n.pt")

# Update interval
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "600"))

# Backward compatibility
CONTROLLER_IP = os.getenv("CONTROLLER_IP", "127.0.0.1")
CONTROLLER_USER = os.getenv("CONTROLLER_USER", "test")
CONTROLLER_PASSWORD = os.getenv("CONTROLLER_PASSWORD", "E512512A")
DOOR_ID = os.getenv("DOOR_ID", 1)

# Detection parameters
DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.3"))
DETECTION_IOU = float(os.getenv("DETECTION_IOU", "0.45"))
TARGET_WIDTH = int(os.getenv("TARGET_WIDTH", "1280"))

# Arabic mapping
ARABIC_MAPPING = {
    "0": "٠",
    "1": "١",
    "2": "٢",
    "3": "٣",
    "4": "٤",
    "5": "٥",
    "6": "٦",
    "7": "٧",
    "8": "٨",
    "9": "٩",
    "Beh": "ب",
    "Daad": "ض",
    "Een": "ع",
    "F": "ف",
    "Heeh": "H",
    "Kaaf": "ك",
    "Laam": "ل",
    "License Plate": "",
    "Meem": "م",
    "Noon": "ن",
    "Q": "ق",
    "R": "ر",
    "Saad": "ص",
    "Seen": "س",
    "Taa": "ط",
    "Wow": "و",
    "Yeeh": "ي",
    "Zah": "ظ",
    "Zeen": "ز",
    "alef": "أ",
    "car": "",
    "daal": "د",
    "geem": "ج",
    "ghayn": "غ",
    "khaa": "خ",
    "sheen": "ش",
    "teh": "ت",
    "theh": "ث",
    "zaal": "ذ",
    "7aah": "ح",
}

logger.info("Configuration loaded successfully")
