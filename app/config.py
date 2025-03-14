import logging
import os

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("app.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
ZONE = os.getenv("ZONE", "default_zone")

# Multi-camera support
CAMERAS = {
    "entry": {
        "url": os.getenv("ENTRY_CAMERA_URL", os.getenv("CAMERA_URL", "http://default/camera")),
        "direction": "entry",
        "roi_config": os.getenv("ENTRY_ROI_CONFIG", "config_entry.json"),
    },
    "exit": {
        "url": os.getenv("EXIT_CAMERA_URL", ""),
        "direction": "exit",
        "roi_config": os.getenv("EXIT_ROI_CONFIG", "config_exit.json"),
    }
}

# For backward compatibility
CAMERA_URL = os.getenv("CAMERA_URL", "http://default/camera")

API_BASE_URL = os.getenv(
    "API_BASE_URL", "https://dev-backend.spherex.eglobalsphere.com"
)
FONT_PATH = os.getenv("FONT_PATH", "fonts/DejaVuSans.ttf")
MODEL_PATH = os.getenv("MODEL_PATH", "models/license_yolo8s_1024.pt")
UPDATE_INTERVAL = 600

# Multi-gate support
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

# For backward compatibility
CONTROLLER_IP = os.getenv("CONTROLLER_IP", "127.0.0.1")
CONTROLLER_USER = os.getenv("CONTROLLER_USER", "test")
CONTROLLER_PASSWORD = os.getenv("CONTROLLER_PASSWORD", "E512512A")
DOOR_ID = os.getenv("DOOR_ID", 1)

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
