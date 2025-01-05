import os

from dotenv import load_dotenv

load_dotenv()

ZONE = os.getenv("ZONE", "default_zone")
CAMERA_URL = os.getenv("CAMERA_URL", "http://default/camera")
API_BASE_URL = os.getenv(
    "API_BASE_URL", "https://dev-backend.spherex.eglobalsphere.com/api"
)
FONT_PATH = os.getenv("FONT_PATH", "fonts/DejaVuSans.ttf")
MODEL_PATH = os.getenv("MODEL_PATH", "models/license_yolo8s_1024.pt")
UPDATE_INTERVAL = 10
CONTROLLER_IP = os.getenv("CONTROLLER_IP", "127.0.0.1")
CONTROLLER_PORT = os.getenv("CONTROLLER_PORT", 4000)
CONTROLLER_CA_PATH = os.getenv("CONTROLLER_CA_PATH", "controller/cert/ca.crt")
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
