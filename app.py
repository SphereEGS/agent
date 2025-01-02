import os
import queue
import threading
from datetime import datetime
from time import sleep
from typing import List, Set
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from collections import deque

load_dotenv()
ZONE = os.getenv("ZONE")
CAMERA_URL = os.getenv("CAMERA_URL")
API_BASE_URL = "https://dev-backend.spherex.eglobalsphere.com/api"
SYNC_INTERVAL = 300
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
    "Heeh": "هه",
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


@dataclass
class GateEntry:
    timestamp: datetime
    license_plate: str
    authorized: bool
    image: np.ndarray


class PlateCache:
    def __init__(self):
        self._allowed_plates: Set[str] = set()
        self._lock = threading.Lock()

    def update(self, plates: List[str]):
        with self._lock:
            self._allowed_plates = set(plates)

    def is_authorized(self, plate: str) -> bool:
        with self._lock:
            return (
                plate in self._allowed_plates
                or plate[::-1] in self._allowed_plates
            )


class LogQueue:
    def __init__(self):
        self._logs = deque()
        self._lock = threading.Lock()

    def add(self, entry: GateEntry):
        with self._lock:
            self._logs.append(entry)

    def get_all(self) -> List[GateEntry]:
        with self._lock:
            logs = list(self._logs)
            self._logs.clear()
            return logs


class SyncManager(threading.Thread):
    def __init__(self, plate_cache: PlateCache, log_queue: LogQueue):
        super().__init__(daemon=True)
        self.plate_cache = plate_cache
        self.log_queue = log_queue
        self.running = True

    def stop(self):
        self.running = False

    def _fetch_allowed_plates(self) -> List[str]:
        try:
            response = requests.get(
                f"{API_BASE_URL}/method/spherex.api.license_plate.get_allowed_plates",
                params={"zone": ZONE},
            )
            if response.status_code == 200:
                return response.json().get("plates", [])
        except Exception as e:
            print(f"❌ Error fetching allowed plates: {e}")
        return []

    def _upload_image(self, image: np.ndarray, orientation: str) -> str:
        temp_file = f"gate_entry_{orientation}.jpg"
        cv2.imwrite(temp_file, image)
        try:
            files = {
                "file": (
                    f"entry_{orientation}.jpg",
                    open(temp_file, "rb"),
                    "image/jpeg",
                )
            }
            response = requests.post(
                f"{API_BASE_URL}/method/spherex.api.upload_file",
                files=files,
            )
            os.remove(temp_file)
            if response.status_code == 200:
                return response.json()["message"]["file_url"]
        except Exception as e:
            print(f"❌ Error uploading image: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return ""

    def _sync_logs(self):
        logs = self.log_queue.get_all()
        for entry in logs:
            try:
                normal_url = self._upload_image(entry.image_normal, "normal")
                reversed_url = self._upload_image(
                    entry.image_reversed, "reversed"
                )

                if not normal_url or not reversed_url:
                    continue

                for orientation, url in [
                    ("normal", normal_url),
                    ("reversed", reversed_url),
                ]:
                    data = {
                        "zone": ZONE,
                        "license_plate": entry.license_plate,
                        "authorized": entry.authorized,
                        "image": url,
                        "timestamp": entry.timestamp.isoformat(),
                    }
                    requests.post(
                        f"{API_BASE_URL}/resource/Gate Entry Log", data=data
                    )
            except Exception as e:
                print(f"❌ Error syncing log entry: {e}")

    def run(self):
        while self.running:
            # Update allowed plates
            plates = self._fetch_allowed_plates()
            if plates:
                self.plate_cache.update(plates)
                print(f"✅ Updated allowed plates cache: {len(plates)} plates")

            # Sync logs
            self._sync_logs()

            # Wait for next sync interval
            for _ in range(SYNC_INTERVAL):
                if not self.running:
                    break
                sleep(1)


class CameraStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.queue = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                return

            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True

    def release(self):
        self.stopped = True
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()


class SpherexAgent:
    def __init__(self):
        self.camera = None
        self.failed_attempts = 0
        self.plate_cache = PlateCache()
        self.log_queue = LogQueue()
        self.sync_manager = SyncManager(self.plate_cache, self.log_queue)

        # Initialize model
        os.makedirs("models", exist_ok=True)
        self.model_path = "models/license_yolo8s_1024.pt"
        self.font_path = "./fonts/DejaVuSans.ttf"
        self._initialize_model()

    def _initialize_model(self):
        try:
            if not os.path.exists(self.model_path):
                print("Downloading model for the first time...")
                model_dir = snapshot_download("omarelsayeed/licence_plates")
                source_model = os.path.join(
                    model_dir, "license_yolo8s_1024.pt"
                )
                import shutil

                shutil.copy2(source_model, self.model_path)
                print("Model downloaded successfully")

            self.model = YOLO(self.model_path)
            print("\033[H\033[J")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            exit(1)

    def add_text_to_image(
        self, image: np.ndarray, text: str, reverse_text: bool = False
    ) -> np.ndarray:
        if not text:
            return image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        height, width = image.shape[:2]
        font_size = int(height / 15)

        try:
            font = (
                ImageFont.truetype(self.font_path, font_size)
                if self.font_path
                else ImageFont.load_default()
            )
            draw = ImageDraw.Draw(pil_image)

            separated_text = "-".join(
                reversed(list(text)) if not reverse_text else list(text)
            )
            padding = 20
            text_bbox = draw.textbbox((0, 0), separated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = padding
            y = height - text_height - padding * 2

            # Draw background
            background_coords = [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding,
            ]
            draw.rectangle(background_coords, fill=(0, 0, 0, 180))

            # Draw text
            draw.text((x, y), separated_text, font=font, fill=(255, 255, 255))

            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️ Warning: Could not add text to image: {str(e)}")
            return image

    def process_frame(self, frame: np.ndarray):
        try:
            # Detect and process license plate
            cropped_plate = self.detect_and_crop_plate(frame)
            if cropped_plate is None:
                self.display_status(
                    "👁️ No license plate detected", datetime.now()
                )
                return

            # Recognize characters
            results = self.model.predict(
                cropped_plate, conf=0.25, iou=0.45, imgsz=1024, verbose=False
            )
            if not results:
                self.display_status(
                    "👁️ No license plate detected", datetime.now()
                )
                return

            # Process detection results
            boxes_and_classes = [
                (
                    float(box[0]),
                    float(box[2]),
                    self.model.names[int(cls)],
                    conf,
                )
                for box, cls, conf in zip(
                    results[0].boxes.xyxy,
                    results[0].boxes.cls,
                    results[0].boxes.conf,
                )
            ]

            # Sort and extract text
            boxes_and_classes.sort(key=lambda b: b[0])
            license_text = "".join(
                ARABIC_MAPPING.get(cls, cls)
                for _, _, cls, _ in boxes_and_classes
                if cls in ARABIC_MAPPING
            )

            if not license_text:
                self.display_status(
                    "👁️ No license plate detected", datetime.now()
                )
                return

            # Check authorization locally
            is_authorized = self.plate_cache.is_authorized(license_text)

            # Create log entry
            timestamp = datetime.now()
            image_normal = self.add_text_to_image(frame, license_text, False)
            image_reversed = self.add_text_to_image(frame, license_text, True)

            entry = GateEntry(
                timestamp=timestamp,
                license_plate=license_text,
                authorized=is_authorized,
                image_normal=image_normal,
                image_reversed=image_reversed,
            )

            self.log_queue.add(entry)

            # Update display
            auth_status = (
                "✅ Authorized" if is_authorized else "❌ Not Authorized"
            )
            status_message = (
                f"✨ License Plate Detected!\n"
                f"📝 Plate Text: {license_text}\n"
                f"🔑 Authorization: {auth_status}"
            )
            self.display_status(status_message, timestamp)

        except Exception as e:
            print(f"❌ Error processing frame: {str(e)}")

    def detect_and_crop_plate(self, image: np.ndarray) -> np.ndarray:
        results = self.model.predict(
            image, conf=0.25, iou=0.45, verbose=False, imgsz=1024
        )
        if not results or len(results[0].boxes) == 0:
            return None

        plate_boxes = []
        plate_scores = []

        for box, cls, conf in zip(
            results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
        ):
            if self.model.names[int(cls)] == "License Plate":
                plate_boxes.append(box.cpu().numpy())
                plate_scores.append(float(conf))

        if not plate_boxes:
            return None

        plate_boxes = np.array(plate_boxes)
        plate_scores = np.array(plate_scores)
        best_idx = np.argmax(plate_scores)

        if plate_scores[best_idx] < 0.6:
            return None

        # Crop with padding
        h, w = image.shape[:2]
        x1, y1, x2, y2 = plate_boxes[best_idx]
        pad_x = (x2 - x1) * 0.1
        pad_y = (y2 - y1) * 0.1

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        return image[y1:y2, x1:x2]

    def display_status(self, message: str, timestamp: datetime):
        print("\033[H", end="")
        print("\033[J", end="")
        print(f"🕒 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{message}\n")

    def start(self):
        try:
            # Start sync manager
            self.sync_manager.start()

            # Connect to camera
            print(f"🔗 Connecting to Camera Stream: {CAMERA_URL}")
            self.camera = CameraStream(CAMERA_URL)
            if not self.camera.start().isOpened():
                raise Exception("Failed to connect to Camera Stream")

            print("✅ Connection successful")
            print("🎥 Starting license plate detection...\n")
            print("Press Ctrl+C to stop the program\n")

            while True:
                if not self.camera.isOpened():
                    self.camera = CameraStream(CAMERA_URL)
                    self.camera.start()
                    continue

                frame = self.camera.read()
                if frame is None:
                    self.camera.release()
                    continue

                self.process_frame(frame)

        except KeyboardInterrupt:
            print("\n👋 Stopping license plate detection...")
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
        finally:
            if self.camera:
                self.camera.release()
            self.sync_manager.stop()
            print("✅ Released all resources")


if __name__ == "__main__":
    agent = SpherexAgent()
    agent.start()
