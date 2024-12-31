import os
import queue
from datetime import datetime
from threading import Thread
from time import sleep

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from ultralytics import YOLO

load_dotenv()
ZONE = os.getenv("ZONE")
CAMERA_URL = os.getenv("CAMERA_URL")
API_BASE_URL = "https://dev-backend.spherex.eglobalsphere.com/api"

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


class CameraStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.queue = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
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
        os.makedirs("models", exist_ok=True)
        model_path = "models/license_yolo8s_1024.pt"

        try:
            if not os.path.exists(model_path):
                print("Downloading model for the first time...")
                model_dir = snapshot_download("omarelsayeed/licence_plates")
                source_model = os.path.join(
                    model_dir, "license_yolo8s_1024.pt"
                )
                import shutil

                shutil.copy2(source_model, model_path)
                print("Model downloaded successfully")

            self.model = YOLO(model_path)
            print("\033[H\033[J")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            exit(1)

    def display_no_plate_message(self):
        """Display message when no license plate is detected"""
        self.display_status(
            "👁️ No license plate detected \n❌ Not Authorized",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def connect_to_stream(self):
        """Establish connection to RTSP stream"""
        try:
            print(f"🔗 Connecting to Camera Stream: {CAMERA_URL}")
            self.camera = CameraStream(CAMERA_URL)
            self.camera.start()

            if self.camera.isOpened():
                print("✅ Connection successful")
                return True

        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            if self.camera:
                self.camera.release()

        return False

    def process_frame(self, frame):
        """Process a single frame to detect and recognize license plate"""
        try:
            cropped_plate = self.detect_and_crop_plate(frame)
            if cropped_plate is None:
                self.display_no_plate_message()
                return

            results = self.model.predict(
                cropped_plate, conf=0.25, iou=0.45, imgsz=1024, verbose=False
            )
            if not results:
                self.display_no_plate_message()
                return

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

            boxes_and_classes.sort(key=lambda b: b[0])
            unmapped_chars = [
                cls
                for _, _, cls, _ in boxes_and_classes
                if cls in ARABIC_MAPPING
            ]

            license_text = "".join(
                [
                    ARABIC_MAPPING.get(c, c)
                    for c in unmapped_chars
                    if c in ARABIC_MAPPING
                ]
            )

            is_authorized = self.check_authorization(license_text)

            if is_authorized:
                self.log_gate_entry(license_text, frame, True)
                self.failed_attempts = 0
                sleep(5)
            else:
                self.failed_attempts += 1
                if self.failed_attempts >= 5:
                    self.log_gate_entry(license_text, frame, None)
                    self.failed_attempts = 0

            if license_text:
                auth_status = '✅ Authorized' if is_authorized else f'❌ Not Authorized (Failed Attempts: {self.failed_attempts}/5)'
                status_message = (
                    "✨ License Plate Detected!\n"
                    f"📝 Plate Text: {license_text}\n"
                    f"🔑 Authorization: {auth_status}\n"
                )
                self.display_status(
                    status_message,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                self.display_no_plate_message()

        except Exception as e:
            print(f"❌ Error processing frame: {str(e)}")

    def detect_and_crop_plate(self, image):
        """Detect and crop license plate from frame"""
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

        h, w = image.shape[:2]
        x1, y1, x2, y2 = plate_boxes[best_idx]
        pad_x = (x2 - x1) * 0.1
        pad_y = (y2 - y1) * 0.1

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        return image[y1:y2, x1:x2]

    def check_authorization(self, plate):
        """Check if plate is authorized using the new endpoint"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/method/spherex.api.license_plate.authorize",
                params={"zone": ZONE, "plate": plate},
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Authorization check error: {e}")
            return False

    def log_gate_entry(self, plate, frame, is_authorized):
        """Log gate entry attempt"""
        try:
            temp_file = "gate_entry.jpg"
            cv2.imwrite(temp_file, frame)

            files = {
                "file": ("entry.jpg", open(temp_file, "rb"), "image/jpeg")
            }
            upload_response = requests.post(
                f"{API_BASE_URL}/method/spherex.api.upload_file", files=files
            )

            if upload_response.status_code != 200:
                print(
                    f"❌ Failed to upload entry image: {upload_response.text}"
                )

            file_url = upload_response.json()["message"]["file_url"]

            data = {
                "zone": ZONE,
                "license_plate": plate,
                "authorized": is_authorized,
                "image": file_url,
            }

            response = requests.post(
                f"{API_BASE_URL}/resource/Gate Entry Log", data=data
            )

            os.remove(temp_file)

            if response.status_code != 200:
                print(f"❌ Failed to log entry attempt: {response.text}")

        except Exception as e:
            print(f"❌ Error logging entry attempt: {e}")

    def display_status(self, message, timestamp):
        """Display status without screen clearing"""
        print("\033[H", end="")
        print("\033[J", end="")
        print(f"🕒 {timestamp}")
        print(f"\n{message}\n")

    def start_processing(self):
        """Main processing loop"""
        try:
            if not self.connect_to_stream():
                raise Exception("Failed to connect to Camera Stream")

            print("🎥 Starting license plate detection...\n")
            print("Press Ctrl+C to stop the program\n")

            while True:
                if not self.camera.isOpened():
                    self.connect_to_stream()
                    continue

                frame = self.camera.read()
                if frame is None:
                    self.camera.release()
                    self.connect_to_stream()
                    continue

                self.process_frame(frame)
        except KeyboardInterrupt:
            print("\n👋 Stopping license plate detection...")
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
        finally:
            if self.camera:
                self.camera.release()
                print("✅ Released video capture resources")


if __name__ == "__main__":
    app = SpherexAgent()
    app.start_processing()
