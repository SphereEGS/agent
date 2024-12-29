import cv2
import time
import os
import numpy as np
from datetime import datetime
import requests
from ultralytics import YOLO
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ZONE = os.getenv("ZONE")
BACKEND_URL = os.getenv("BACKEND_URL")
CAMERA_URL = os.getenv("CAMERA_URL")

# Arabic mapping for license plates
ARABIC_MAPPING = {
    "baa": "\u0628",
    "Laam": "\u0644",
    "Taa": "\u0637",
    "7aah": "\u062d",
    "Daad": "\u0636",
    "Een": "\u0639",
    "F": "\u0641",
    "Kaaf": "\u0643",
    "Meem": "\u0645",
    "Noon": "\u0646",
    "Q": "\u0642",
    "R": "\u0631",
    "Saad": "\u0635",
    "Seen": "\u0633",
    "Wow": "\u0648",
    "Yeeh": "\u064a",
    "Zeen": "\u0632",
    "alef": "\u0623",
    "daal": "\u062f",
    "geem": "\u062c",
    "Heeh": "\u0647",
    "1": "\u0661",
    "2": "\u0662",
    "3": "\u0663",
    "4": "\u0664",
    "5": "\u0665",
    "6": "\u0666",
    "7": "\u0667",
    "8": "\u0668",
    "9": "\u0669",
}


class SpherexAgent:
    def __init__(self):
        self.cap = None
        os.makedirs("frames", exist_ok=True)

        # Create a models directory to cache the downloaded model
        os.makedirs("models", exist_ok=True)
        model_path = "models/license_yolo_N_96.5_1024.pt"

        try:
            if not os.path.exists(model_path):
                print("Downloading model for the first time...")
                model_dir = snapshot_download("omarelsayeed/licence_plates")
                source_model = os.path.join(
                    model_dir, "license_yolo_N_96.5_1024.pt"
                )
                import shutil

                shutil.copy2(source_model, model_path)
                print("Model downloaded successfully")

            self.model = YOLO(model_path)
            print("\033[H\033[J")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            exit(1)

    def connect_to_stream(self):
        """Establish connection to RTSP stream"""
        try:
            print(f"🔗 Connecting to Camera Stream: {CAMERA_URL}")
            self.cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.cap.isOpened():
                print("✅ Connection successful")
                return True

        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            if self.cap:
                self.cap.release()

        return False

    def save_frame(self, frame, plate_text=None):
        """Save frame with timestamp and optional plate text"""
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        frame_filename = f"frames/{timestamp}.jpg"
        cv2.imwrite(frame_filename, frame)

    def process_frame(self, frame):
        """Process a single frame to detect and recognize license plate"""
        try:
            cropped_plate = self.detect_and_crop_plate(frame)
            if cropped_plate is None:
                return None, None, False

            results = self.model.predict(
                cropped_plate, conf=0.25, iou=0.45, imgsz=1024, verbose=False
            )
            if not results:
                return None, None, False

            detections = [
                (float(box[0]), self.model.names[int(cls)])
                for box, cls in zip(
                    results[0].boxes.xyxy, results[0].boxes.cls
                )
            ]
            detections.sort()

            unmapped_chars = [cls for _, cls in detections]
            unmapped_text = "-".join(unmapped_chars)

            processed_chars = [
                ARABIC_MAPPING[cls]
                for cls in unmapped_chars
                if cls in ARABIC_MAPPING
            ]
            license_text = "-".join(processed_chars)

            is_authorized = self.check_authorization(
                license_text.replace("-", "")
            )

            if license_text:
                self.save_frame(frame, license_text)

            return license_text, unmapped_text, is_authorized

        except Exception as e:
            print(f"❌ Error processing frame: {str(e)}")
            return None, None, False

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
        """Check if plate is authorized"""
        try:
            response = requests.get(BACKEND_URL, params={"zone": ZONE})
            if response.status_code == 200:
                data = response.json()
                return plate in data.get("message", [])
            return False
        except Exception as e:
            print(f"❌ Authorization check error: {e}")
            return False

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
                if not self.cap.isOpened():
                    self.connect_to_stream()
                    continue

                self.cap.grab()
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    self.connect_to_stream()
                    continue

                license_text, unmapped_text, is_authorized = (
                    self.process_frame(frame)
                )

                if license_text:
                    status_message = (
                        "✨ License Plate Detected!\n"
                        f"📝 Plate Text (Arabic): {license_text}\n"
                        f"📝 Plate Text (Raw): {unmapped_text}\n"
                        f"🔑 Authorization: {'✅ Authorized' if is_authorized else '❌ Not Authorized'}"
                    )
                    self.display_status(
                        status_message,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    time.sleep(10)
                else:
                    self.display_status(
                        "👁️ No license plate detected \n" "❌ Not Authorized",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n👋 Stopping license plate detection...")
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
                print("✅ Released video capture resources")


if __name__ == "__main__":
    app = SpherexAgent()
    app.start_processing()
