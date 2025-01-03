import os
from datetime import datetime
import requests
import cv2
from .camera import CameraStream
from .config import CAMERA_URL, ZONE, API_BASE_URL
from .model import PlateDetector
from .sync import SyncManager
# from .gate import GateControl


class SpherexAgent:
    def __init__(self):
        self.camera = None
        self.failed_attempts = 0
        # self.gate = GateControl()
        self.plate_detector = PlateDetector()
        self.plate_cache = SyncManager()
        self.plate_cache.start()
        self.is_logged = False

    def connect_to_stream(self):
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

    def log_gate_entry(self, plate, frame, is_authorized):
        try:
            frame_with_text = self.plate_detector.add_text_to_image(
                frame, plate
            )
            temp_file = "gate_entry.jpg"
            cv2.imwrite(temp_file, frame_with_text)

            log_data = {
                "zone": ZONE,
                "license_plate": plate,
                "authorized": is_authorized,
                "image": temp_file,
            }

            with open(temp_file, "rb") as image_file:
                files = {"file": image_file}
                upload_response = requests.post(
                    f"{API_BASE_URL}/method/spherex.api.upload_file",
                    files=files,
                )
                log_data["image"] = upload_response.json().get("file_url")

            requests.post(
                f"{API_BASE_URL}/resource/Gate Entry Log",
                data=log_data,
            )
        except Exception as e:
            print(f"❌ Error logging entry: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def display_status(self, message, timestamp):
        print("\033[H", end="")
        print("\033[J", end="")
        print(f"🕒 {timestamp}")
        print(f"\n{message}\n")

    def process_frame(self, frame):
        try:
            cropped_plate = self.plate_detector.detect_and_crop_plate(frame)
            if cropped_plate is None:
                # self.gate.lock()
                self.is_logged = False
                self.failed_attempts = 0
                self.display_status(
                    "👁️ No license plate detected \n❌ Not Authorized",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                return

            license_text = self.plate_detector.recognize_plate(cropped_plate)
            if not license_text:
                # self.gate.lock()
                self.is_logged = False
                self.failed_attempts = 0
                self.display_status(
                    "👁️ No license plate detected \n❌ Not Authorized",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                return

            is_authorized = self.plate_cache.is_authorized(license_text)
            print(self.plate_cache.allowed_plates)

            if is_authorized:
                # self.gate.unlock()
                self.log_gate_entry(license_text, frame, True)
                self.failed_attempts = 0
                self.is_logged = False
            else:
                self.failed_attempts += 1
                # self.gate.lock()
                if self.failed_attempts >= 3 and not self.is_logged:
                    self.log_gate_entry(license_text, frame, False)
                    self.is_logged = True

            auth_status = (
                "✅ Authorized"
                if is_authorized
                else "❌ Not Authorized (Failed Attempts)"
                + (" [Logged]" if self.is_logged else "")
            )
            status_message = (
                "✨ License Plate Detected!\n"
                f"📝 Plate Text: {license_text[::-1]}\n"
                f"🔑 Authorization: {auth_status}\n"
            )
            self.display_status(
                status_message,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

        except Exception as e:
            print(f"❌ Error processing frame: {str(e)}")

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