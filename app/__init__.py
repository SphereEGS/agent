import os
from datetime import datetime
from time import sleep

import cv2
import requests

from .camera import CameraStream
from .config import API_BASE_URL, CAMERA_URL, ZONE, logger
from .gate import GateControl
from .model import PlateDetector
from .sync import SyncManager


class SpherexAgent:
    def __init__(self):
        self.camera = None
        self.failed_attempts = 0
        self.gate = GateControl()
        self.model = PlateDetector()
        self.cache = SyncManager()
        self.is_logged = False

    def connect_to_stream(self):
        try:
            logger.info(f"🔗 Connecting to Camera Stream: {CAMERA_URL}")
            self.camera = CameraStream(CAMERA_URL)
            self.camera.start()

            if self.camera.isOpened():
                logger.info("✅ Connection successful")
                return True

        except Exception as e:
            logger.error(f"❌ Connection failed: {str(e)}")
            if self.camera:
                self.camera.release()

        return False

    def log_gate_entry(self, plate, frame, is_authorized):
        try:
            frame_with_text = self.model.add_text_to_image(frame, plate)
            frame_with_roi = self.model.visualize_roi(frame, frame_with_text)
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
                    f"{API_BASE_URL}/api/method/spherex.api.upload_file",
                    files=files,
                )
                log_data["image"] = upload_response.json()["message"][
                    "file_url"
                ]

            requests.post(
                f"{API_BASE_URL}/api/resource/Gate Entry Log",
                data=log_data,
            )
        except Exception as e:
            logger.error(f"❌ Error logging entry: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def display_status(self, message, timestamp):
        logger.info(f"🕒 {timestamp}")
        logger.info(f"\n{message}\n")

    def process_frame(self, frame):
        try:
            cropped_plate = self.model.detect_and_crop_plate(frame)
            if cropped_plate is None:
                self.is_logged = False
                self.failed_attempts = 0
                self.display_status(
                    "👁️ No license plate detected \n❌ Not Authorized",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                return

            license_text = self.model.recognize_plate(cropped_plate)
            if not license_text:
                self.is_logged = False
                self.failed_attempts = 0
                self.display_status(
                    "👁️ No license plate detected \n❌ Not Authorized",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                return

            is_authorized = self.cache.is_authorized(license_text)
            auth_status = (
                "✅ Authorized"
                if is_authorized
                else "❌ Not Authorized"
                + (" [Logged]" if self.is_logged else "")
            )
            status_message = (
                "✨ License Plate Detected!\n"
                f"📝 Plate Text: {license_text}\n"
                f"🔑 Authorization: {auth_status}\n"
            )
            if is_authorized:
                self.gate.open()
                self.failed_attempts = 0
                self.is_logged = False
                self.display_status(
                    status_message,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                self.log_gate_entry(license_text, frame, 1)
                sleep(1)
            else:
                self.failed_attempts += 1
                if self.failed_attempts >= 2 and not self.is_logged:
                    self.log_gate_entry(license_text, frame, 0)
                    self.is_logged = True

                self.display_status(
                    status_message,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )

        except Exception as e:
            logger.error(f"❌ Error processing frame: {str(e)}")

    def start_processing(self):
        """Main processing loop"""
        try:
            if not self.connect_to_stream():
                raise Exception("Failed to connect to Camera Stream")

            logger.info("🎥 Starting license plate detection...\n")
            logger.info("Press Ctrl+C to stop the program\n")

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
            logger.info("\n👋 Stopping license plate detection...")
        except Exception as e:
            logger.error(f"\n❌ An error occurred: {str(e)}")
        finally:
            if self.camera:
                self.camera.release()
                logger.info("✅ Released video capture resources")
