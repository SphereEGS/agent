import os
from datetime import datetime
from time import sleep
from threading import Thread

import cv2
import requests

from .camera import CameraStream
from .config import API_BASE_URL, CAMERAS, ZONE, logger
from .gate import GateControl
from .model import PlateDetector
from .sync import SyncManager


class GateProcessor:
    def __init__(self, gate_type, camera_url, roi_config):
        self.gate_type = gate_type
        self.camera = None
        self.failed_attempts = 0
        self.gate = GateControl(gate_type)
        self.model = PlateDetector(roi_config)
        self.is_logged = False
        self.camera_url = camera_url
        self.running = False
        
    def connect_to_stream(self):
        try:
            logger.info(f"🔗 Connecting to {self.gate_type} Camera Stream: {self.camera_url}")
            self.camera = CameraStream(self.camera_url, self.gate_type)
            self.camera.start()

            if self.camera.isOpened():
                logger.info(f"✅ Connection successful for {self.gate_type} camera")
                return True

        except Exception as e:
            logger.error(f"❌ Connection failed for {self.gate_type} camera: {str(e)}")
            if self.camera:
                self.camera.release()

        return False
        
    def start_processing(self, cache):
        """Start processing frames from this gate's camera"""
        self.running = True
        Thread(target=self._process_loop, args=(cache,), daemon=True).start()
        
    def _process_loop(self, cache):
        """Main processing loop for this gate"""
        try:
            if not self.connect_to_stream():
                logger.error(f"Failed to connect to {self.gate_type} Camera Stream")
                self.running = False
                return

            logger.info(f"🎥 Starting license plate detection for {self.gate_type} gate...")

            while self.running:
                if not self.camera.isOpened():
                    self.connect_to_stream()
                    continue

                frame = self.camera.read()
                if frame is None:
                    self.camera.release()
                    self.connect_to_stream()
                    continue

                self.process_frame(frame, cache)
                sleep(0.01)  # Small sleep to reduce CPU usage

        except Exception as e:
            logger.error(f"\n❌ An error occurred in {self.gate_type} gate processing: {str(e)}")
        finally:
            if self.camera:
                self.camera.release()
                logger.info(f"✅ Released {self.gate_type} video capture resources")
    
    def process_frame(self, frame, cache):
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

            is_authorized = cache.is_authorized(license_text)
            auth_status = (
                "✅ Authorized"
                if is_authorized
                else "❌ Not Authorized"
                + (" [Logged]" if self.is_logged else "")
            )
            status_message = (
                f"✨ License Plate Detected on {self.gate_type.upper()} gate!\n"
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
                if self.failed_attempts >= 1 and not self.is_logged:
                    self.log_gate_entry(license_text, frame, 0)
                    self.is_logged = True

                self.display_status(
                    status_message,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )

        except Exception as e:
            logger.error(f"❌ Error processing frame on {self.gate_type} gate: {str(e)}")
            
    def display_status(self, message, timestamp):
        logger.info(f"🕒 {timestamp} - {self.gate_type.upper()} gate")
        logger.info(f"\n{message}\n")

    def log_gate_entry(self, plate, frame, is_authorized):
        try:
            frame_with_text = self.model.add_text_to_image(frame, plate)
            frame_with_roi = self.model.visualize_roi(frame_with_text)
            temp_file = f"gate_entry_{self.gate_type}.jpg"
            cv2.imwrite(temp_file, frame_with_roi)

            log_data = {
                "zone": ZONE,
                "license_plate": plate,
                "authorized": is_authorized,
                "gate_type": self.gate_type,
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
            logger.error(f"❌ Error logging entry for {self.gate_type} gate: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class SpherexAgent:
    def __init__(self):
        self.cache = SyncManager()
        self.gate_processors = {}
        
        # Initialize gate processors for all configured cameras
        for gate_type, camera_config in CAMERAS.items():
            camera_url = camera_config["url"]
            roi_config = camera_config["roi_config"]
            
            if camera_url:
                self.gate_processors[gate_type] = GateProcessor(
                    gate_type, camera_url, roi_config
                )
                logger.info(f"Initialized {gate_type} gate processor")
            else:
                logger.info(f"Skipping {gate_type} gate - no camera URL configured")

    def start_processing(self):
        """Main processing loop"""
        try:
            if not self.gate_processors:
                logger.error("No gates configured! Please check your configuration.")
                return
                
            logger.info("🎥 Starting license plate detection system...\n")
            logger.info("Press Ctrl+C to stop the program\n")
            
            # Start all gate processors
            for gate_type, processor in self.gate_processors.items():
                processor.start_processing(self.cache)
                logger.info(f"Started processing for {gate_type} gate")
            
            # Keep main thread alive until keyboard interrupt
            while True:
                sleep(1)

        except KeyboardInterrupt:
            logger.info("\n👋 Stopping license plate detection...")
        except Exception as e:
            logger.error(f"\n❌ An error occurred: {str(e)}")
        finally:
            # Stop all processors
            for gate_type, processor in self.gate_processors.items():
                processor.running = False
                logger.info(f"Stopped {gate_type} processor")
