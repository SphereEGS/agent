import os
from datetime import datetime
from time import sleep, time

import cv2
import requests

from app.camera import InputStream
from app.config import API_BASE_URL, CAMERA_URL, GATE, PROCESS_EVERY, logger
from app.gate import GateControl
from app.lpr_model import PlateProcessor
from app.sync import SyncManager
from app.vehicle_tracking import VehicleTracker


class SpherexAgent:
    def __init__(self):
        logger.info("[AGENT] Initializing SpherexAgent")
        self.stream = None
        self.gate = GateControl()
        self.processor = PlateProcessor()
        self.vehicle_tracker = VehicleTracker()
        self.cache = SyncManager()
        self.frame_count = 0
        self.last_detection_time = 0
        self.detection_cooldown = 2  # 2 second cooldown between detections
        self.is_logged = False
        logger.info("[AGENT] SpherexAgent initialized successfully")

    def initialize_stream(self):
        """Initialize the input stream (camera or video)"""
        logger.info(
            f"[AGENT] Initializing input stream from source: {CAMERA_URL}"
        )
        try:
            self.stream = InputStream()
            logger.info("[AGENT] Input stream initialized successfully")
            return True
        except Exception as e:
            logger.error(f"[AGENT] Failed to initialize input stream: {e}")
            return False

    def log_gate_entry(self, plate, frame, is_authorized):
        try:
            frame_with_text = self.processor.add_text_to_image(frame, plate)
            frame_with_roi = self.processor.visualize_roi(frame_with_text)
            temp_file = "gate_entry.jpg"
            cv2.imwrite(temp_file, frame_with_roi)

            log_data = {
                "gate": GATE,
                "license_plate": plate,
                "authorized": is_authorized,
                "image": temp_file,
                "access_type": "Entry"
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

    def process_frame(self, frame):
        """Process a single frame"""
        if frame is None:
            logger.warning("[AGENT] Received None frame, skipping processing")
            return

        self.frame_count += 1

        # Log frame processing periodically
        if self.frame_count % 100 == 0:
            logger.info(f"[AGENT] Processing frame {self.frame_count}")

        # Create a working copy of the frame for visualization
        display_frame = frame.copy()

        # Add frame counter to display
        cv2.putText(
            display_frame,
            f"Frame: {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Skip frames based on PROCESS_EVERY setting but always display
        if self.frame_count % PROCESS_EVERY == 0:
            # Check cooldown period
            current_time = time()
            time_since_last = current_time - self.last_detection_time

            if time_since_last > self.detection_cooldown:
                logger.debug(
                    f"[AGENT] Running detection on frame {self.frame_count}"
                )
                try:
                    # Store the previous count of detected plates
                    prev_plate_count = len(
                        self.vehicle_tracker.detected_plates
                    )
                    prev_plates = set(
                        self.vehicle_tracker.detected_plates.items()
                    )

                    # Detect vehicles and license plates
                    detected, vis_frame = self.vehicle_tracker.detect_vehicles(
                        frame
                    )

                    if detected and vis_frame is not None:
                        # Use the visualization frame that comes from the tracker
                        display_frame = vis_frame

                        # Only log plate info if there's a change in detected plates
                        curr_plates = set(
                            self.vehicle_tracker.detected_plates.items()
                        )
                        new_plates = curr_plates - prev_plates

                        if new_plates:
                            logger.info(
                                f"[AGENT] Newly detected plates: {dict(new_plates)}"
                            )

                            # Process newly detected plates
                            for track_id, plate_text in new_plates:
                                # Check if plate is authorized
                                is_authorized = self.cache.is_authorized(
                                    plate_text
                                )
                                timestamp = datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )

                                # Update the authorization status for display
                                # Make sure the vehicle tracker's last recognized plate is updated
                                self.vehicle_tracker.last_recognized_plate = (
                                    plate_text
                                )
                                self.vehicle_tracker.last_plate_authorized = (
                                    is_authorized
                                )
                                logger.info(
                                    f"[AGENT] Updated last_recognized_plate to {plate_text}, auth: {is_authorized}"
                                )

                                # Log the detection
                                auth_status = (
                                    "Authorized"
                                    if is_authorized
                                    else "Not Authorized"
                                )
                                logger.info(
                                    f"[AGENT] [{timestamp}] Vehicle {track_id} with plate: {plate_text} - {auth_status}"
                                )

                                # Handle gate control
                                if is_authorized:
                                    logger.info(
                                        f"[GATE] Opening gate for authorized plate: {plate_text}"
                                    )
                                    # self.gate.open()
                                    self.log_gate_entry(plate_text, vis_frame, 1)
                                    self.last_detection_time = current_time
                                else:
                                    logger.info(
                                        f"[GATE] Not opening gate for unauthorized plate: {plate_text}"
                                    )
                                    self.log_gate_entry(plate_text, vis_frame, 0)
                                    self.is_logged = True
                        elif self.frame_count % 200 == 0:
                            # Log total plate count periodically
                            plates = self.vehicle_tracker.detected_plates
                            logger.info(
                                f"[AGENT] Total plates detected: {len(plates)}"
                            )
                    elif vis_frame is not None:
                        # Even if no detection, use the visualization frame which should have ROI
                        display_frame = vis_frame
                        if (
                            self.frame_count % 50 == 0
                        ):  # Less frequent logging for no detections
                            logger.debug(
                                f"[AGENT] No vehicle detections in frame {self.frame_count}"
                            )

                except Exception as e:
                    logger.error(f"[AGENT] Frame processing error: {e}")
                    # If no visualization available, make sure ROI is still drawn on display frame
                    if self.vehicle_tracker.roi_polygon is not None:
                        cv2.polylines(
                            display_frame,
                            [self.vehicle_tracker.roi_polygon],
                            True,
                            (0, 0, 255),
                            3,
                        )
            else:
                if (
                    self.frame_count % 50 == 0
                ):  # Less frequent logging for cooldown
                    logger.debug(
                        f"[AGENT] In cooldown period, {self.detection_cooldown - time_since_last:.1f}s remaining"
                    )

        key = cv2.waitKey(1) & 0xFF

        # Allow quitting with 'q' key
        if key == ord("q"):
            logger.info("[AGENT] User pressed 'q', exiting")
            self.stream.release()
            cv2.destroyAllWindows()
            import sys

            sys.exit(0)

    def start(self):
        """Start the main processing loop"""
        logger.info("[AGENT] Starting SpherexAgent...")

        if not self.initialize_stream():
            logger.error("[AGENT] Failed to initialize stream. Exiting.")
            return

        logger.info(
            f"[AGENT] Starting video processing... Press 'q' to stop the program"
        )

        try:
            frame_count = 0
            start_time = time()

            while self.stream:
                # Read frame and check if successful
                ret, frame = self.stream.read()

                if not ret or frame is None:
                    logger.warning("[AGENT] Failed to read frame, retrying...")
                    # Wait a bit before retrying
                    sleep(0.1)
                    continue

                # Calculate FPS every 100 frames
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"[AGENT] Processing speed: {fps:.1f} FPS ({frame_count} frames in {elapsed:.1f}s)"
                    )
                    # Reset counters for more accurate recent FPS
                    start_time = time()
                    frame_count = 0

                self.process_frame(frame)

                # Check for 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("[AGENT] Stopping due to user input (q)...")
                    break

                # Add a small sleep to reduce CPU usage
                sleep(0.01)

        except KeyboardInterrupt:
            logger.info("[AGENT] Stopping due to keyboard interrupt...")
        except Exception as e:
            logger.error(f"[AGENT] Processing error: {e}")
        finally:
            if self.stream:
                self.stream.release()
            cv2.destroyAllWindows()
            logger.info("[AGENT] Processing stopped.")


if __name__ == "__main__":
    agent = SpherexAgent()
    agent.start()
