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

    def log_gate_entry(self, plate, frame_to_log, is_authorized):
        """Logs the gate entry with the provided frame."""
        try:
            frame_with_text = self.processor.add_text_to_image(frame_to_log, plate)
            frame_with_roi = self.processor.visualize_roi(frame_with_text)
            temp_file = f"gate_entry_{plate}_{time()}.jpg"
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
                upload_response.raise_for_status()
                file_url = upload_response.json()["message"]["file_url"]
                log_data["image"] = file_url
                logger.info(f"[AGENT] Image uploaded successfully: {file_url}")

            logger.info(f"[AGENT] Logging gate entry for plate {plate}...")
            log_response = requests.post(
                f"{API_BASE_URL}/api/resource/Gate Entry Log",
                data=log_data,
            )
            logger.info(f"[AGENT] Gate entry logged successfully for plate {plate}.")
        except Exception as e:
            logger.error(f"❌ Error logging entry for plate {plate}: {e}")
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"[AGENT] Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.error(f"❌ Error removing temporary file {temp_file}: {e}")


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
                    prev_plates = set(
                        self.vehicle_tracker.detected_plates.items()
                    )

                    # Detect vehicles and license plates
                    # Assume detect_vehicles uses 'frame' for inference and returns:
                    # detected (bool), vis_frame (for display), recognized_plates (dict: {track_id: (plate_text, frame_used_for_ocr)})
                    # Modify VehicleTracker accordingly if it doesn't return the frame used for OCR.
                    # For now, we assume the 'frame' passed to detect_vehicles is the one used if recognition occurs.
                    detected, vis_frame, recognized_plates_data = self.vehicle_tracker.detect_vehicles(
                        frame
                    )

                    if detected and vis_frame is not None:
                        # Use the visualization frame that comes from the tracker for display
                        display_frame = vis_frame

                        # Get current plates from the tracker (might include plates detected in previous frames)
                        curr_plates = set(
                            self.vehicle_tracker.detected_plates.items()
                        )
                        # Identify plates newly confirmed in *this* processing cycle
                        newly_confirmed_plates = {
                            track_id: plate_text for track_id, (plate_text, _) in recognized_plates_data.items()
                            if (track_id, plate_text) not in prev_plates
                        }


                        if newly_confirmed_plates:
                            logger.info(
                                f"[AGENT] Newly confirmed plates: {newly_confirmed_plates}"
                            )

                            # Process newly confirmed plates
                            for track_id, plate_text in newly_confirmed_plates.items():
                                # Retrieve the specific frame used for recognizing this plate
                                # This frame should be returned by detect_vehicles or stored in the tracker
                                _, frame_for_logging = recognized_plates_data[track_id]

                                # Check if plate is authorized
                                is_authorized = self.cache.is_authorized(
                                    plate_text
                                )
                                timestamp = datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )

                                # Update the authorization status for display
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

                                # Handle gate control and log using the specific frame
                                if is_authorized:
                                    logger.info(
                                        f"[GATE] Opening gate for authorized plate: {plate_text}"
                                    )
                                    # self.gate.open()
                                    self.log_gate_entry(plate_text, frame_for_logging, 1) # Pass specific frame
                                    self.last_detection_time = current_time
                                else:
                                    logger.info(
                                        f"[GATE] Not opening gate for unauthorized plate: {plate_text}"
                                    )
                                    self.log_gate_entry(plate_text, frame_for_logging, 0) # Pass specific frame
                                    # self.is_logged = True # Consider if this flag is still needed/correct

                        elif self.frame_count % 200 == 0:
                            # Log total plate count periodically if no new plates confirmed
                            plates = self.vehicle_tracker.detected_plates
                            logger.info(
                                f"[AGENT] Total plates currently tracked: {len(plates)}"
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
                    logger.exception(f"[AGENT] Frame processing error: {e}") # Use logger.exception for traceback
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

        # Display the frame (potentially with visualizations)
        # Ensure ROI is always visible on the displayed frame if defined
        if self.vehicle_tracker.roi_polygon is not None:
             cv2.polylines(display_frame, [self.vehicle_tracker.roi_polygon], True, (0, 255, 255), 2) # Yellow ROI

        # Display last recognized plate info
        if self.vehicle_tracker.last_recognized_plate:
            plate_text = self.vehicle_tracker.last_recognized_plate
            auth_text = "AUTH" if self.vehicle_tracker.last_plate_authorized else "UNAUTH"
            color = (0, 255, 0) if self.vehicle_tracker.last_plate_authorized else (0, 0, 255)
            cv2.putText(display_frame, f"Last: {plate_text} ({auth_text})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


        cv2.imshow("Spherex Agent Feed", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Allow quitting with 'q' key
        if key == ord("q"):
            logger.info("[AGENT] User pressed 'q', exiting")
            if self.stream:
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
            frame_read_count = 0 # Use a separate counter for FPS calculation
            start_time = time()

            while True: # Loop indefinitely until break
                if not self.stream:
                    logger.error("[AGENT] Stream is not available. Exiting.")
                    break

                # Read frame and check if successful
                ret, frame = self.stream.read()

                if not ret:
                    logger.warning("[AGENT] Failed to read frame, retrying or ending stream...")
                    # Implement retry logic or check if stream ended
                    if self.stream.is_file(): # Check if it's a file stream that ended
                         logger.info("[AGENT] End of video file reached.")
                         break
                    sleep(0.5) # Wait longer before retrying live stream
                    # Re-initialize stream if necessary
                    if not self.initialize_stream():
                         logger.error("[AGENT] Failed to re-initialize stream. Exiting.")
                         break
                    continue # Skip processing this cycle

                if frame is None:
                     logger.warning("[AGENT] Read successful but frame is None, skipping.")
                     sleep(0.1)
                     continue


                # Calculate FPS periodically
                frame_read_count += 1
                current_time = time()
                elapsed = current_time - start_time
                if elapsed >= 5.0: # Calculate FPS every 5 seconds
                    fps = frame_read_count / elapsed
                    logger.info(
                        f"[AGENT] Processing speed: {fps:.1f} FPS ({frame_read_count} frames in {elapsed:.1f}s)"
                    )
                    # Reset counters
                    start_time = current_time
                    frame_read_count = 0

                # Process the valid frame
                self.process_frame(frame)

                # Check for 'q' key press (handled within process_frame now)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #    logger.info("[AGENT] Stopping due to user input (q)...")
                #    break

                # Small sleep moved inside process_frame's waitKey or removed if waitKey is sufficient

        except KeyboardInterrupt:
            logger.info("[AGENT] Stopping due to keyboard interrupt...")
        except Exception as e:
            logger.exception(f"[AGENT] Unhandled error in main loop: {e}") # Use exception for traceback
        finally:
            if self.stream:
                self.stream.release()
            cv2.destroyAllWindows()
            logger.info("[AGENT] Processing stopped.")


if __name__ == "__main__":
    agent = SpherexAgent()
    agent.start()
