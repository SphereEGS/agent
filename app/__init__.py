import os
from datetime import datetime
from time import sleep, time

import cv2
import requests
import threading

from app.camera import InputStream
from app.config import API_BASE_URL, CAMERA_URLS, CAMERA_TYPES, GATE, PROCESS_EVERY, logger
from app.gate import GateControl
from app.lpr_model import PlateProcessor
from app.sync import SyncManager
from app.vehicle_tracking import VehicleTracker


class CameraManager:
    """Manages multiple camera streams and their associated trackers"""
    def __init__(self, shared_plate_processor=None):
        self.streams = {}
        self.trackers = {}
        self.frame_counts = {}
        self.shared_plate_processor = shared_plate_processor
        
        # Initialize cameras
        for camera_id, url in CAMERA_URLS.items():
            logger.info(f"[MANAGER] Initializing camera {camera_id} with URL {url}")
            self.frame_counts[camera_id] = 0
            
    def initialize_streams(self):
        """Initialize all camera streams"""
        for camera_id in CAMERA_URLS.keys():
            try:
                self.streams[camera_id] = InputStream(camera_id)
                # Pass the shared plate processor to each VehicleTracker
                self.trackers[camera_id] = VehicleTracker(camera_id, plate_processor=self.shared_plate_processor)
                logger.info(f"[MANAGER] Successfully initialized camera {camera_id} and its tracker")
            except Exception as e:
                logger.error(f"[MANAGER] Failed to initialize camera {camera_id}: {e}")
        
        return len(self.streams) > 0
    
    def get_camera_ids(self):
        """Get list of active camera IDs"""
        return list(self.streams.keys())
    
    def read_frame(self, camera_id):
        """Read a frame from a specific camera"""
        if camera_id in self.streams:
            ret, frame = self.streams[camera_id].read()
            if ret:
                self.frame_counts[camera_id] += 1
            return ret, frame, self.frame_counts[camera_id]
        return False, None, 0
    
    def release_all(self):
        """Release all camera resources"""
        for camera_id, stream in self.streams.items():
            try:
                stream.release()
                logger.info(f"[MANAGER] Released camera {camera_id}")
            except Exception as e:
                logger.error(f"[MANAGER] Error releasing camera {camera_id}: {e}")


class SpherexAgent:
    def __init__(self):
        logger.info("[AGENT] Initializing SpherexAgent")
        
        # Common components
        self.gate = GateControl()
        # Create a single shared PlateProcessor for all cameras
        self.processor = PlateProcessor()
        self.cache = SyncManager()
        
        # Camera handling - pass the shared processor
        self.camera_manager = CameraManager(shared_plate_processor=self.processor)
        
        # Tracking variables
        self.last_detection_time = 0
        self.detection_cooldown = 2  # 2 second cooldown between detections
        self.is_logged = False
        
        # Processing flag for each camera
        self.processing_flags = {}

        logger.info("[AGENT] SpherexAgent initialized successfully")
        
        # Flag to indicate if the system should continue running
        self.is_running = True

    def initialize_streams(self):
        """Initialize all camera streams"""
        logger.info("[AGENT] Initializing camera streams")
        success = self.camera_manager.initialize_streams()
        
        if success:
            logger.info(f"[AGENT] Successfully initialized {len(self.camera_manager.get_camera_ids())} camera streams")
            # Initialize processing flags for each camera
            for camera_id in self.camera_manager.get_camera_ids():
                self.processing_flags[camera_id] = True
            return True
        else:
            logger.error("[AGENT] Failed to initialize any camera streams")
            return False

    def log_gate_entry(self, plate, frame, is_authorized, camera_id="main"):
        try:
            frame_with_text = self.processor.add_text_to_image(frame, plate)
            
            # Try to get the ROI for this specific camera
            if camera_id in self.camera_manager.trackers:
                tracker = self.camera_manager.trackers[camera_id]
                frame_with_roi = self.processor.visualize_roi(frame_with_text, tracker.roi_polygon)
            #else:
            #    frame_with_roi = frame_with_text
                
            temp_file = f"gate_entry_{camera_id}.jpg"
            cv2.imwrite(temp_file, frame_with_roi)

            log_data = {
                "gate": GATE,
                "license_plate": plate,
                "authorized": is_authorized,
                "image": temp_file,
                "access_type": CAMERA_TYPES.get(camera_id, "Entry"),
                "camera": camera_id
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
            logger.error(f"❌ Error logging entry from camera {camera_id}: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def process_frame(self, frame, camera_id, frame_count):
        """Process a single frame from a specific camera"""
        if frame is None:
            logger.warning(f"[AGENT:{camera_id}] Received None frame, skipping processing")
            return

        # Create a working copy of the frame for visualization
        display_frame = frame.copy()

        # Add frame counter and camera ID to display
        cv2.putText(
            display_frame,
            f"Frame: {frame_count} - Camera: {camera_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Skip frames based on PROCESS_EVERY setting but always display
        if frame_count % PROCESS_EVERY == 0:
            # Check cooldown period
            current_time = time()
            time_since_last = current_time - self.last_detection_time

            if time_since_last > self.detection_cooldown:
                logger.debug(
                    f"[AGENT:{camera_id}] Running detection on frame {frame_count}"
                )
                try:
                    # Get the tracker for this camera
                    tracker = self.camera_manager.trackers[camera_id]
                    
                    # Store the previous count of detected plates
                    prev_plate_count = len(tracker.detected_plates)
                    prev_plates = set(tracker.detected_plates.items())

                    # Detect vehicles and license plates
                    detected, vis_frame = tracker.detect_vehicles(frame)

                    if detected and vis_frame is not None:
                        # Use the visualization frame that comes from the tracker
                        display_frame = vis_frame

                        # Only log plate info if there's a change in detected plates
                        curr_plates = set(tracker.detected_plates.items())
                        new_plates = curr_plates - prev_plates

                        if new_plates:
                            logger.info(
                                f"[AGENT:{camera_id}] Newly detected plates: {dict(new_plates)}"
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
                                tracker.last_recognized_plate = plate_text
                                tracker.last_plate_authorized = is_authorized
                                logger.info(
                                    f"[AGENT:{camera_id}] Updated last_recognized_plate to {plate_text}, auth: {is_authorized}"
                                )

                                # Log the detection
                                auth_status = (
                                    "Authorized"
                                    if is_authorized
                                    else "Not Authorized"
                                )
                                logger.info(
                                    f"[AGENT:{camera_id}] [{timestamp}] Vehicle {track_id} with plate: {plate_text} - {auth_status}"
                                )

                                # Handle gate control
                                if is_authorized:
                                    logger.info(
                                        f"[GATE] Opening gate for authorized plate: {plate_text} detected by camera {camera_id}"
                                    )
                                    cam_type = CAMERA_TYPES.get(camera_id, "Entry")
                                    if cam_type.lower() == "entry":
                                        self.gate.open_entry()
                                        sleep(5)  # Allow time for gate to open
                                        self.gate.close_entry()
                                    elif cam_type.lower() == "exit":
                                        self.gate.open_exit()
                                        sleep(5)
                                        self.gate.close_exit()
                                    else:
                                        logger.warning(f"Unknown camera type '{cam_type}' for camera {camera_id}, defaulting to entry barrier")
                                        self.gate.open_entry()
                                    self.log_gate_entry(plate_text, vis_frame, 1, camera_id)
                                    self.last_detection_time = current_time
                                else:
                                    logger.info(
                                        f"[GATE] Not opening gate for unauthorized plate: {plate_text} detected by camera {camera_id}"
                                    )
                                    self.log_gate_entry(plate_text, vis_frame, 0, camera_id)
                                    self.is_logged = True
                        elif frame_count % 200 == 0:
                            # Log total plate count periodically
                            plates = tracker.detected_plates
                            logger.info(
                                f"[AGENT:{camera_id}] Total plates detected: {len(plates)}"
                            )
                    elif vis_frame is not None:
                        # Even if no detection, use the visualization frame which should have ROI
                        display_frame = vis_frame
                        if (
                            frame_count % 50 == 0
                        ):  # Less frequent logging for no detections
                            logger.debug(
                                f"[AGENT:{camera_id}] No vehicle detections in frame {frame_count}"
                            )

                except Exception as e:
                    logger.error(f"[AGENT:{camera_id}] Frame processing error: {e}")
                    
    def process_camera(self, camera_id):
        """Process frames from a specific camera in a continuous loop"""
        logger.info(f"[AGENT] Starting processing loop for camera {camera_id}")
        
        while self.is_running and self.processing_flags.get(camera_id, False):
            ret, frame, frame_count = self.camera_manager.read_frame(camera_id)
            
            if ret:
                self.process_frame(frame, camera_id, frame_count)
            else:
                logger.warning(f"[AGENT] Failed to read frame from camera {camera_id}")
                sleep(0.5)  # Avoid tight loop if camera is failing
        
        logger.info(f"[AGENT] Stopped processing camera {camera_id}")

    def start(self):
        """Start the main processing loop"""
        logger.info("[AGENT] Starting SpherexAgent...")

        if not self.initialize_streams():
            logger.error("[AGENT] Failed to initialize streams. Exiting.")
            return

        logger.info(
            f"[AGENT] Starting video processing for {len(self.camera_manager.get_camera_ids())} cameras... Press 'q' to stop the program"
        )

        try:
            # Start a thread for each camera
            threads = []
            for camera_id in self.camera_manager.get_camera_ids():
                thread = threading.Thread(
                    target=self.process_camera, 
                    args=(camera_id,),
                    daemon=True
                )
                thread.start()
                threads.append(thread)
                # Add a delay between starting camera threads to avoid resource contention
                time.sleep(1)  # 1 second delay between camera threads
                logger.info(f"[AGENT] Started thread for camera {camera_id}, waiting 1s before next camera")
                
            # Main thread monitors keypresses for exiting
            while self.is_running:
                key = cv2.waitKey(1) & 0xFF
                
                # Allow quitting with 'q' key
                if key == ord("q"):
                    logger.info("[AGENT] User pressed 'q', exiting")
                    self.is_running = False
                    break
                
                sleep(0.1)  # Reduce CPU usage in main thread

        except KeyboardInterrupt:
            logger.info("[AGENT] Keyboard interrupt received, exiting")
        except Exception as e:
            logger.error(f"[AGENT] Unexpected error: {e}")
        finally:
            # Clean up
            self.is_running = False
            logger.info("[AGENT] Stopping all camera threads")
            
            # Wait for all threads to finish
            for thread in threads:
                thread.join(timeout=1.0)
                
            # Release all cameras
            self.camera_manager.release_all()
            cv2.destroyAllWindows()
            logger.info("[AGENT] SpherexAgent shutdown complete")


if __name__ == "__main__":
    agent = SpherexAgent()
    agent.start()
