import os
import cv2
import requests
from datetime import datetime
from time import sleep, time

# Direct imports (no package imports)
from app.camera import InputStream  # Change to relative import
from app.config import API_BASE_URL, ZONE, logger, INPUT_SOURCE, PROCESS_EVERY
from app.gate import GateControl
from app.lpr_model import PlateProcessor
from app.vehicle_tracking import VehicleTracker
from app.sync import SyncManager


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
        logger.info(f"[AGENT] Initializing input stream from source: {INPUT_SOURCE}")
        try:
            self.stream = InputStream()
            logger.info(f"[AGENT] Input stream initialized successfully")
            return True
        except Exception as e:
            logger.error(f"[AGENT] Failed to initialize input stream: {e}")
            return False

    def log_gate_entry(self, plate, frame, is_authorized):
        """Log gate entry to the server and save image"""
        try:
            # Add text and visualize ROI on the frame
            logger.info(f"[GATE] Processing gate entry for plate: {plate}, authorized: {is_authorized}")
            frame_with_text = self.processor.add_text_to_image(frame, plate)
            frame_with_roi = self.processor.visualize_roi(frame_with_text)
            
            # Save temporary image file
            temp_file = "gate_entry.jpg"
            cv2.imwrite(temp_file, frame_with_roi)
            logger.info(f"[GATE] Saved gate entry image to {temp_file}")

            # For testing, we'll skip the actual server upload
            logger.info(f"[GATE] Logging entry: Plate={plate}, Authorized={is_authorized}")
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"[GATE] Cleaned up temporary image file")
                
        except Exception as e:
            logger.error(f"[GATE] Error logging entry: {e}")

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
        cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Skip frames based on PROCESS_EVERY setting but always display
        if self.frame_count % PROCESS_EVERY == 0:
            # Check cooldown period
            current_time = time()
            time_since_last = current_time - self.last_detection_time
            
            if time_since_last > self.detection_cooldown:
                logger.debug(f"[AGENT] Running detection on frame {self.frame_count}")
                try:
                    # Detect vehicles and license plates
                    detected, vis_frame = self.vehicle_tracker.detect_vehicles(frame)
                    
                    if detected and vis_frame is not None:
                        # Use the visualization frame that comes from the tracker
                        display_frame = vis_frame
                        
                        # Process detected license plates
                        plates = self.vehicle_tracker.detected_plates
                        logger.info(f"[AGENT] Detected {len(plates)} plates: {plates}")
                        
                        for track_id, plate_text in plates.items():
                            # Check if plate is authorized
                            is_authorized = self.cache.is_authorized(plate_text)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Log the detection
                            auth_status = "Authorized" if is_authorized else "Not Authorized"
                            logger.info(f"[AGENT] [{timestamp}] Vehicle {track_id} with plate: {plate_text} - {auth_status}")
                            
                            # Handle gate control
                            if is_authorized:
                                logger.info(f"[GATE] Opening gate for authorized plate: {plate_text}")
                                self.gate.open()
                                self.log_gate_entry(plate_text, frame, 1)
                                self.last_detection_time = current_time
                            else:
                                logger.info(f"[GATE] Not opening gate for unauthorized plate: {plate_text}")
                                self.log_gate_entry(plate_text, frame, 0)
                                self.is_logged = True
                    elif vis_frame is not None:
                        # Even if no detection, use the visualization frame which should have ROI
                        display_frame = vis_frame
                        if self.frame_count % 50 == 0:  # Less frequent logging for no detections
                            logger.debug(f"[AGENT] No vehicle detections in frame {self.frame_count}")
                        
                except Exception as e:
                    logger.error(f"[AGENT] Frame processing error: {e}")
                    # If no visualization available, make sure ROI is still drawn on display frame
                    if self.vehicle_tracker.roi_polygon is not None:
                        cv2.polylines(display_frame, [self.vehicle_tracker.roi_polygon], True, (0, 0, 255), 3)
            else:
                if self.frame_count % 50 == 0:  # Less frequent logging for cooldown
                    logger.debug(f"[AGENT] In cooldown period, {self.detection_cooldown - time_since_last:.1f}s remaining")
        
        # Always display the current frame
        cv2.imshow('Camera Feed', display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Allow quitting with 'q' key
        if key == ord('q'):
            logger.info("[AGENT] User pressed 'q', exiting")
            self.stream.stop()
            cv2.destroyAllWindows()
            import sys
            sys.exit(0)

    def start(self):
        """Start the main processing loop"""
        logger.info("[AGENT] Starting SpherexAgent...")
        
        if not self.initialize_stream():
            logger.error("[AGENT] Failed to initialize stream. Exiting.")
            return

        logger.info(f"[AGENT] Starting video processing... Press 'q' to stop the program")

        try:
            frame_count = 0
            start_time = time()
            
            while self.stream and self.stream.isOpened():
                frame = self.stream.read()
                
                if frame is None:
                    logger.warning("[AGENT] Failed to read frame. Stopping.")
                    break
                
                # Calculate FPS every 100 frames
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"[AGENT] Processing speed: {fps:.1f} FPS ({frame_count} frames in {elapsed:.1f}s)")
                    # Reset counters for more accurate recent FPS
                    start_time = time()
                    frame_count = 0
                    
                self.process_frame(frame)
                
                # Check for 'q' key press (must have a window open for this to work)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("[AGENT] Stopping due to user input (q)...")
                    break
                
                # Add a small sleep to reduce CPU usage
                sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("[AGENT] Stopping due to keyboard interrupt...")
        except Exception as e:
            logger.error(f"[AGENT] Processing error: {e}")
            import traceback
            logger.error("[AGENT] Stack trace: " + traceback.format_exc())
        finally:
            if self.stream:
                self.stream.stop()
            cv2.destroyAllWindows()
            logger.info("[AGENT] Processing stopped.")

if __name__ == "__main__":
    agent = SpherexAgent()
    agent.start()