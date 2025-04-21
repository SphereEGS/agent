import os
import cv2
import requests
from datetime import datetime
from time import sleep, time
import threading
import numpy as np

# Direct imports (no package imports)
from app.camera import InputStream  # Change to relative import
from app.config import API_BASE_URL, GATE, logger, CAMERA_URL, PROCESS_EVERY
from app.gate import GateControl
from app.lpr_model import PlateProcessor
from app.vehicle_tracking import VehicleTracker
from app.sync import SyncManager


class SpherexAgent:
    def __init__(self):
        logger.info("[AGENT] Initializing SpherexAgent")
        self.streams = {}
        self.gate = GateControl()
        self.processor = PlateProcessor()
        self.vehicle_trackers = {}
        self.cache = SyncManager()
        self.frame_counts = {}
        self.last_detection_times = {}
        self.detection_cooldown = 2  # 2 second cooldown between detections
        self.camera_urls = self._load_camera_urls_from_env()
        logger.info("[AGENT] SpherexAgent initialized successfully")
    
    def _load_camera_urls_from_env(self):
        """Load camera URLs from environment variables"""
        camera_urls = {}
        
        # Look for numbered camera URLs
        for key, value in os.environ.items():
            if key.startswith("CAMERA_URL_"):
                camera_id = key[len("CAMERA_URL_"):].lower()
                # Clean any quotes or whitespace
                camera_url = value.strip()
                if camera_url.startswith('"') and camera_url.endswith('"'):
                    camera_url = camera_url[1:-1].strip()
                elif camera_url.startswith("'") and camera_url.endswith("'"):
                    camera_url = camera_url[1:-1].strip()
                
                camera_urls[camera_id] = camera_url
                logger.info(f"[AGENT] Found camera {camera_id}: {camera_url}")
        
        # Use default camera if no numbered cameras found
        if not camera_urls and "CAMERA_URL" in os.environ:
            default_url = os.environ["CAMERA_URL"].strip()
            if default_url.startswith('"') and default_url.endswith('"'):
                default_url = default_url[1:-1].strip()
            elif default_url.startswith("'") and default_url.endswith("'"):
                default_url = default_url[1:-1].strip()
            
            camera_urls["main"] = default_url
            logger.info(f"[AGENT] Using default camera: {default_url}")
            
        # If still no cameras, add a default
        if not camera_urls:
            logger.warning("[AGENT] No cameras found in environment variables, using default")
            camera_urls["main"] = CAMERA_URL
        
        return camera_urls

    def initialize_streams(self):
        """Initialize all configured camera streams"""
        if not self.camera_urls:
            logger.error("[AGENT] No camera URLs configured")
            return False
        
        success = False
        
        for camera_id, camera_url in self.camera_urls.items():
            logger.info(f"[AGENT] Initializing camera {camera_id} with URL: {camera_url}")
            try:
                # Create input stream for this camera
                self.streams[camera_id] = InputStream(camera_url=camera_url, camera_id=camera_id)
                
                # Create vehicle tracker for this camera
                self.vehicle_trackers[camera_id] = VehicleTracker(camera_id=camera_id)
                
                # Initialize frame count and detection time
                self.frame_counts[camera_id] = 0
                self.last_detection_times[camera_id] = 0
                
                logger.info(f"[AGENT] Successfully initialized camera {camera_id}")
                success = True
            except Exception as e:
                logger.error(f"[AGENT] Failed to initialize camera {camera_id}: {e}")
                # Continue with other cameras
        
        return success

    def log_gate_entry(self, plate, frame, is_authorized, camera_id="main"):
        """Log gate entry to the server and save image"""
        try:
            # Add text and visualize ROI on the frame
            logger.info(f"[GATE] Processing gate entry for plate: {plate}, authorized: {is_authorized}")
            frame_with_text = self.processor.add_text_to_image(frame, plate)
            frame_with_roi = self.processor.visualize_roi(frame_with_text)
            
            # Save temporary image file
            temp_file = f"gate_entry_{camera_id}.jpg"
            cv2.imwrite(temp_file, frame_with_roi)
            logger.info(f"[GATE] Saved gate entry image to {temp_file}")

            # For testing, we'll skip the actual server upload
            logger.info(f"[GATE] Logging entry from camera {camera_id}: Plate={plate}, Authorized={is_authorized}")
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"[GATE] Cleaned up temporary image file")
                
        except Exception as e:
            logger.error(f"[GATE] Error logging entry: {e}")
    
    def update_fps(self, camera_id):
        """Update FPS for a specific camera"""
        if camera_id not in self.fps_stats:
            self.fps_stats[camera_id] = {
                'start_time': time(),
                'frame_count': 0
            }
        
        # Increment frame count
        self.fps_stats[camera_id]['frame_count'] += 1
        
        # Check if it's time to update FPS
        if self.fps_stats[camera_id]['frame_count'] % 100 == 0:
            elapsed = time() - self.fps_stats[camera_id]['start_time']
            fps = self.fps_stats[camera_id]['frame_count'] / elapsed
            logger.info(f"[AGENT:{camera_id}] Processing speed: {fps:.1f} FPS ({self.fps_stats[camera_id]['frame_count']} frames in {elapsed:.1f}s)")
            
            # Reset counters for more accurate recent FPS
            self.fps_stats[camera_id]['start_time'] = time()
            self.fps_stats[camera_id]['frame_count'] = 0

    def process_frame(self, frame, camera_id):
        """Process a single frame from a specific camera"""
        if frame is None:
            logger.warning(f"[AGENT:{camera_id}] Received None frame, skipping processing")
            return None
            
        self.frame_counts[camera_id] += 1
        
        # Log frame processing periodically
        if self.frame_counts[camera_id] % 100 == 0:
            logger.info(f"[AGENT:{camera_id}] Processing frame {self.frame_counts[camera_id]}")
        
        # Create a working copy of the frame for visualization
        display_frame = frame.copy()
        
        # Add frame counter to display
        cv2.putText(display_frame, f"Camera: {camera_id} | Frame: {self.frame_counts[camera_id]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Skip frames based on PROCESS_EVERY setting but always display
        if self.frame_counts[camera_id] % PROCESS_EVERY == 0:
            # Check cooldown period
            current_time = time()
            time_since_last = current_time - self.last_detection_times.get(camera_id, 0)
            
            if time_since_last > self.detection_cooldown:
                logger.debug(f"[AGENT:{camera_id}] Running detection on frame {self.frame_counts[camera_id]}")
                try:
                    # Get vehicle tracker for this camera
                    vehicle_tracker = self.vehicle_trackers.get(camera_id)
                    if not vehicle_tracker:
                        logger.error(f"[AGENT:{camera_id}] No vehicle tracker found for this camera")
                        return display_frame
                    
                    # Store the previous count of detected plates
                    prev_plate_count = len(vehicle_tracker.detected_plates)
                    prev_plates = set(vehicle_tracker.detected_plates.items())
                    
                    # Detect vehicles and license plates
                    detected, vis_frame = vehicle_tracker.detect_vehicles(frame)
                    
                    if detected and vis_frame is not None:
                        # Use the visualization frame that comes from the tracker
                        display_frame = vis_frame
                        
                        # Only log plate info if there's a change in detected plates
                        curr_plates = set(vehicle_tracker.detected_plates.items())
                        new_plates = curr_plates - prev_plates
                        
                        if new_plates:
                            logger.info(f"[AGENT:{camera_id}] Newly detected plates: {dict(new_plates)}")
                            
                            # Process newly detected plates
                            for track_id, plate_text in new_plates:
                                # Check if plate is authorized
                                is_authorized = self.cache.is_authorized(plate_text)
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Update the authorization status for display
                                # Make sure the vehicle tracker's last recognized plate is updated
                                vehicle_tracker.last_recognized_plate = plate_text
                                vehicle_tracker.last_plate_authorized = is_authorized
                                logger.info(f"[AGENT:{camera_id}] Updated last_recognized_plate to {plate_text}, auth: {is_authorized}")
                                
                                # Log the detection
                                auth_status = "Authorized" if is_authorized else "Not Authorized"
                                logger.info(f"[AGENT:{camera_id}] [{timestamp}] Vehicle {track_id} with plate: {plate_text} - {auth_status}")
                                
                                # Handle gate control
                                if is_authorized:
                                    logger.info(f"[GATE] Opening gate for authorized plate: {plate_text}")
                                    self.gate.open()
                                    self.log_gate_entry(plate_text, frame, 1, camera_id)
                                    self.last_detection_times[camera_id] = current_time
                                else:
                                    logger.info(f"[GATE] Not opening gate for unauthorized plate: {plate_text}")
                                    self.log_gate_entry(plate_text, frame, 0, camera_id)
                        elif self.frame_counts[camera_id] % 200 == 0:
                            # Log total plate count periodically
                            plates = vehicle_tracker.detected_plates
                            logger.info(f"[AGENT:{camera_id}] Total plates detected: {len(plates)}")
                    elif vis_frame is not None:
                        # Even if no detection, use the visualization frame which should have ROI
                        display_frame = vis_frame
                        if self.frame_counts[camera_id] % 50 == 0:  # Less frequent logging for no detections
                            logger.debug(f"[AGENT:{camera_id}] No vehicle detections in frame {self.frame_counts[camera_id]}")
                        
                except Exception as e:
                    logger.error(f"[AGENT:{camera_id}] Frame processing error: {e}")
                    # If no visualization available, make sure ROI is still drawn on display frame
                    if vehicle_tracker.roi_manager.detection_roi_polygon is not None:
                        cv2.polylines(display_frame, [vehicle_tracker.roi_manager.detection_roi_polygon], True, (0, 0, 255), 3)
            else:
                if self.frame_counts[camera_id] % 50 == 0:  # Less frequent logging for cooldown
                    logger.debug(f"[AGENT:{camera_id}] In cooldown period, {self.detection_cooldown - time_since_last:.1f}s remaining")
        
        # Return the processed frame (we'll display in grid in the main loop)
        return display_frame

    def create_grid_view(self, frames):
        """
        Create a grid view from multiple camera frames
        
        Args:
            frames: Dictionary of {camera_id: frame}
        
        Returns:
            Combined grid frame
        """
        # Get the size of frames and count
        frame_count = len(frames)
        if frame_count == 0:
            return None
        
        # Determine grid dimensions based on number of cameras
        if frame_count == 1:
            grid_cols, grid_rows = 1, 1
        elif frame_count <= 4:
            grid_cols, grid_rows = 2, 2
        elif frame_count <= 9:
            grid_cols, grid_rows = 3, 3
        else:
            grid_cols, grid_rows = 4, 3  # Max 12 cameras
        
        # Get dimensions from first frame
        sample_frame = next(iter(frames.values()))
        frame_h, frame_w = sample_frame.shape[:2]
        
        # Calculate target size for each grid cell
        # We'll resize all frames to this size
        target_w = min(480, frame_w)
        target_h = int(target_w * frame_h / frame_w)
        
        # Create black canvas for the grid
        grid_width = target_w * grid_cols
        grid_height = target_h * grid_rows
        grid_view = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Sort camera IDs to maintain consistent positions
        camera_ids = sorted(frames.keys())
        
        # Place frames in grid
        for i, camera_id in enumerate(camera_ids):
            if i >= grid_rows * grid_cols:
                break  # Don't exceed grid size
            
            frame = frames[camera_id]
            
            # Make sure we're preserving aspect ratio properly when resizing
            current_h, current_w = frame.shape[:2]
            aspect_ratio = current_w / current_h
            
            # Calculate dimensions that preserve aspect ratio
            if aspect_ratio > (target_w / target_h):
                # Image is wider than target
                resize_w = target_w
                resize_h = int(target_w / aspect_ratio)
            else:
                # Image is taller than target
                resize_h = target_h
                resize_w = int(target_h * aspect_ratio)
                
            # Ensure dimensions are even numbers
            resize_w = resize_w - (resize_w % 2)
            resize_h = resize_h - (resize_h % 2)
            
            # Resize the frame preserving aspect ratio
            resized_frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            
            # Create a full-size cell with black background
            cell = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Calculate position to center the resized frame in the cell
            y_offset = (target_h - resize_h) // 2
            x_offset = (target_w - resize_w) // 2
            
            # Place the resized frame in the cell
            cell[y_offset:y_offset+resize_h, x_offset:x_offset+resize_w] = resized_frame
            
            row = i // grid_cols
            col = i % grid_cols
            
            y_start = row * target_h
            y_end = y_start + target_h
            x_start = col * target_w
            x_end = x_start + target_w
            
            grid_view[y_start:y_end, x_start:x_end] = cell
        
        # Add grid title
        cv2.putText(grid_view, f"SpherexAgent - Monitoring {frame_count} Cameras", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return grid_view

    def start(self):
        """Start the main processing loop for all cameras"""
        logger.info("[AGENT] Starting SpherexAgent...")
        
        if not self.initialize_streams():
            logger.error("[AGENT] Failed to initialize one or more streams. Continuing with available streams.")
            if not self.streams:
                logger.error("[AGENT] No streams available. Exiting.")
                return

        logger.info(f"[AGENT] Starting video processing for {len(self.streams)} cameras... Press 'q' to stop the program")

        try:
            frame_counts = {camera_id: 0 for camera_id in self.streams}
            start_times = {camera_id: time() for camera_id in self.streams}
            self.fps_stats = {camera_id: {'start_time': time(), 'frame_count': 0} for camera_id in self.streams}
            
            # Create single window for grid view
            window_name = "SpherexAgent - Multi-Camera View"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # If we have 4 or more cameras, make the window larger
            if len(self.streams) >= 4:
                cv2.resizeWindow(window_name, 1280, 720)
            
            processed_frames = {}
            
            while True:
                # Process each camera stream
                for camera_id, stream in self.streams.items():
                    # Read frame and check if successful
                    ret, frame = stream.read()
                    
                    if not ret or frame is None:
                        logger.warning(f"[AGENT:{camera_id}] Failed to read frame, retrying...")
                        # Use last processed frame if available, otherwise skip
                        if camera_id in processed_frames:
                            continue
                        else:
                            # Create a black frame with error message
                            h, w = 480, 640
                            if camera_id in processed_frames and processed_frames[camera_id] is not None:
                                h, w = processed_frames[camera_id].shape[:2]
                            
                            error_frame = np.zeros((h, w, 3), dtype=np.uint8)
                            cv2.putText(error_frame, f"Camera {camera_id}: No Signal", 
                                      (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1.0, (0, 0, 255), 2)
                            processed_frames[camera_id] = error_frame
                            continue
                    
                    # Update FPS counter
                    self.fps_stats[camera_id]['frame_count'] += 1
                    if self.fps_stats[camera_id]['frame_count'] % 100 == 0:
                        elapsed = time() - self.fps_stats[camera_id]['start_time']
                        fps = self.fps_stats[camera_id]['frame_count'] / elapsed if elapsed > 0 else 0
                        logger.info(f"[AGENT:{camera_id}] Processing speed: {fps:.1f} FPS ({self.fps_stats[camera_id]['frame_count']} frames in {elapsed:.1f}s)")
                        # Reset counters for more accurate recent FPS
                        self.fps_stats[camera_id]['start_time'] = time()
                        self.fps_stats[camera_id]['frame_count'] = 0
                    
                    # Process the frame
                    processed_frame = self.process_frame(frame, camera_id)
                    if processed_frame is not None:
                        processed_frames[camera_id] = processed_frame
                
                # Create and display the grid view
                if processed_frames:
                    grid_view = self.create_grid_view(processed_frames)
                    if grid_view is not None:
                        cv2.imshow(window_name, grid_view)
                
                # Check for 'q' key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
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
            # Release all streams and close windows
            for camera_id, stream in self.streams.items():
                if stream:
                    stream.release()
            cv2.destroyAllWindows()
            logger.info("[AGENT] Processing stopped.")

if __name__ == "__main__":
    agent = SpherexAgent()
    agent.start()