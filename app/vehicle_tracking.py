import cv2
import numpy as np
import json
import threading
import time
import os
import os.path as osp
from collections import defaultdict
from ultralytics import YOLO

from .config import logger, YOLO_MODEL_PATH
from .lpr_model import PlateProcessor

# Vehicle classes from COCO dataset that we want to detect
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

class VehicleTracker:
    def __init__(self, camera_id="main", roi_config_path=None):
        logger.info(f"[TRACKER:{camera_id}] Initializing vehicle detection model...")
        try:
            # Store camera ID for display and logging
            self.camera_id = camera_id
            
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)
            
            # Use absolute path for model
            model_path = osp.abspath(YOLO_MODEL_PATH)
            logger.info(f"[TRACKER:{camera_id}] Loading YOLO model from: {model_path}")
            
            if not osp.exists(model_path):
                logger.error(f"[TRACKER:{camera_id}] Model not found at {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            try:
                self.model = YOLO(model_path)
                logger.info(f"[TRACKER:{camera_id}] YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"[TRACKER:{camera_id}] Error loading YOLO model: {str(e)}")
                raise
            
            # Determine ROI config path if not provided
            if roi_config_path is None:
                # Check for camera-specific config first
                config_dir = "configs"
                camera_config_path = f"{config_dir}/roi_{camera_id}.json"
                
                if os.path.exists(camera_config_path):
                    roi_config_path = camera_config_path
                    logger.info(f"[TRACKER:{camera_id}] Using camera-specific ROI configuration: {camera_config_path}")
                elif camera_id == "main" and os.path.exists("config.json"):
                    # Fallback to legacy config for main camera
                    roi_config_path = "config.json"
                    logger.info(f"[TRACKER:{camera_id}] Using legacy ROI configuration: {roi_config_path}")
                else:
                    # No config found
                    roi_config_path = None
                    logger.warning(f"[TRACKER:{camera_id}] No ROI configuration found for camera.")
            
            # Load ROI configuration
            self.original_roi = self._load_roi_polygon(roi_config_path)
            self.roi_polygon = self.original_roi  # Will be scaled per frame later
            
            if self.original_roi is not None:
                logger.info(f"[TRACKER:{camera_id}] ROI loaded with {len(self.original_roi)} points")
                logger.debug(f"[TRACKER:{camera_id}] Original ROI points: {self.original_roi.tolist()}")
            else:
                logger.warning(f"[TRACKER:{camera_id}] No ROI configuration found. Using full frame.")
                
            # Initialize rest of components
            self.roi_lock = threading.Lock()
            self.plate_processor = PlateProcessor()
            self.tracked_vehicles = {}
            self.plate_attempts = defaultdict(int)
            self.detected_plates = {}
            self.max_attempts = 3
            self.vehicle_tracking_timeout = 10
            self.last_vehicle_tracking_time = {}
            self.frame_buffer = {}
            self.max_buffer_size = 5
            
            # Add CPU optimization attributes
            self.processing_active = False  # Whether active detection is running
            self.last_activity_time = 0  # Last time activity was detected
            self.cooldown_period = 5  # Seconds to wait after no activity before stopping processing
            self.idle_timeout = 30  # Seconds without activity before resetting background model
            self.frame_skip = 10  # Process only every Nth frame when in idle mode
            self.frame_counter = 0  # Counter for frame skipping
            self.pixel_diff_threshold = 15  # Minimum threshold for pixel-based changes
            self.area_diff_threshold = 0.03  # Percentage of frame that needs to change
            self.background_model = None  # Background model for change detection
            self.background_update_rate = 0.01  # Background model learning rate
            self.roi_activity_only = True  # Only detect changes within ROI
            self.in_cooldown = False  # Whether in cooldown phase before stopping
            self.last_background_reset = 0  # Time when background model was last reset
            
            # State for UI
            if not hasattr(self, 'last_recognized_plate'):
                self.last_recognized_plate = None
                self.last_plate_authorized = False
            
        except Exception as e:
            logger.error(f"[TRACKER:{camera_id}] Error in VehicleTracker initialization: {str(e)}")
            raise

    def _load_roi_polygon(self, config_path):
        """Load ROI polygon from config file and scale it to match the current frame size."""
        if not config_path:
            return None
            
        try:
            if not os.path.exists(config_path):
                logger.warning(f"[TRACKER:{self.camera_id}] ROI config file not found: {config_path}")
                return None
                
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            # Check if this is the new format (dictionary with keys) or old format (plain array)
            if isinstance(config_data, dict) and "original_points" in config_data:
                # New format - store additional information for scaling
                self.roi_config = config_data
                self.original_dimensions = config_data.get("original_dimensions", None)
                self.display_dimensions = config_data.get("display_dimensions", None)
                self.scale_ratios = config_data.get("scale_ratios", None)
                
                # Verify camera ID matches if it exists in config
                if "camera_id" in config_data and config_data["camera_id"] != self.camera_id:
                    logger.warning(f"[TRACKER:{self.camera_id}] ROI config was created for camera '{config_data['camera_id']}', but being used with '{self.camera_id}'")
                
                # Use original points for ROI
                roi_points = config_data["original_points"]
                logger.info(f"[TRACKER:{self.camera_id}] Loaded ROI in new format from {config_path}")
            else:
                # Old format - just a list of points
                roi_points = config_data
                self.roi_config = None
                self.original_dimensions = None
                self.display_dimensions = None
                self.scale_ratios = None
                logger.info(f"[TRACKER:{self.camera_id}] Loaded ROI in old format from {config_path}")
                
            if not isinstance(roi_points, list) or len(roi_points) < 3:
                logger.warning(f"[TRACKER:{self.camera_id}] Invalid ROI points format in {config_path}: {roi_points}")
                return None
                
            # Convert to numpy array
            roi_polygon = np.array(roi_points, dtype=np.int32)
            logger.info(f"[TRACKER:{self.camera_id}] Loaded ROI from {config_path} with {len(roi_polygon)} points")
            return roi_polygon
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Error loading ROI polygon: {str(e)}")
        return None

    def _scale_roi_to_frame(self, frame):
        """Scale ROI to match current frame dimensions"""
        if self.original_roi is None:
            return self.original_roi
            
        try:
            # Get the current frame dimensions
            frame_h, frame_w = frame.shape[:2]
            
            # Store first frame as reference for scaling if not already stored
            if not hasattr(self, 'first_frame'):
                self.first_frame = frame.copy()
                self.first_frame_dims = (frame_w, frame_h)
                # Initialize last plate tracking
                if not hasattr(self, 'last_recognized_plate'):
                    self.last_recognized_plate = None
                    self.last_plate_authorized = False
                logger.info(f"[TRACKER:{self.camera_id}] Stored first frame with dimensions {frame_w}x{frame_h} as reference")
            
            # Create a copy of the original ROI
            scaled_roi = self.original_roi.copy()
            
            # If we have the original dimensions from config, use those for precise scaling
            if hasattr(self, 'original_dimensions') and self.original_dimensions:
                orig_width, orig_height = self.original_dimensions
                
                logger.debug(f"[TRACKER:{self.camera_id}] Using dimensions from config - Original: {orig_width}x{orig_height}, Frame: {frame_w}x{frame_h}")
                
                # Calculate direct scaling factors from original frame to current frame
                scale_x = frame_w / orig_width
                scale_y = frame_h / orig_height
                
                logger.debug(f"[TRACKER:{self.camera_id}] Using precise ROI scaling factors from config: {scale_x:.4f}x{scale_y:.4f}")
            else:
                # If frame dimensions match the first frame, no scaling needed
                if (frame_w, frame_h) == self.first_frame_dims:
                    return self.original_roi
                
                # Use first frame dimensions as reference point for scaling
                ref_width, ref_height = self.first_frame_dims
                
                # Log scaling operation
                logger.debug(f"[TRACKER:{self.camera_id}] Scaling ROI from {ref_width}x{ref_height} to {frame_w}x{frame_h}")
                
                # Calculate scale factors
                scale_x = frame_w / ref_width
                scale_y = frame_h / ref_height
                
                logger.debug(f"[TRACKER:{self.camera_id}] Using estimated ROI scaling factors: {scale_x:.4f}x{scale_y:.4f}")
            
            # Scale the ROI coordinates
            scaled_roi[:, 0] = (scaled_roi[:, 0] * scale_x).astype(np.int32)
            scaled_roi[:, 1] = (scaled_roi[:, 1] * scale_y).astype(np.int32)
            
            # Log only when dimension changes for debugging
            if not hasattr(self, 'prev_scale') or self.prev_scale != (scale_x, scale_y):
                logger.info(f"[TRACKER] ROI scale factors changed: {scale_x:.4f}x{scale_y:.4f}")
                self.prev_scale = (scale_x, scale_y)
            
            return scaled_roi
        except Exception as e:
            logger.error(f"Error scaling ROI: {str(e)}")
            return self.original_roi

    def _is_vehicle_in_roi(self, box):
        """Check if vehicle's center point is within ROI."""
        if self.roi_polygon is None:
            return True
            
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        with self.roi_lock:
            try:
                # Make sure we're using the properly scaled ROI from the current frame
                result = cv2.pointPolygonTest(self.roi_polygon, (center_x, center_y), False)
                return result >= 0
            except Exception as e:
                logger.error(f"ROI check error: {str(e)}")
                return True

    def visualize_detection(self, frame, boxes, track_ids, class_ids):
        """Draw detection boxes, IDs and plates on frame"""
        vis_frame = frame.copy()
        # Draw vehicle boxes and info
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if class_id not in VEHICLE_CLASSES:
                continue
                
            class_name = VEHICLE_CLASSES[class_id]
            x1, y1, x2, y2 = map(int, box)
            
            # Check if vehicle is in ROI
            is_in_roi = self._is_vehicle_in_roi(box)
            
            # Use different colors based on whether vehicle is in ROI
            box_color = (0, 255, 0) if is_in_roi else (0, 165, 255)  # Green if in ROI, orange if not
            
            # Draw box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw ID and class
            label = f"ID: {track_id} {class_name}"
            cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)  # White text
            
            # Draw plate if detected
            if track_id in self.detected_plates:
                plate_text = self.detected_plates[track_id]
                # Draw plate text with thicker font and brighter color for better visibility of Arabic characters
                cv2.putText(vis_frame, f"Plate: {plate_text}",
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 0), 3)  # Black outline
                cv2.putText(vis_frame, f"Plate: {plate_text}",
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2)  # Red text
                
                # Update last recognized plate
                self.last_recognized_plate = plate_text
                # Debug log to ensure we're seeing plate detections
                logger.debug(f"[TRACKER] Vehicle {track_id} has plate {plate_text}, updated last_recognized_plate")
        
        # Always draw the last plate info section, even if empty
        h, w = vis_frame.shape[:2]
        
        # Draw smaller background rectangle for better aesthetics
        cv2.rectangle(vis_frame, (10, h-70), (350, h-10), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (10, h-70), (350, h-10), (255, 255, 255), 2)
                
        return vis_frame

    def _update_vehicle_state(self, track_id, frame, box):
        """Update vehicle tracking state and frame buffer"""
        current_time = time.time()
        self.last_vehicle_tracking_time[track_id] = current_time
        
        # Add frame to buffer
        if track_id not in self.frame_buffer:
            self.frame_buffer[track_id] = []
        
        vehicle_img = self._extract_vehicle_image(frame, box)
        if vehicle_img is not None:
            # Calculate image clarity score - higher is better
            clarity_score = self._calculate_image_quality(vehicle_img)
            self.frame_buffer[track_id].append((vehicle_img, clarity_score))
            logger.debug(f"[TRACKER] Vehicle {track_id} frame quality: {clarity_score:.2f}")
            
            # Keep buffer at maximum size
            if len(self.frame_buffer[track_id]) > self.max_buffer_size:
                self.frame_buffer[track_id].pop(0)

    def _calculate_image_quality(self, image):
        """Calculate image quality/clarity score based on Laplacian variance"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Calculate Laplacian variance - measure of focus/clarity
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            score = laplacian.var()
            
            # Calculate histogram distribution - well-exposed images have good distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_std = np.std(hist)
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = np.std(gray)
            
            # Combined score (higher is better)
            combined_score = (score * 0.5) + (hist_std * 0.2) + (contrast * 0.3)
            
            # Penalize very dark or very bright images
            if brightness < 30 or brightness > 220:
                combined_score *= 0.7
                
            return combined_score
        except Exception as e:
            logger.error(f"[TRACKER] Error calculating image quality: {str(e)}")
            return 0

    def _cleanup_stale_vehicles(self):
        """Remove vehicles that haven't been seen recently"""
        current_time = time.time()
        stale_ids = []
        
        for track_id, last_time in self.last_vehicle_tracking_time.items():
            if current_time - last_time > self.vehicle_tracking_timeout:
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            self.last_vehicle_tracking_time.pop(track_id, None)
            self.frame_buffer.pop(track_id, None)
            logger.info(f"Vehicle {track_id} tracking timed out")

    def _should_process_frame(self, frame):
        """Determine if this frame should be processed based on activity detection"""
        current_time = time.time()
        self.frame_counter += 1
        
        # Initialize background model if needed
        if self.background_model is None:
            self.background_model = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
            self.last_background_reset = current_time
            logger.info("[TRACKER] Background model initialized")
            # Always process first frame
            self.processing_active = True
            return True, False
        
        # If we're in active processing mode
        if self.processing_active:
            # Check if we've timed out with no activity
            if current_time - self.last_activity_time > self.cooldown_period:
                if not self.in_cooldown:
                    logger.info("[TRACKER] No activity detected for cooldown period, entering cooldown")
                    self.in_cooldown = True
                
                # If cooldown is exceeded, stop active processing
                if current_time - self.last_activity_time > self.cooldown_period * 2:
                    logger.info("[TRACKER] Exiting active processing mode after cooldown")
                    self.processing_active = False
                    self.in_cooldown = False
            
            # In active mode, process frame (possible cooldown)
            return True, False
        
        # In idle mode, only process every Nth frame for motion detection
        if self.frame_counter % self.frame_skip != 0:
            return False, False
        
        # Apply foreground mask
        fgmask = self.background_model.apply(frame)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        # Apply ROI mask if needed
        if self.roi_activity_only and self.roi_polygon is not None:
            # Create a blank mask
            roi_mask = np.zeros_like(fgmask)
            # Fill the ROI polygon with white
            cv2.fillPoly(roi_mask, [self.roi_polygon], 255)
            # Apply ROI mask to the foreground mask
            fgmask = cv2.bitwise_and(fgmask, roi_mask)
        
        # Calculate the percentage of changed pixels
        total_pixels = fgmask.shape[0] * fgmask.shape[1]
        changed_pixels = cv2.countNonZero(fgmask)
        percent_changed = changed_pixels / total_pixels
        
        # Check if there's significant motion
        activity_detected = percent_changed > self.area_diff_threshold
        
        # Reset background model periodically or when needed
        if current_time - self.last_background_reset > self.idle_timeout:
            logger.info("[TRACKER] Resetting background model due to timeout")
            self.background_model = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
            self.last_background_reset = current_time
        
        # If activity is detected, switch to active processing
        if activity_detected:
            logger.info(f"[TRACKER] Activity detected ({percent_changed:.1%} of frame), activating processing")
            self.processing_active = True
            self.last_activity_time = current_time
            return True, True
        
        # No activity, stay in idle mode
        return False, False

    def detect_vehicles(self, frame):
        """Detect and track vehicles in frame, with CPU optimization using trigger system."""
        if frame is None:
            logger.warning("[TRACKER] Received None frame for detection")
            return False, None
            
        try:
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Scale ROI to match current frame dimensions
            self.roi_polygon = self._scale_roi_to_frame(frame)
            
            # Draw ROI first
            if self.roi_polygon is not None:
                # Draw ROI with a thicker line and brighter color
                cv2.polylines(vis_frame, [self.roi_polygon], True, (0, 255, 0), 3)  # Bright green, thick line
                
                # Fill the ROI with semi-transparent green
                overlay = vis_frame.copy()
                cv2.fillPoly(overlay, [self.roi_polygon], (0, 200, 0, 50))  # Semi-transparent green
                cv2.addWeighted(overlay, 0.15, vis_frame, 0.85, 0, vis_frame)  # Subtle transparency
                
                # Add a label at the first point of the ROI polygon
                roi_x, roi_y = self.roi_polygon[0]
                cv2.putText(vis_frame, "ROI", (roi_x, roi_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black outline
                cv2.putText(vis_frame, "ROI", (roi_x, roi_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)  # Green text
            
            # Check if we should process this frame
            should_process, activity_detected = self._should_process_frame(frame)
            
            # Display processing status
            status_text = "ACTIVE" if self.processing_active else "IDLE"
            status_color = (0, 255, 0) if self.processing_active else (0, 0, 255)
            cv2.putText(vis_frame, f"Processing: {status_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_frame, f"Processing: {status_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 1)  # Colored text
            
            # Only run detection if we should process this frame
            if should_process:
                # Run detection
                logger.debug("[TRACKER] Running YOLO detection and tracking")
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=0.3,
                    iou=0.45,
                    verbose=False
                )
                
                # Process detections if we have any
                if len(results) > 0 and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    try:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        class_ids = results[0].boxes.cls.int().cpu().tolist()
                        
                        logger.debug(f"[TRACKER] Detected {len(track_ids)} vehicles: {dict(zip(track_ids, [VEHICLE_CLASSES.get(c, 'unknown') for c in class_ids]))}")
                        
                        # Process each detected vehicle
                        vehicles_in_roi = 0
                        vehicles_processed_for_plates = 0
                        
                        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                            if class_id not in VEHICLE_CLASSES:
                                continue
                                
                            # Check if vehicle is in ROI
                            is_in_roi = self._is_vehicle_in_roi(box)
                            
                            # Update vehicle state for tracking
                            self._update_vehicle_state(track_id, frame, box)
                            
                            # Only process license plates for vehicles in ROI
                            if is_in_roi:
                                vehicles_in_roi += 1
                                # Keep processing active when vehicles are in ROI
                                self.last_activity_time = time.time()
                                
                                # Only try to detect license plate if not already detected for this vehicle
                                if track_id not in self.detected_plates and track_id in self.frame_buffer:
                                    # Process license plate if we have enough frames buffered
                                    if len(self.frame_buffer[track_id]) >= 3:
                                        vehicles_processed_for_plates += 1
                                        # Get best frame from buffer
                                        best_frame = self._select_best_frame(self.frame_buffer[track_id])
                                        if best_frame is not None:
                                            # Process the plate on the best quality frame
                                            logger.info(f"[TRACKER] Processing license plate for vehicle {track_id} using best quality frame")
                                            self._process_plate(best_frame, track_id)
                                        else:
                                            logger.warning(f"[TRACKER] Could not select best frame for vehicle {track_id}")
                        
                        if vehicles_in_roi > 0:
                            logger.debug(f"[TRACKER] {vehicles_in_roi} vehicles in ROI, {vehicles_processed_for_plates} processed for plates")
                        
                        # Draw all detections on the visualization frame
                        vis_frame = self.visualize_detection(vis_frame, boxes, track_ids, class_ids)
                    except Exception as e:
                        logger.error(f"[TRACKER] Error processing detection boxes: {str(e)}")
                        
                # Cleanup stale vehicles periodically
                self._cleanup_stale_vehicles()               
            
            # If activity was just detected, show an indicator
            if activity_detected:
                h, w = vis_frame.shape[:2]
                cv2.putText(vis_frame, "ACTIVITY DETECTED", (w//2-150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display the visualization frame in a camera-specific window
            window_name = f'Detections - {self.camera_id}'
            cv2.imshow(window_name, vis_frame)
            cv2.waitKey(1)
            
            # Return the visualization frame
            return True, vis_frame

        except Exception as e:
            logger.error(f"[TRACKER] Error in detect_vehicles: {str(e)}")
            return False, None

    def _extract_vehicle_image(self, frame, box):
        """Extract vehicle image from frame using bounding box"""
        try:
            x1, y1, x2, y2 = map(int, box)
            # Add padding around the vehicle
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            # Ensure we don't go out of bounds
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)
            # Extract the vehicle image
            vehicle_img = frame[y1:y2, x1:x2]
            return vehicle_img
        except Exception as e:
            logger.error(f"[TRACKER] Error extracting vehicle image: {str(e)}")
            return None

    def _process_plate(self, vehicle_img, track_id):
        """Process vehicle image to detect and recognize license plate"""
        try:
            if self.plate_attempts[track_id] >= self.max_attempts:
                return
                
            # Step 1: Find the best license plate in the vehicle image
            plate_img = self.plate_processor.find_best_plate_in_image(vehicle_img)
            if plate_img is None:
                self.plate_attempts[track_id] += 1
                return
                
            # Step 2: Recognize characters on the license plate
            plate_text = self.plate_processor.recognize_plate(plate_img)
            if plate_text and len(plate_text) >= 3:
                logger.info(f"[TRACKER] License plate recognized for vehicle {track_id}: {plate_text}")
                self.detected_plates[track_id] = plate_text
                return
                
            self.plate_attempts[track_id] += 1
        except Exception as e:
            logger.error(f"[TRACKER] Error in license plate processing: {str(e)}")
            self.plate_attempts[track_id] += 1

    def _select_best_frame(self, frames_with_scores):
        """Select the best frame from the buffer based on quality score"""
        if not frames_with_scores:
            return None
            
        try:
            # Sort by score (descending)
            sorted_frames = sorted(frames_with_scores, key=lambda x: x[1], reverse=True)
            # Return the frame with the highest score
            return sorted_frames[0][0]
        except Exception as e:
            logger.error(f"[TRACKER] Error selecting best frame: {str(e)}")
            # Return the last frame as fallback
            return frames_with_scores[-1][0]

    def get_detected_plate(self, track_id):
        """Get the detected plate for a specific vehicle ID"""
        return self.detected_plates.get(track_id, None)
