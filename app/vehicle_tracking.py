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
    def __init__(self, roi_config_path="config.json"):
        logger.info("Initializing vehicle detection model...")
        try:
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)
            
            # Use absolute path for model
            model_path = osp.abspath(YOLO_MODEL_PATH)
            logger.info(f"Loading YOLO model from: {model_path}")
            
            if not osp.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            try:
                self.model = YOLO(model_path)
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading YOLO model: {str(e)}")
                raise
            
            # Load ROI configuration
            self.original_roi = self._load_roi_polygon(roi_config_path)
            self.roi_polygon = self.original_roi  # Will be scaled per frame later
            
            if self.original_roi is not None:
                logger.info(f"ROI loaded with {len(self.original_roi)} points")
                logger.debug(f"Original ROI points: {self.original_roi.tolist()}")
            else:
                logger.warning("No ROI configuration found. Using full frame.")
                
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
            
        except Exception as e:
            logger.error(f"Error in VehicleTracker initialization: {str(e)}")
            raise

    def _load_roi_polygon(self, config_path):
        """Load ROI polygon from config file and scale it to match the current frame size."""
        if not config_path:
            return None
            
        try:
            if not os.path.exists(config_path):
                logger.warning(f"ROI config file not found: {config_path}")
                return None
                
            with open(config_path, "r") as f:
                config_data = json.load(f)
                
            if not isinstance(config_data, list) or len(config_data) < 3:
                logger.warning(f"Invalid ROI format in {config_path}: {config_data}")
                return None
                
            # Convert to numpy array
            roi_polygon = np.array(config_data, dtype=np.int32)
            logger.info(f"Loaded ROI from {config_path} with {len(roi_polygon)} points: {roi_polygon.tolist()}")
            return roi_polygon
        except Exception as e:
            logger.error(f"Error loading ROI polygon: {str(e)}")
        return None

    def _scale_roi_to_frame(self, frame):
        """Scale ROI to match current frame dimensions"""
        if self.original_roi is None or not hasattr(self, 'first_frame'):
            return self.original_roi
            
        try:
            # Get the current frame dimensions
            frame_h, frame_w = frame.shape[:2]
            # Get the original frame dimensions from roi.py
            original_w, original_h = 1920, 1080  # These are the dimensions from the CCTV camera
            
            logger.debug(f"Scaling ROI from {original_w}x{original_h} to {frame_w}x{frame_h}")
            
            # Calculate scale factors
            scale_x = frame_w / original_w
            scale_y = frame_h / original_h
            
            # Create a copy of the original ROI
            scaled_roi = self.original_roi.copy()
            
            # Scale the ROI coordinates
            scaled_roi[:, 0] = (scaled_roi[:, 0] * scale_x).astype(np.int32)
            scaled_roi[:, 1] = (scaled_roi[:, 1] * scale_y).astype(np.int32)
            
            logger.debug(f"Scaled ROI points: {scaled_roi.tolist()}")
            
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
        
        # We don't need to draw ROI here as it's already drawn in detect_vehicles
        # This avoids drawing the ROI twice with different styles

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
                cv2.putText(vis_frame, f"Plate: {plate_text}",
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 0, 255), 2)  # Red for plate text
                
                # Update last recognized plate
                self.last_recognized_plate = plate_text
                # Check authorization (this is simplified, replace with your actual authorization logic)
                self.last_plate_authorized = False  # Default to not authorized
        
        # Display the last recognized plate at the lower left corner
        if hasattr(self, 'last_recognized_plate') and self.last_recognized_plate is not None:
            h, w = vis_frame.shape[:2]
            auth_status = "Authorized" if self.last_plate_authorized else "Not Authorized"
            auth_color = (0, 255, 0) if self.last_plate_authorized else (0, 0, 255)  # Green or Red
            
            # Draw background rectangle for better visibility
            cv2.rectangle(vis_frame, (10, h-70), (400, h-10), (0, 0, 0), -1)
            cv2.rectangle(vis_frame, (10, h-70), (400, h-10), (255, 255, 255), 1)
            
            # Display last plate info
            cv2.putText(vis_frame, f"Last Plate: {self.last_recognized_plate}", 
                       (20, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Status: {auth_status}", 
                       (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, auth_color, 2)
        
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

    def detect_vehicles(self, frame):
        """Detect and track vehicles in frame."""
        if frame is None:
            logger.warning("[TRACKER] Received None frame for detection")
            return False, None
            
        try:
            # Store first frame for reference
            if not hasattr(self, 'first_frame'):
                self.first_frame = frame.copy()
                logger.info(f"Stored first frame with dimensions: {frame.shape[1]}x{frame.shape[0]}")
                
                # Initialize last plate tracking
                self.last_recognized_plate = None
                self.last_plate_authorized = False
            
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Scale ROI to match current frame dimensions
            self.roi_polygon = self._scale_roi_to_frame(frame)
            
            # Draw ROI first - make it more visible
            if self.roi_polygon is not None:
                # Draw ROI with a thicker line and brighter color
                cv2.polylines(vis_frame, [self.roi_polygon], True, (0, 255, 0), 4)  # Bright green, thick line
                
                # Fill the ROI with semi-transparent green
                overlay = vis_frame.copy()
                cv2.fillPoly(overlay, [self.roi_polygon], (0, 200, 0, 50))  # Semi-transparent green
                cv2.addWeighted(overlay, 0.2, vis_frame, 0.8, 0, vis_frame)
                
                # Add a label
                cv2.putText(vis_frame, "ROI", (self.roi_polygon[0][0], self.roi_polygon[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                logger.debug("[TRACKER] Drawing ROI polygon on visualization frame")
            else:
                logger.debug("[TRACKER] No ROI polygon defined")
            
            # Ensure frame dimensions are consistent for optical flow
            if hasattr(self, 'prev_frame_shape') and self.prev_frame_shape != frame.shape:
                logger.warning(f"[TRACKER] Frame size changed from {self.prev_frame_shape} to {frame.shape}. This may cause optical flow errors.")
            self.prev_frame_shape = frame.shape
            
            # Use the frame as is since it's already resized by the camera class
            detection_frame = frame
            
            # Run detection
            logger.debug("[TRACKER] Running YOLO detection and tracking")
            results = self.model.track(
                detection_frame,
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
                    
                    # Cleanup stale vehicles periodically
                    self._cleanup_stale_vehicles()
                    
                    if vehicles_in_roi > 0:
                        logger.debug(f"[TRACKER] {vehicles_in_roi} vehicles in ROI, {vehicles_processed_for_plates} processed for plates")
                    
                    # Draw all detections on the same frame
                    vis_frame = self.visualize_detection(vis_frame, boxes, track_ids, class_ids)
                except Exception as e:
                    logger.error(f"[TRACKER] Error processing detection boxes: {str(e)}")
            
            # Return visualization frame without showing it
            return True, vis_frame
                            
        except Exception as e:
            logger.error(f"[TRACKER] Error during vehicle detection: {str(e)}")
            # Still process the frame with ROI even if error occurs
            if frame is not None:
                try:
                    error_frame = frame.copy()
                    if self.roi_polygon is not None:
                        cv2.polylines(error_frame, [self.roi_polygon], True, (0, 255, 0), 3)
                    cv2.putText(error_frame, "Detection error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return False, error_frame
                except:
                    pass
            return False, frame

    def _extract_vehicle_image(self, frame, box):
        """Extract vehicle region from frame with padding."""
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            # Add padding around vehicle
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            return frame[y1:y2, x1:x2].copy()
        except Exception as e:
            logger.error(f"Error extracting vehicle image: {str(e)}")
            return None

    def _process_plate(self, vehicle_img, track_id):
        """Process vehicle image to detect and recognize license plate."""
        try:
            plate_text, processed_image = self.plate_processor.process_vehicle_image(vehicle_img)
            if plate_text:
                self.detected_plates[track_id] = plate_text
                logger.info(f"[TRACKER] Detected plate {plate_text} for vehicle {track_id}")
                
                # Update the last recognized plate for display
                self.last_recognized_plate = plate_text
                
                # Default to unauthorized - the actual check happens in app/__init__.py
                # when processing frame, and visualize_detection will be called again
                self.last_plate_authorized = False
                
                return True
            else:
                logger.debug(f"[TRACKER] No plate detected for vehicle {track_id}")
                return False
        except Exception as e:
            logger.error(f"[TRACKER] Error processing plate: {str(e)}")
            return False

    def _select_best_frame(self, frames_with_scores):
        """Select the best frame from the buffer based on clarity score"""
        if not frames_with_scores:
            return None
            
        # Sort by clarity score (descending)
        sorted_frames = sorted(frames_with_scores, key=lambda x: x[1], reverse=True)
        
        # Return the frame with the highest score
        best_frame, best_score = sorted_frames[0]
        logger.debug(f"[TRACKER] Selected best frame with quality score: {best_score:.2f}")
        
        return best_frame

    def get_detected_plate(self, track_id):
        """Get detected plate number for a vehicle if available."""
        return self.detected_plates.get(track_id)
