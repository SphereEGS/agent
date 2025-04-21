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
from .roi_manager import RoiManager

# Vehicle classes from COCO dataset that we want to detect
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

class VehicleTracker:
    def __init__(self, roi_config_path="config.json", trigger_roi_config_path="trigger_roi.json", camera_id="main"):
        logger.info(f"[TRACKER:{camera_id}] Initializing vehicle detection model...")
        self.camera_id = camera_id
        
        try:
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
            
            # Initialize ROI manager
            self.roi_manager = RoiManager(
                camera_id=camera_id,
                roi_config_path=roi_config_path,
                trigger_roi_config_path=trigger_roi_config_path
            )
                
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
            
            # Variables for tracking detection stats
            self.frames_processed = 0
            self.frames_with_detection = 0
            self.frames_skipped = 0
            self.last_recognized_plate = None
            self.last_plate_authorized = False
            
        except Exception as e:
            logger.error(f"[TRACKER:{camera_id}] Error in VehicleTracker initialization: {str(e)}")
            raise

    def _is_vehicle_in_roi(self, box):
        """Check if vehicle's center point is within ROI."""
        with self.roi_lock:
            try:
                return self.roi_manager.is_vehicle_in_roi(box, self.roi_manager.detection_roi_polygon)
            except Exception as e:
                logger.error(f"[TRACKER:{self.camera_id}] ROI check error: {str(e)}")
                return True

    def visualize_detection(self, frame, boxes, track_ids, class_ids):
        """Draw detection boxes, IDs and plates on frame"""
        # First visualize all ROIs
        vis_frame = self.roi_manager.visualize_rois(frame)
        
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
                logger.info(f"[TRACKER:{self.camera_id}] Vehicle {track_id} has plate {plate_text}, updated last_recognized_plate")
                # Check authorization (this is simplified, replace with your actual authorization logic)
                self.last_plate_authorized = False  # Default to not authorized
        
        # Always draw the last plate info section, even if empty
        h, w = vis_frame.shape[:2]
        
        # Draw smaller background rectangle for better aesthetics
        cv2.rectangle(vis_frame, (10, h-70), (350, h-10), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (10, h-70), (350, h-10), (255, 255, 255), 2)
        
        # Draw detection efficiency statistics
        detection_status = "ACTIVE" if self.roi_manager.detection_active else "Standby"
        efficiency_text = f"Detection: {detection_status} | Cam: {self.camera_id}"
        cv2.putText(vis_frame, efficiency_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black outline
        cv2.putText(vis_frame, efficiency_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # White text
        
        # Display last plate info or "No plate detected" message
        if hasattr(self, 'last_recognized_plate') and self.last_recognized_plate is not None:
            plate_text = self.last_recognized_plate
            auth_status = "Authorized" if self.last_plate_authorized else "Not Authorized"
            auth_color = (0, 255, 0) if self.last_plate_authorized else (0, 0, 255)  # Green or Red
            
            # Log that we're displaying the last recognized plate
            logger.debug(f"[TRACKER:{self.camera_id}] Displaying last plate: {plate_text}, status: {auth_status}")
        else:
            # Show default message when no plate is detected
            plate_text = "No plate detected"
            auth_status = "N/A"
            auth_color = (128, 128, 128)  # Gray for N/A
            
        # Always display plate info with enhanced visibility but smaller font
        # Add black outline first for better visibility
        cv2.putText(vis_frame, f"Last Plate: {plate_text}", 
                   (20, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # Then add white text
        cv2.putText(vis_frame, f"Last Plate: {plate_text}", 
                   (20, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Draw status with outline for better visibility
        cv2.putText(vis_frame, f"Status: {auth_status}", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black outline
        cv2.putText(vis_frame, f"Status: {auth_status}", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, auth_color, 1)  # Colored text
        
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
            logger.debug(f"[TRACKER:{self.camera_id}] Vehicle {track_id} frame quality: {clarity_score:.2f}")
            
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
            logger.error(f"[TRACKER:{self.camera_id}] Error calculating image quality: {str(e)}")
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
            logger.info(f"[TRACKER:{self.camera_id}] Vehicle {track_id} tracking timed out")

    def detect_vehicles(self, frame):
        """Detect and track vehicles in frame."""
        if frame is None:
            logger.warning(f"[TRACKER:{self.camera_id}] Received None frame for detection")
            return False, None
            
        try:
            self.frames_processed += 1
            
            # First check if we should run detection based on motion in trigger ROI
            should_run_detection = self.roi_manager.should_process_detection(frame)
            
            # Create visualization frame with ROIs, regardless of whether we detect vehicles
            vis_frame = self.roi_manager.visualize_rois(frame.copy())
            
            # If we should not run detection, return early with just the ROI visualization
            if not should_run_detection:
                self.frames_skipped += 1
                if self.frames_skipped % 50 == 0:  # Log periodically to avoid spam
                    logger.debug(f"[TRACKER:{self.camera_id}] Skipping detection, no motion in trigger ROI ({self.frames_skipped} frames skipped)")
                return False, vis_frame
            
            # Run detection since motion was detected in trigger ROI
            logger.debug(f"[TRACKER:{self.camera_id}] Running YOLO detection and tracking")
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
                    
                    logger.debug(f"[TRACKER:{self.camera_id}] Detected {len(track_ids)} vehicles: {dict(zip(track_ids, [VEHICLE_CLASSES.get(c, 'unknown') for c in class_ids]))}")
                    
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
                                        logger.info(f"[TRACKER:{self.camera_id}] Processing license plate for vehicle {track_id} using best quality frame")
                                        self._process_plate(best_frame, track_id)
                                    else:
                                        logger.warning(f"[TRACKER:{self.camera_id}] Could not select best frame for vehicle {track_id}")
                    
                    # Cleanup stale vehicles periodically
                    self._cleanup_stale_vehicles()
                    
                    if vehicles_in_roi > 0:
                        self.frames_with_detection += 1
                        logger.debug(f"[TRACKER:{self.camera_id}] {vehicles_in_roi} vehicles in ROI, {vehicles_processed_for_plates} processed for plates")
                    
                    # Draw all detections on the same frame
                    vis_frame = self.visualize_detection(frame, boxes, track_ids, class_ids)
                except Exception as e:
                    logger.error(f"[TRACKER:{self.camera_id}] Error processing detection boxes: {str(e)}")
            
            # Return the visualization frame and detection flag
            return True, vis_frame
                            
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Error during vehicle detection: {str(e)}")
            # Still return the frame with ROI and display it
            if frame is not None:
                try:
                    error_frame = frame.copy()
                    cv2.putText(error_frame, f"Detection error: {str(e)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
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
            logger.error(f"[TRACKER:{self.camera_id}] Error extracting vehicle image: {str(e)}")
            return None

    def _process_plate(self, vehicle_img, track_id):
        """Process vehicle image to detect and recognize license plate."""
        try:
            plate_text, processed_image = self.plate_processor.process_vehicle_image(vehicle_img)
            if plate_text:
                self.detected_plates[track_id] = plate_text
                logger.info(f"[TRACKER:{self.camera_id}] Detected plate {plate_text} for vehicle {track_id}")
                
                # Update the last recognized plate for display
                self.last_recognized_plate = plate_text
                
                # Default to unauthorized - the actual check happens in app/__init__.py
                # when processing frame, and visualize_detection will be called again
                self.last_plate_authorized = False
                
                return True
            else:
                logger.debug(f"[TRACKER:{self.camera_id}] No plate detected for vehicle {track_id}")
                return False
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Error processing plate: {str(e)}")
            return False

    def _select_best_frame(self, frames_with_scores):
        """Select the best frame from the buffer based on clarity score"""
        if not frames_with_scores:
            return None
            
        # Sort by clarity score (descending)
        sorted_frames = sorted(frames_with_scores, key=lambda x: x[1], reverse=True)
        
        # Return the frame with the highest score
        best_frame, best_score = sorted_frames[0]
        logger.debug(f"[TRACKER:{self.camera_id}] Selected best frame with quality score: {best_score:.2f}")
        
        return best_frame

    def get_detected_plate(self, track_id):
        """Get detected plate number for a vehicle if available."""
        return self.detected_plates.get(track_id)

    def get_efficiency_stats(self):
        """Return detection efficiency statistics."""
        if self.frames_processed == 0:
            return 0, 0
            
        detection_rate = (self.frames_with_detection / self.frames_processed) * 100
        skip_rate = (self.frames_skipped / self.frames_processed) * 100
        return detection_rate, skip_rate