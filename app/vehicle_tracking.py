import cv2
import numpy as np
import time
from collections import defaultdict
import torch
from ultralytics import YOLO
import os

from .config import logger, YOLO_MODEL_PATH, DETECTION_CONF, DETECTION_IOU
from .roi_manager import ROIManager
from .lpr_model import PlateProcessor

class VehicleTracker:
    """
    Tracks vehicles and performs license plate recognition
    """
    
    def __init__(self, camera_id="main"):
        """
        Initialize the vehicle tracker
        
        Args:
            camera_id: Camera identifier
        """
        self.camera_id = camera_id
        self.model = None
        self.roi_manager = ROIManager(camera_id)
        self.plate_processor = PlateProcessor()
        
        # Detection parameters
        self.conf_threshold = DETECTION_CONF
        self.iou_threshold = DETECTION_IOU
        
        # Vehicle tracking
        self.vehicles = {}  # track_id -> vehicle information
        self.detection_active = False
        self.detection_start_time = 0
        self.detection_duration = 5  # How long to keep detection active after no vehicles in detection zone
        
        # Performance metrics
        self.last_detection_time = 0
        self.detection_count = 0
        
        # Plate recognition results
        self.detected_plates = {}  # track_id -> plate text
        self.last_recognized_plate = None
        self.last_plate_authorized = False
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load YOLO detection model"""
        try:
            logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] Loading YOLO model from {YOLO_MODEL_PATH}")
            
            # Check if the model file exists
            if not os.path.exists(YOLO_MODEL_PATH):
                logger.error(f"[VEHICLE_TRACKER:{self.camera_id}] YOLO model file not found: {YOLO_MODEL_PATH}")
                self.model = None
                return
            
            # Load the model
            self.model = YOLO(YOLO_MODEL_PATH)
            
            # Log model information
            logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] YOLO model loaded successfully")
            logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] Model type: {type(self.model).__name__}")
            logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] Detection confidence: {self.conf_threshold}")
            logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] IOU threshold: {self.iou_threshold}")
        except Exception as e:
            logger.error(f"[VEHICLE_TRACKER:{self.camera_id}] Error loading YOLO model: {str(e)}")
            import traceback
            logger.error(f"[VEHICLE_TRACKER:{self.camera_id}] {traceback.format_exc()}")
            self.model = None
    
    def check_detection_zone(self, frame):
        """
        Check if there's any activity in the detection zone using simple motion detection
        
        Args:
            frame: Current frame to check
        
        Returns:
            bool: True if activity detected, False otherwise
        """
        # Skip if we don't have a detection ROI configured
        if self.roi_manager.detection_roi_polygon is None:
            return True  # Default to active if no detection ROI is set
        
        # Create a mask for the detection ROI
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_manager.detection_roi_polygon], 255)
        
        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Store the frame for next comparison if this is the first frame
        if not hasattr(self, 'previous_gray'):
            self.previous_gray = gray
            return False
        
        # Compute the absolute difference between the current frame and previous frame
        frame_delta = cv2.absdiff(self.previous_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours on thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contours are large enough to be vehicles
        activity_detected = False
        min_area = 500  # Minimum area to consider as a potential vehicle
        
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                activity_detected = True
                break
        
        # Update previous frame
        self.previous_gray = gray
        
        return activity_detected
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in the frame and track them
        
        Args:
            frame: Current frame
        
        Returns:
            tuple: (detection_performed, visualization_frame)
        """
        if frame is None:
            return False, None
            
        if self.model is None:
            logger.error(f"[VEHICLE_TRACKER:{self.camera_id}] YOLO model not loaded")
            return False, frame
        
        # Log frame dimensions periodically for debugging
        if self.detection_count % 100 == 0:
            height, width = frame.shape[:2]
            logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] Frame dimensions: {width}x{height}")
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Check if there's activity in the detection zone
        # activity_detected = self.check_detection_zone(frame)
        activity_detected = True  # Always set to True for testing
        
        # Update detection state
        current_time = time.time()
        
        if activity_detected:
            # Activity detected in the zone, activate detection
            self.detection_active = True
            self.detection_start_time = current_time
        elif self.detection_active:
            # No activity, but detection was active - check if we should deactivate
            if current_time - self.detection_start_time > self.detection_duration:
                logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] No activity for {self.detection_duration}s, deactivating detection")
                self.detection_active = False
        
        # Skip detection if not active - TEMPORARILY DISABLED FOR TESTING
        # if not self.detection_active:
        #     # Just draw ROIs on the frame and return
        #     vis_frame = self.roi_manager.draw_roi(vis_frame, roi_type="both")
        #     return False, vis_frame
        
        # Run detection model
        try:
            self.detection_count += 1
            detection_start = time.time()
            
            # Log detection attempt periodically
            if self.detection_count % 10 == 0:
                logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] Running detection #{self.detection_count}, frame size: {frame.shape[1]}x{frame.shape[0]}")
            
            # Perform detection
            results = self.model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
                tracker="bytetrack.yaml",
                persist=True
            )
            
            detection_time = time.time() - detection_start
            self.last_detection_time = detection_time
            
            # Log detection performance periodically
            if self.detection_count % 10 == 0:
                logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] Detection took {detection_time:.3f}s")
            
            # No detections
            if len(results) == 0 or not hasattr(results[0], 'boxes'):
                # Log when no detections occur periodically
                if self.detection_count % 50 == 0:
                    logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] No detections in frame")
                    
                # Just draw ROIs on the frame and return
                vis_frame = self.roi_manager.draw_roi(vis_frame, roi_type="both")
                return False, vis_frame
                
            # Process detection results
            boxes = results[0].boxes
            vehicle_detected = False
            
            # Log number of detections periodically
            if self.detection_count % 10 == 0:
                logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] Detected {len(boxes)} object(s)")
            
            # Draw ROIs on the visualization frame
            vis_frame = self.roi_manager.draw_roi(vis_frame, roi_type="both")
            
            # Process each bounding box
            vehicles_in_roi = 0
            vehicles_total = 0
            
            for box in boxes:
                vehicles_total += 1
                
                # Get tracking ID if available
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id.item())
                else:
                    # Skip boxes without tracking ID
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                
                # Get confidence score
                conf = float(box.conf[0].item()) if hasattr(box, 'conf') else 0.0
                
                # Get class ID and name
                cls_id = int(box.cls[0].item()) if hasattr(box, 'cls') else -1
                cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else "unknown"
                
                # Log detection details periodically
                if self.detection_count % 50 == 0:
                    logger.debug(f"[VEHICLE_TRACKER:{self.camera_id}] Detected {cls_name} (ID: {track_id}) at {x1},{y1},{x2},{y2} with confidence {conf:.2f}")
                
                # Check if the vehicle is in the detection ROI
                in_detection_roi = True  # Temporarily set to True for testing
                # in_detection_roi = self.roi_manager.is_object_in_roi(
                #     [x1, y1, x2, y2], roi_type="detection"
                # )
                
                # Check if the vehicle is in the LPR ROI
                in_lpr_roi = self.roi_manager.is_object_in_roi(
                    [x1, y1, x2, y2], roi_type="lpr"
                )
                
                # Only track if in detection ROI
                if in_detection_roi:
                    vehicles_in_roi += 1
                    vehicle_detected = True
                    
                    # Update or add vehicle to tracking dictionary
                    if track_id not in self.vehicles:
                        self.vehicles[track_id] = {
                            'first_seen': current_time,
                            'last_seen': current_time,
                            'bbox': [x1, y1, x2, y2],
                            'in_lpr_roi': in_lpr_roi,
                            'plate_processed': False,
                            'class': cls_name,
                            'confidence': conf
                        }
                    else:
                        self.vehicles[track_id]['last_seen'] = current_time
                        self.vehicles[track_id]['bbox'] = [x1, y1, x2, y2]
                        self.vehicles[track_id]['in_lpr_roi'] = in_lpr_roi
                        self.vehicles[track_id]['class'] = cls_name
                        self.vehicles[track_id]['confidence'] = conf
                    
                    # Set color based on detection ROI status
                    color = (0, 255, 0)  # Green for tracked vehicles
                    
                    # Only attempt LPR when in LPR ROI and not processed before
                    if in_lpr_roi and not self.vehicles[track_id].get('plate_processed', False):
                        # Extract vehicle image
                        vehicle_img = frame[y1:y2, x1:x2]
                        
                        # Process license plate
                        success, plate_text = self.plate_processor.process_image(vehicle_img)
                        
                        if success and plate_text:
                            # Mark as processed
                            self.vehicles[track_id]['plate_processed'] = True
                            self.detected_plates[track_id] = plate_text
                            
                            # Log plate detection
                            logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] Detected plate: {plate_text} on vehicle {track_id} ({cls_name})")
                            
                            # Draw license plate text on the vehicle
                            cv2.putText(vis_frame, f"Plate: {plate_text}",
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, (255, 255, 0), 2)
                            
                            # Update the visualization color
                            color = (0, 255, 255)  # Yellow for vehicles with plates
                    
                    # Draw bounding box for tracked vehicles
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add tracking ID
                    cv2.putText(vis_frame, f"ID: {track_id} ({cls_name})",
                              (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, 2)
            
            # Log summary of detected vehicles periodically
            if self.detection_count % 50 == 0 and vehicles_total > 0:
                logger.info(f"[VEHICLE_TRACKER:{self.camera_id}] Detection summary: {vehicles_in_roi}/{vehicles_total} vehicles in ROI")
            
            # Clean up vehicles that haven't been seen for a while
            current_time = time.time()
            ids_to_remove = []
            
            for track_id, vehicle_info in self.vehicles.items():
                if current_time - vehicle_info['last_seen'] > 5.0:  # 5 seconds timeout
                    ids_to_remove.append(track_id)
            
            for track_id in ids_to_remove:
                del self.vehicles[track_id]
                if track_id in self.detected_plates:
                    del self.detected_plates[track_id]
            
            return vehicle_detected, vis_frame
            
        except Exception as e:
            logger.error(f"[VEHICLE_TRACKER:{self.camera_id}] Error during vehicle detection: {str(e)}")
            import traceback
            logger.error(f"[VEHICLE_TRACKER:{self.camera_id}] {traceback.format_exc()}")
            return False, vis_frame