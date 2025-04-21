import cv2
import numpy as np
import json
import os
import time
from .config import logger, DEFAULT_ROI_CONFIG_PATH, DEFAULT_TRIGGER_ROI_CONFIG_PATH, ENABLE_TRIGGER_ROI, MOTION_THRESHOLD, TRIGGER_PATIENCE

class RoiManager:
    """
    Manages multiple Regions of Interest (ROI) for vehicle detection
    - Trigger ROI: Used to detect motion and activate full detection pipeline
    - Detection ROI: Used for actual license plate recognition
    """
    def __init__(self, camera_id="main", roi_config_path=None, trigger_roi_config_path=None):
        self.camera_id = camera_id
        # Use provided paths or defaults with camera id prefixes
        self.roi_config_path = roi_config_path or f"{camera_id}_{DEFAULT_ROI_CONFIG_PATH}"
        # If the camera-specific file doesn't exist, fall back to the default
        if not os.path.exists(self.roi_config_path):
            self.roi_config_path = DEFAULT_ROI_CONFIG_PATH
            
        self.trigger_roi_config_path = trigger_roi_config_path or f"{camera_id}_{DEFAULT_TRIGGER_ROI_CONFIG_PATH}"
        # If the camera-specific trigger file doesn't exist, fall back to the default
        if not os.path.exists(self.trigger_roi_config_path):
            self.trigger_roi_config_path = DEFAULT_TRIGGER_ROI_CONFIG_PATH
        
        # ROI polygons and configs
        self.detection_roi_polygon = self._load_roi_polygon(self.roi_config_path)
        self.trigger_roi_polygon = self._load_roi_polygon(self.trigger_roi_config_path)
        
        # Motion detection variables
        self.prev_frame = None
        self.motion_detected = False
        self.last_trigger_time = 0
        self.detection_active = False
        self.trigger_countdown = 0
        self.enable_trigger = ENABLE_TRIGGER_ROI
        self.motion_threshold = MOTION_THRESHOLD
        self.trigger_patience = TRIGGER_PATIENCE
        
        # Log configuration
        logger.info(f"[ROI:{camera_id}] Initialized with detection ROI from {self.roi_config_path}")
        if self.detection_roi_polygon is not None:
            logger.info(f"[ROI:{camera_id}] Detection ROI loaded with {len(self.detection_roi_polygon)} points")
        else:
            logger.warning(f"[ROI:{camera_id}] No detection ROI defined, will use full frame")
            
        if self.enable_trigger:
            logger.info(f"[ROI:{camera_id}] Trigger ROI loaded from {self.trigger_roi_config_path}")
            if self.trigger_roi_polygon is not None:
                logger.info(f"[ROI:{camera_id}] Trigger ROI loaded with {len(self.trigger_roi_polygon)} points")
            else:
                logger.warning(f"[ROI:{camera_id}] No trigger ROI defined, will use full frame triggering")
        else:
            logger.info(f"[ROI:{camera_id}] Trigger ROI disabled, detection always active")
            self.detection_active = True

    def _load_roi_polygon(self, config_path):
        """Load ROI polygon from config file."""
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"[ROI:{self.camera_id}] ROI config file not found: {config_path}")
            return None
            
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            # Check if this is the new format (dictionary with keys) or old format (plain array)
            if isinstance(config_data, dict) and "original_points" in config_data:
                # New format - store additional information for scaling
                roi_points = config_data["original_points"]
                self.original_dimensions = config_data.get("original_dimensions", None)
                self.display_dimensions = config_data.get("display_dimensions", None)
                self.scale_ratios = config_data.get("scale_ratios", None)
                
                logger.info(f"[ROI:{self.camera_id}] Loaded ROI in new format from {config_path}")
            else:
                # Old format - just a list of points
                roi_points = config_data
                self.original_dimensions = None
                self.display_dimensions = None
                self.scale_ratios = None
                logger.info(f"[ROI:{self.camera_id}] Loaded ROI in old format from {config_path}")
                
            if not isinstance(roi_points, list) or len(roi_points) < 3:
                logger.warning(f"[ROI:{self.camera_id}] Invalid ROI points format in {config_path}: {roi_points}")
                return None
                
            # Convert to numpy array
            roi_polygon = np.array(roi_points, dtype=np.int32)
            return roi_polygon
        except Exception as e:
            logger.error(f"[ROI:{self.camera_id}] Error loading ROI polygon: {str(e)}")
        return None

    def scale_roi_to_frame(self, frame, roi_polygon):
        """Scale ROI to match current frame dimensions"""
        if roi_polygon is None:
            return None
            
        try:
            # Get the current frame dimensions
            frame_h, frame_w = frame.shape[:2]
            
            # Store first frame as reference for scaling if not already stored
            if not hasattr(self, 'first_frame_dims'):
                self.first_frame_dims = (frame_w, frame_h)
                logger.info(f"[ROI:{self.camera_id}] Stored first frame dimensions {frame_w}x{frame_h} as reference")
            
            # Create a copy of the original ROI
            scaled_roi = roi_polygon.copy()
            
            # If we have the original dimensions from config, use those for precise scaling
            if hasattr(self, 'original_dimensions') and self.original_dimensions:
                orig_width, orig_height = self.original_dimensions
                
                # Calculate direct scaling factors from original frame to current frame
                scale_x = frame_w / orig_width
                scale_y = frame_h / orig_height
            else:
                # If frame dimensions match the first frame, no scaling needed
                if (frame_w, frame_h) == self.first_frame_dims:
                    return roi_polygon
                
                # Use first frame dimensions as reference point for scaling
                ref_width, ref_height = self.first_frame_dims
                
                # Calculate scale factors
                scale_x = frame_w / ref_width
                scale_y = frame_h / ref_height
            
            # Scale the ROI coordinates
            scaled_roi[:, 0] = (scaled_roi[:, 0] * scale_x).astype(np.int32)
            scaled_roi[:, 1] = (scaled_roi[:, 1] * scale_y).astype(np.int32)
            
            return scaled_roi
        except Exception as e:
            logger.error(f"[ROI:{self.camera_id}] Error scaling ROI: {str(e)}")
            return roi_polygon

    def is_point_in_roi(self, point, roi_polygon):
        """Check if a point is within an ROI polygon."""
        if roi_polygon is None:
            return True  # If no ROI defined, consider everything inside
            
        try:
            result = cv2.pointPolygonTest(roi_polygon, point, False)
            return result >= 0
        except Exception as e:
            logger.error(f"[ROI:{self.camera_id}] ROI check error: {str(e)}")
            return True

    def is_vehicle_in_roi(self, box, roi_polygon):
        """Check if vehicle's center point is within ROI."""
        if roi_polygon is None:
            return True
            
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        return self.is_point_in_roi((center_x, center_y), roi_polygon)

    def detect_motion_in_trigger_roi(self, frame):
        """
        Detect motion within the trigger ROI to activate full detection pipeline.
        Returns True if motion detected, False otherwise.
        """
        if not self.enable_trigger:
            return True  # If trigger ROI is disabled, always return True
            
        # Scale trigger ROI to match current frame dimensions
        scaled_trigger_roi = self.scale_roi_to_frame(frame, self.trigger_roi_polygon)
        
        # If no trigger ROI defined, use a simplified motion detection across the whole frame
        if scaled_trigger_roi is None:
            return self._detect_motion_simple(frame)
            
        try:
            # Convert frame to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Initialize previous frame if None
            if self.prev_frame is None:
                self.prev_frame = gray
                return False
                
            # Compute absolute difference between current and previous frame
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate the thresholded image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Create a mask from the trigger ROI
            mask = np.zeros_like(thresh)
            cv2.fillPoly(mask, [scaled_trigger_roi], 255)
            
            # Apply mask to threshold image
            thresh_roi = cv2.bitwise_and(thresh, mask)
            
            # Find contours in the masked threshold image
            contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if any contour is large enough to be considered motion
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > self.motion_threshold:
                    motion_detected = True
                    break
                    
            # Update previous frame
            self.prev_frame = gray
            
            # If motion detected, update last trigger time
            if motion_detected:
                self.last_trigger_time = time.time()
                self.trigger_countdown = self.trigger_patience
                if not self.detection_active:
                    logger.info(f"[ROI:{self.camera_id}] Motion detected in trigger ROI, activating detection")
                self.detection_active = True
            elif self.detection_active:
                # Decrement countdown if detection is active but no motion detected
                self.trigger_countdown -= 1
                if self.trigger_countdown <= 0:
                    logger.info(f"[ROI:{self.camera_id}] No motion detected for {self.trigger_patience} frames, deactivating detection")
                    self.detection_active = False
            
            return motion_detected
        except Exception as e:
            logger.error(f"[ROI:{self.camera_id}] Error in motion detection: {str(e)}")
            # Return True on error to ensure we don't miss vehicles
            return True

    def _detect_motion_simple(self, frame):
        """Simplified motion detection for when no trigger ROI is defined."""
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Initialize previous frame if None
            if self.prev_frame is None:
                self.prev_frame = gray
                return False
                
            # Compute absolute difference between current and previous frame
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Calculate the amount of motion (white pixels)
            motion_pixels = np.sum(thresh == 255)
            motion_percentage = motion_pixels / (thresh.shape[0] * thresh.shape[1])
            
            # Update previous frame
            self.prev_frame = gray
            
            # Consider it motion if the percentage exceeds a threshold
            motion_detected = motion_percentage > 0.01  # 1% of the frame changes
            
            if motion_detected:
                self.last_trigger_time = time.time()
                self.trigger_countdown = self.trigger_patience
                if not self.detection_active:
                    logger.info(f"[ROI:{self.camera_id}] Motion detected in frame, activating detection")
                self.detection_active = True
            elif self.detection_active:
                # Decrement countdown if detection is active but no motion detected
                self.trigger_countdown -= 1
                if self.trigger_countdown <= 0:
                    logger.info(f"[ROI:{self.camera_id}] No motion detected for {self.trigger_patience} frames, deactivating detection")
                    self.detection_active = False
            
            return motion_detected
        except Exception as e:
            logger.error(f"[ROI:{self.camera_id}] Error in simple motion detection: {str(e)}")
            # Return True on error to ensure we don't miss vehicles
            return True

    def visualize_rois(self, frame):
        """Draw both ROIs on the frame for visualization."""
        if frame is None:
            return frame
            
        vis_frame = frame.copy()
        
        # Scale ROIs to match current frame dimensions
        scaled_detection_roi = self.scale_roi_to_frame(frame, self.detection_roi_polygon)
        scaled_trigger_roi = self.scale_roi_to_frame(frame, self.trigger_roi_polygon)
        
        # Draw trigger ROI if enabled
        if self.enable_trigger and scaled_trigger_roi is not None:
            # Draw with a different color to distinguish from detection ROI
            color = (0, 165, 255) if self.detection_active else (0, 0, 255)  # Orange if active, red if inactive
            cv2.polylines(vis_frame, [scaled_trigger_roi], True, color, 5)  # Thicker line (5 pixels)
            
            # Fill with semi-transparent color
            overlay = vis_frame.copy()
            fill_color = (0, 165, 255) if self.detection_active else (0, 0, 255)  # Orange or red
            cv2.fillPoly(overlay, [scaled_trigger_roi], fill_color)
            # More visible transparency (30%)
            cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
            
            # Add a more visible label
            status = "ACTIVE" if self.detection_active else "Standby"
            label_position = scaled_trigger_roi[0]
            cv2.putText(vis_frame, f"Trigger Zone ({status})", 
                       (label_position[0], label_position[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)  # Thicker black outline (4 pixels)
            cv2.putText(vis_frame, f"Trigger Zone ({status})", 
                       (label_position[0], label_position[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Thicker colored text (2 pixels)
        
        # Draw detection ROI
        if scaled_detection_roi is not None:
            # Draw ROI with a thicker line and brighter color
            cv2.polylines(vis_frame, [scaled_detection_roi], True, (0, 255, 0), 5)  # Bright green, thicker line
            
            # Fill the ROI with semi-transparent green
            overlay = vis_frame.copy()
            cv2.fillPoly(overlay, [scaled_detection_roi], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)  # More visible transparency
            
            # Add a more visible label
            label_position = scaled_detection_roi[0]
            cv2.putText(vis_frame, "LPR Zone", 
                       (label_position[0], label_position[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)  # Thicker black outline
            cv2.putText(vis_frame, "LPR Zone", 
                       (label_position[0], label_position[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Thicker green text
        
        # Add camera ID to top-left corner
        cv2.putText(vis_frame, f"Camera: {self.camera_id}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)  # Black outline
        cv2.putText(vis_frame, f"Camera: {self.camera_id}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # White text
        
        return vis_frame

    def should_process_detection(self, frame):
        """
        Determine if we should run the full vehicle detection pipeline on this frame
        based on motion in the trigger ROI.
        """
        # First check if trigger ROI is enabled
        if not self.enable_trigger:
            return True  # If disabled, always run detection
        
        # Check if there's motion in the trigger ROI
        motion_detected = self.detect_motion_in_trigger_roi(frame)
        
        # We should run detection if it's already active (due to previous motion)
        return self.detection_active 