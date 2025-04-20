import cv2
import json
import os
import numpy as np
from .config import logger

class ROIManager:
    """
    Manages multiple ROIs for vehicle detection and license plate recognition
    """
    
    def __init__(self, camera_id="main"):
        """
        Initialize ROI manager for a specific camera
        
        Args:
            camera_id: Camera identifier for which this ROI applies
        """
        self.camera_id = camera_id
        self.config_file = f"config_{camera_id}.json" if camera_id != "main" else "config.json"
        
        # ROI polygons
        self.lpr_roi_polygon = None  # License plate recognition ROI
        self.detection_roi_polygon = None  # Vehicle detection trigger ROI
        
        # Load ROIs from config file
        self.load_roi_config()
    
    def load_roi_config(self):
        """Load ROI configuration from file"""
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"[ROI:{self.camera_id}] ROI config file not found: {self.config_file}")
                return False
                
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                
            # Get the LPR ROI
            if "lpr_roi" in config_data:
                points = config_data["lpr_roi"]["original_points"]
                self.lpr_roi_polygon = np.array(points, np.int32).reshape((-1, 1, 2))
                logger.info(f"[ROI:{self.camera_id}] Loaded LPR ROI with {len(points)} points")
            else:
                # Legacy format support
                points = config_data.get("original_points", [])
                if points:
                    self.lpr_roi_polygon = np.array(points, np.int32).reshape((-1, 1, 2))
                    logger.info(f"[ROI:{self.camera_id}] Loaded legacy LPR ROI with {len(points)} points")
            
            # Get the detection ROI if available
            if "detection_roi" in config_data:
                points = config_data["detection_roi"]["original_points"]
                self.detection_roi_polygon = np.array(points, np.int32).reshape((-1, 1, 2))
                logger.info(f"[ROI:{self.camera_id}] Loaded detection ROI with {len(points)} points")
            
            return True
            
        except Exception as e:
            logger.error(f"[ROI:{self.camera_id}] Error loading ROI config: {str(e)}")
            return False
    
    def save_roi_config(self, lpr_roi=None, detection_roi=None, original_dimensions=None):
        """
        Save ROI configuration to file
        
        Args:
            lpr_roi: List of (x,y) points for LPR ROI
            detection_roi: List of (x,y) points for detection ROI
            original_dimensions: Original frame dimensions [width, height]
        """
        try:
            # Start with existing config or create new
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
            else:
                config_data = {}
            
            # Update with new ROIs if provided
            if lpr_roi is not None:
                lpr_points = [(int(p[0]), int(p[1])) for p in lpr_roi]
                config_data["lpr_roi"] = {
                    "original_points": lpr_points
                }
                
            if detection_roi is not None:
                detection_points = [(int(p[0]), int(p[1])) for p in detection_roi]
                config_data["detection_roi"] = {
                    "original_points": detection_points
                }
                
            # Save original dimensions if provided
            if original_dimensions is not None:
                config_data["original_dimensions"] = original_dimensions
                
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"[ROI:{self.camera_id}] Saved ROI configuration to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"[ROI:{self.camera_id}] Error saving ROI config: {str(e)}")
            return False
    
    def is_point_in_roi(self, point, roi_type="lpr"):
        """
        Check if a point is inside the specified ROI
        
        Args:
            point: (x,y) point to check
            roi_type: "lpr" or "detection" - which ROI to check against
            
        Returns:
            bool: True if point is in ROI, False otherwise
        """
        x, y = point
        
        # Select appropriate ROI polygon
        roi_polygon = self.lpr_roi_polygon if roi_type == "lpr" else self.detection_roi_polygon
        
        if roi_polygon is None:
            return False
            
        # Test point against polygon
        return cv2.pointPolygonTest(roi_polygon, (x, y), False) >= 0
    
    def is_object_in_roi(self, bbox, roi_type="lpr", threshold=0.3):
        """
        Check if object bounding box overlaps with the specified ROI
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            roi_type: "lpr" or "detection" - which ROI to check against
            threshold: Minimum overlap threshold (0.0-1.0)
            
        Returns:
            bool: True if object is in ROI, False otherwise
        """
        # Select appropriate ROI polygon
        roi_polygon = self.lpr_roi_polygon if roi_type == "lpr" else self.detection_roi_polygon
        
        if roi_polygon is None:
            return False
            
        # Convert ROI polygon to a mask
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        
        # Check if the center point is in the ROI
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        return self.is_point_in_roi((center_x, center_y), roi_type)
    
    def draw_roi(self, frame, roi_type="both"):
        """
        Draw ROI polygon on frame
        
        Args:
            frame: Image to draw on
            roi_type: "lpr", "detection", or "both"
            
        Returns:
            frame with ROI drawn
        """
        if frame is None:
            return frame
            
        vis_frame = frame.copy()
        
        # Create a mask for ROI highlighting with transparency
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw LPR ROI
        if (roi_type == "lpr" or roi_type == "both") and self.lpr_roi_polygon is not None:
            # Fill the polygon with a semi-transparent red
            lpr_fill_color = (0, 0, 192)  # Red (B, G, R)
            cv2.fillPoly(overlay, [self.lpr_roi_polygon], lpr_fill_color)
            
            # Draw the outline on the main frame
            cv2.polylines(vis_frame, [self.lpr_roi_polygon], True, (0, 0, 255), 2)
            cv2.putText(vis_frame, "LPR Zone", 
                       (self.lpr_roi_polygon[0][0][0], self.lpr_roi_polygon[0][0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw detection ROI
        if (roi_type == "detection" or roi_type == "both") and self.detection_roi_polygon is not None:
            # Fill the polygon with a semi-transparent blue
            detection_fill_color = (192, 0, 0)  # Blue (B, G, R)
            cv2.fillPoly(overlay, [self.detection_roi_polygon], detection_fill_color)
            
            # Draw the outline on the main frame
            cv2.polylines(vis_frame, [self.detection_roi_polygon], True, (255, 0, 0), 2)
            cv2.putText(vis_frame, "Detection Zone", 
                       (self.detection_roi_polygon[0][0][0], self.detection_roi_polygon[0][0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Apply overlay with transparency
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
        
        return vis_frame