import cv2
import numpy as np
import json
import threading
import time
import os
import os.path as osp
from collections import defaultdict
from ultralytics import YOLO
import sys
import torch

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
            
            # Setup GPU acceleration
            self.gpu_available = self._setup_gpu()
            
            # Use absolute path for model
            model_path = osp.abspath(YOLO_MODEL_PATH)
            logger.info(f"[TRACKER:{camera_id}] Loading YOLO model from: {model_path}")
            
            if not osp.exists(model_path):
                logger.error(f"[TRACKER:{camera_id}] Model not found at {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            try:
                # Load the model with device specification
                device = 0 if self.gpu_available else 'cpu'  # Use GPU 0 if available, otherwise CPU
                self.model = YOLO(model_path)
                
                # Set device explicitly
                if hasattr(self.model, 'to'):
                    self.model.to(device)
                
                # Log GPU information
                if self.gpu_available and hasattr(torch, 'cuda') and torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    memory_allocated = torch.cuda.memory_allocated(0)
                    memory_reserved = torch.cuda.memory_reserved(0)
                    logger.info(f"[TRACKER:{camera_id}] Using GPU: {device_name}")
                    logger.info(f"[TRACKER:{camera_id}] GPU memory: allocated={memory_allocated/1024**2:.2f}MB, reserved={memory_reserved/1024**2:.2f}MB")
                
                logger.info(f"[TRACKER:{camera_id}] YOLO model loaded successfully on {'GPU' if self.gpu_available else 'CPU'}")
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
                
            # Initialize components
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
            
            # Processing state
            self.processing_active = True  # Start with active processing
            self.last_activity_time = time.time()
            self.frame_counter = 0
            self.background_model = None
            
            # State for UI
            self.last_recognized_plate = None
            self.last_plate_authorized = False
            
            # Display window setup
            self.window_name = f'Detections-{self.camera_id}'
            self._setup_display_window()
            
        except Exception as e:
            logger.error(f"[TRACKER:{camera_id}] Error in VehicleTracker initialization: {str(e)}")
            raise

    def _setup_display_window(self):
        """Setup proper display window for Jetson Nano"""
        try:
            # Create named window with proper properties
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Set initial window size (adjust as needed for your display)
            cv2.resizeWindow(self.window_name, 1280, 720)
            logger.info(f"[TRACKER:{self.camera_id}] Display window created with name: {self.window_name}")
            
            # Try to move window to a good position
            try:
                cv2.moveWindow(self.window_name, 50, 50)
            except Exception as e:
                logger.warning(f"[TRACKER:{self.camera_id}] Could not position window: {str(e)}")
                
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Error setting up display window: {str(e)}")

    def _setup_gpu(self):
        """Setup and verify GPU availability for Jetson Nano"""
        try:
            # Check via torch CUDA
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"[TRACKER:{self.camera_id}] CUDA available via PyTorch with device: {device_name}")
                    return True
            
            # Jetson-specific checks
            if os.path.exists('/dev/nvhost-ctrl'):
                logger.info(f"[TRACKER:{self.camera_id}] Detected Jetson hardware via /dev/nvhost-ctrl")
                return True
                
            if os.path.exists('/usr/local/cuda'):
                logger.info(f"[TRACKER:{self.camera_id}] CUDA installation detected at /usr/local/cuda")
                return True
                
            # Check for Jetson hardware
            if os.path.exists('/proc/device-tree/model'):
                try:
                    with open('/proc/device-tree/model', 'r') as f:
                        model = f.read()
                        if 'Jetson' in model:
                            logger.info(f"[TRACKER:{self.camera_id}] Detected Jetson hardware from device-tree model")
                            return True
                except Exception as e:
                    logger.debug(f"[TRACKER:{self.camera_id}] Error reading device model: {str(e)}")
            
            # OpenCV CUDA check
            cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_device_count > 0:
                logger.info(f"[TRACKER:{self.camera_id}] GPU acceleration available via OpenCV CUDA")
                return True
                
            logger.warning(f"[TRACKER:{self.camera_id}] No CUDA-capable GPU detected, falling back to CPU")
            return False
            
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Error setting up GPU: {str(e)}")
            logger.warning(f"[TRACKER:{self.camera_id}] Falling back to CPU processing")
            return False

    def _load_roi_polygon(self, config_path):
        """Load ROI polygon from config file"""
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
                # New format
                self.roi_config = config_data
                self.original_dimensions = config_data.get("original_dimensions", None)
                self.display_dimensions = config_data.get("display_dimensions", None)
                self.scale_ratios = config_data.get("scale_ratios", None)
                roi_points = config_data["original_points"]
            else:
                # Old format - just a list of points
                roi_points = config_data
                self.roi_config = None
                self.original_dimensions = None
                self.display_dimensions = None
                self.scale_ratios = None
                
            if not isinstance(roi_points, list) or len(roi_points) < 3:
                logger.warning(f"[TRACKER:{self.camera_id}] Invalid ROI points format in {config_path}")
                return None
                
            # Convert to numpy array
            roi_polygon = np.array(roi_points, dtype=np.int32)
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
            if not hasattr(self, 'first_frame_dims'):
                self.first_frame_dims = (frame_w, frame_h)
                logger.info(f"[TRACKER:{self.camera_id}] Stored frame dimensions {frame_w}x{frame_h} as reference")
            
            # Create a copy of the original ROI
            scaled_roi = self.original_roi.copy()
            
            # If we have the original dimensions from config, use those for precise scaling
            if hasattr(self, 'original_dimensions') and self.original_dimensions:
                orig_width, orig_height = self.original_dimensions
                
                # Calculate direct scaling factors
                scale_x = frame_w / orig_width
                scale_y = frame_h / orig_height
            else:
                # Use first frame dimensions as reference
                ref_width, ref_height = self.first_frame_dims
                
                # If dimensions match, no scaling needed
                if (frame_w, frame_h) == self.first_frame_dims:
                    return self.original_roi
                
                # Calculate scale factors
                scale_x = frame_w / ref_width
                scale_y = frame_h / ref_height
            
            # Scale the ROI coordinates
            scaled_roi[:, 0] = (scaled_roi[:, 0] * scale_x).astype(np.int32)
            scaled_roi[:, 1] = (scaled_roi[:, 1] * scale_y).astype(np.int32)
            
            return scaled_roi
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Error scaling ROI: {str(e)}")
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
                result = cv2.pointPolygonTest(self.roi_polygon, (center_x, center_y), False)
                return result >= 0
            except Exception as e:
                logger.error(f"[TRACKER:{self.camera_id}] ROI check error: {str(e)}")
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
                cv2.putText(vis_frame, f"Plate: Detected", (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(vis_frame, f"Plate: Detected", (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Update last recognized plate
                self.last_recognized_plate = self.detected_plates[track_id]
        
        # Draw ROI if available
        if self.roi_polygon is not None:
            cv2.polylines(vis_frame, [self.roi_polygon], True, (0, 255, 0), 2)
            
            # Add semi-transparent overlay
            overlay = vis_frame.copy()
            cv2.fillPoly(overlay, [self.roi_polygon], (0, 200, 0, 50))
            cv2.addWeighted(overlay, 0.2, vis_frame, 0.8, 0, vis_frame)
        
        # Draw info section at bottom
        h, w = vis_frame.shape[:2]
        
        # Draw background for info section
        cv2.rectangle(vis_frame, (10, h-70), (350, h-10), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (10, h-70), (350, h-10), (255, 255, 255), 2)
        
        # Display plate info
        if self.last_recognized_plate is not None:
            plate_text = self.last_recognized_plate
            auth_status = "Authorized" if self.last_plate_authorized else "Not Authorized"
            auth_color = (0, 255, 0) if self.last_plate_authorized else (0, 0, 255)
        else:
            plate_text = "No plate detected"
            auth_status = "N/A"
            auth_color = (128, 128, 128)
            
        cv2.putText(vis_frame, f"Last Plate: {plate_text}", (20, h-45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Status: {auth_status}", (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, auth_color, 1)
        
        # Add processing info at top
        gpu_text = "GPU" if self.gpu_available else "CPU"
        gpu_color = (0, 255, 0) if self.gpu_available else (0, 0, 255)
        cv2.putText(vis_frame, f"Processing: {gpu_text}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, gpu_color, 2)
        
        return vis_frame

    def _update_vehicle_state(self, track_id, frame, box):
        """Update vehicle tracking state and frame buffer"""
        current_time = time.time()
        self.last_vehicle_tracking_time[track_id] = current_time
        
        # Check if vehicle is in ROI
        is_in_roi = self._is_vehicle_in_roi(box)
        
        # Track ROI state transitions
        if not hasattr(self, 'vehicle_roi_state'):
            self.vehicle_roi_state = {}
        
        # If vehicle was previously in ROI but now it's not, clear its detection data
        if track_id in self.vehicle_roi_state and self.vehicle_roi_state[track_id] == True and not is_in_roi:
            logger.info(f"[TRACKER:{self.camera_id}] Vehicle {track_id} exited ROI - clearing detection data")
            # Remove from detected plates when exiting ROI
            if track_id in self.detected_plates:
                del self.detected_plates[track_id]
            # Reset plate attempts counter
            self.plate_attempts[track_id] = 0
        
        # Update vehicle's current ROI state
        self.vehicle_roi_state[track_id] = is_in_roi
        
        # Add frame to buffer for vehicles in ROI
        if is_in_roi:
            if track_id not in self.frame_buffer:
                self.frame_buffer[track_id] = []
            
            vehicle_img = self._extract_vehicle_image(frame, box)
            if vehicle_img is not None:
                # Calculate image quality score
                clarity_score = self._calculate_image_quality(vehicle_img)
                self.frame_buffer[track_id].append((vehicle_img, clarity_score))
                
                # Keep buffer at maximum size
                if len(self.frame_buffer[track_id]) > self.max_buffer_size:
                    self.frame_buffer[track_id].pop(0)

    def _calculate_image_quality(self, image):
        """Calculate image quality/clarity score"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate Laplacian variance (sharpness)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            score = laplacian.var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = np.std(gray)
            
            # Combined score (higher is better)
            combined_score = (score * 0.5) + (contrast * 0.5)
            
            # Penalize very dark or very bright images
            if brightness < 30 or brightness > 220:
                combined_score *= 0.7
                
            return combined_score
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Error calculating image quality: {str(e)}")
            return 50.0  # Return a default middle value

    def _cleanup_stale_vehicles(self):
        """Remove vehicles that haven't been seen recently"""
        current_time = time.time()
        stale_ids = []
        
        for track_id, last_time in self.last_vehicle_tracking_time.items():
            if current_time - last_time > self.vehicle_tracking_timeout:
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            # Remove from all tracking collections
            self.last_vehicle_tracking_time.pop(track_id, None)
            self.frame_buffer.pop(track_id, None)
            if hasattr(self, 'vehicle_roi_state'):
                self.vehicle_roi_state.pop(track_id, None)
            self.detected_plates.pop(track_id, None)
            self.plate_attempts.pop(track_id, None)

    def detect_vehicles(self, frame):
        """Detect and track vehicles in frame, optimized for Jetson Nano"""
        if frame is None:
            logger.warning(f"[TRACKER:{self.camera_id}] Received None frame for detection")
            return False, None
            
        try:
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Scale ROI to match current frame dimensions
            self.roi_polygon = self._scale_roi_to_frame(frame)
            
            # Always process frames for detection on Jetson Nano
            # Skip background subtraction optimization for simplicity
            
            try:
                # For Jetson Nano, reduce resolution if frame is large
                h, w = frame.shape[:2]
                if h > 720 or w > 1280:
                    scale_factor = min(720 / h, 1280 / w)
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    scaled_frame = cv2.resize(frame, (new_w, new_h))
                    logger.debug(f"[TRACKER:{self.camera_id}] Resized frame from {w}x{h} to {new_w}x{new_h}")
                else:
                    scaled_frame = frame
            except Exception as e:
                logger.warning(f"[TRACKER:{self.camera_id}] Frame resize failed: {str(e)}")
                scaled_frame = frame
            
            # Run detection with optimizations for Jetson Nano
            try:
                # Always use half precision for faster inference
                half_precision = self.gpu_available
                
                # Run the model
                results = self.model.track(
                    scaled_frame,
                    persist=True,
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=0.3,
                    iou=0.45,
                    half=half_precision,
                    verbose=False,
                    device=0 if self.gpu_available else 'cpu'
                )
                
                # Process detections if we have valid results
                if len(results) > 0 and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    # Get detections
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    
                    # If we used a scaled frame, adjust boxes back to original size
                    if scaled_frame is not frame and scaled_frame.shape[:2] != frame.shape[:2]:
                        h_ratio = frame.shape[0] / scaled_frame.shape[0]
                        w_ratio = frame.shape[1] / scaled_frame.shape[1]
                        for i in range(len(boxes)):
                            boxes[i][0] *= w_ratio  # x1
                            boxes[i][1] *= h_ratio  # y1
                            boxes[i][2] *= w_ratio  # x2
                            boxes[i][3] *= h_ratio  # y2
                    
                    # Process each detected vehicle
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        if class_id not in VEHICLE_CLASSES:
                            continue
                            
                        # Update vehicle state for tracking
                        self._update_vehicle_state(track_id, frame, box)
                        
                        # Only process license plates for vehicles in ROI
                        if self._is_vehicle_in_roi(box):
                            # Only try to detect license plate if not already detected
                            if track_id not in self.detected_plates and track_id in self.frame_buffer:
                                # Process license plate if we have enough frames buffered
                                if len(self.frame_buffer[track_id]) >= 3:
                                    # Get best frame from buffer
                                    best_frame = self._select_best_frame(self.frame_buffer[track_id])
                                    if best_frame is not None:
                                        # Process the plate on the best quality frame
                                        self._process_plate(best_frame, track_id)
                    
                    # Draw all detections on visualization frame
                    vis_frame = self.visualize_detection(vis_frame, boxes, track_ids, class_ids)
                else:
                    logger.debug(f"[TRACKER:{self.camera_id}] No detections in this frame")
            except Exception as e:
                logger.error(f"[TRACKER:{self.camera_id}] Model inference failed: {str(e)}")
                # Show error on visualization
                cv2.putText(vis_frame, "DETECTION ERROR", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            # Cleanup stale vehicles periodically
            self._cleanup_stale_vehicles()
            
            # Display the visualization frame
            self._display_frame(vis_frame)
            
            return True, vis_frame
            
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Detection error: {str(e)}")
            return False, None

    def _display_frame(self, frame):
        """Display the frame with proper window management for Jetson Nano"""
        try:
            # Try different display methods based on available libraries
            
            # First try jetson.utils if available (much faster display)
            try:
                if 'jetson.utils' in sys.modules:
                    import jetson.utils
                    
                    # Convert OpenCV BGR to RGBA for jetson.utils
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    cuda_mem = jetson.utils.cudaFromNumpy(frame_rgb)
                    
                    # Create display if not already made
                    if not hasattr(self, 'display'):
                        # Create a display window with title
                        self.display = jetson.utils.glDisplay(f"Detections - {self.camera_id}")
                        logger.info(f"[TRACKER:{self.camera_id}] Created jetson.utils display window")
                    
                    # Render the frame
                    self.display.RenderOnce(cuda_mem, frame.shape[1], frame.shape[0])
                    jetson.utils.cudaDeviceSynchronize()
                    return
            except Exception as e:
                logger.warning(f"[TRACKER:{self.camera_id}] jetson.utils display failed: {str(e)}")
            
            # Fall back to standard OpenCV display
            cv2.imshow(self.window_name, frame)
            # Important: use a short wait key to process window events
            cv2.waitKey(1)
            
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Display error: {str(e)}")

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
            
            # Extract region
            return frame[y1:y2, x1:x2].copy()
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Error extracting vehicle image: {str(e)}")
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
                self.last_recognized_plate = plate_text
                self.last_plate_authorized = False  # Default to not authorized
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
