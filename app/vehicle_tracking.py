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
            
            # Setup GPU acceleration if available
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
                
                # Set device explicitly to ensure it's using the right hardware
                if hasattr(self.model, 'to'):
                    self.model.to(device)
                
                # Log additional GPU information if available
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

    def _setup_gpu(self):
        """Setup and verify GPU availability for Jetson Nano and other CUDA devices"""
        try:
            # First check via OpenCV CUDA support
            cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_device_count > 0:
                # Initialize CUDA context
                cv2.cuda.setDevice(0)
                try:
                    # Create a small test operation to verify CUDA is working
                    test_mat = cv2.cuda_GpuMat((10, 10), cv2.CV_8UC3)
                    test_mat.upload(np.zeros((10, 10, 3), dtype=np.uint8))
                    test_mat.release()
                    logger.info(f"[TRACKER:{self.camera_id}] OpenCV CUDA acceleration verified working")
                except Exception as e:
                    logger.warning(f"[TRACKER:{self.camera_id}] OpenCV CUDA initialized but test operation failed: {str(e)}")
                    # Continue to try other detection methods
            
            # Check via PyTorch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"[TRACKER:{self.camera_id}] CUDA available via PyTorch with device: {device_name}")
                    return True
            
            # Jetson-specific checks
            if os.path.exists('/dev/nvhost-ctrl'):
                logger.info(f"[TRACKER:{self.camera_id}] Detected Jetson hardware via /dev/nvhost-ctrl")
                # Apply Jetson-specific optimizations
                try:
                    import jetson.utils
                    logger.info(f"[TRACKER:{self.camera_id}] Jetson utils package detected")
                except ImportError:
                    logger.warning(f"[TRACKER:{self.camera_id}] Running on Jetson but jetson.utils not available")
                return True
                
            if os.path.exists('/usr/local/cuda'):
                logger.info(f"[TRACKER:{self.camera_id}] CUDA installation detected at /usr/local/cuda")
                return True
                
            # Check for Jetson-specific environment via model file
            if os.path.exists('/proc/device-tree/model'):
                try:
                    with open('/proc/device-tree/model', 'r') as f:
                        model = f.read()
                        if 'Jetson' in model:
                            logger.info(f"[TRACKER:{self.camera_id}] Detected Jetson Nano hardware from device-tree model")
                            return True
                except Exception as e:
                    logger.debug(f"[TRACKER:{self.camera_id}] Error reading device model: {str(e)}")
            
            # If we get here and OpenCV detected CUDA, we can still use it
            if cuda_device_count > 0:
                logger.info(f"[TRACKER:{self.camera_id}] GPU acceleration available via OpenCV CUDA")
                return True
                
            logger.error(f"[TRACKER:{self.camera_id}] No CUDA-capable GPU detected")
            raise RuntimeError("GPU acceleration is required but no CUDA device is available")
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Error setting up GPU: {str(e)}")
            raise RuntimeError(f"Failed to initialize GPU acceleration: {str(e)}")

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
                cv2.putText(vis_frame, f"Plate: Detected",
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 0), 3)  # Black outline
                cv2.putText(vis_frame, f"Plate Detected",
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
        
        # Display last plate info or "No plate detected" message
        if hasattr(self, 'last_recognized_plate') and self.last_recognized_plate is not None:
            plate_text = self.last_recognized_plate
            auth_status = "Authorized" if self.last_plate_authorized else "Not Authorized"
            auth_color = (0, 255, 0) if self.last_plate_authorized else (0, 0, 255)  # Green or Red
            
            # Log that we're displaying the last recognized plate
            logger.debug(f"[TRACKER] Displaying last plate: {plate_text}, status: {auth_status}")
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
        
        # Check if vehicle is in ROI
        is_in_roi = self._is_vehicle_in_roi(box)
        
        # Track ROI state transitions
        if not hasattr(self, 'vehicle_roi_state'):
            self.vehicle_roi_state = {}
        
        # If vehicle was previously in ROI but now it's not, clear its detection data
        if track_id in self.vehicle_roi_state and self.vehicle_roi_state[track_id] == True and not is_in_roi:
            logger.info(f"[TRACKER] Vehicle {track_id} exited ROI - clearing its detection data")
            # Remove from detected plates when exiting ROI
            if track_id in self.detected_plates:
                del self.detected_plates[track_id]
            # Reset plate attempts counter
            self.plate_attempts[track_id] = 0
            # Keep last recognized plate for display but mark vehicle as exited
            # Could optionally add: self.frame_buffer[track_id] = []
        
        # Update vehicle's current ROI state
        self.vehicle_roi_state[track_id] = is_in_roi
        
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
        """Calculate image quality/clarity score based on Laplacian variance with GPU acceleration and CPU fallback"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Try GPU calculation first if available
            if self.gpu_available:
                try:
                    # Upload to GPU
                    gpu_gray = cv2.cuda_GpuMat()
                    gpu_gray.upload(gray)
                    
                    # GPU Laplacian
                    gpu_laplacian = cv2.cuda.createLaplacianFilter(cv2.CV_64F, 1)
                    gpu_result = gpu_laplacian.apply(gpu_gray)
                    
                    # Download result
                    laplacian = gpu_result.download()
                    score = laplacian.var()
                    
                    # Release GPU resources
                    gpu_gray.release()
                    gpu_result.release()
                    
                    logger.debug(f"[TRACKER:{self.camera_id}] Image quality calculated using GPU")
                except Exception as e:
                    logger.warning(f"[TRACKER:{self.camera_id}] GPU calculation failed, falling back to CPU: {str(e)}")
                    # Fall back to CPU
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    score = laplacian.var()
                    logger.debug(f"[TRACKER:{self.camera_id}] Image quality calculated using CPU fallback")
            else:
                # CPU calculation
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                score = laplacian.var()
                logger.debug(f"[TRACKER:{self.camera_id}] Image quality calculated using CPU")
                
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
            # Return a default score rather than failing completely
            return 50.0  # Return a reasonable middle value as fallback

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
            # Also remove from ROI tracking state and detected plates
            if hasattr(self, 'vehicle_roi_state'):
                self.vehicle_roi_state.pop(track_id, None)
            self.detected_plates.pop(track_id, None)
            self.plate_attempts.pop(track_id, None)
            logger.info(f"Vehicle {track_id} tracking timed out - all data cleared")

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
        
        # Verify GPU is available
        if not self.gpu_available:
            logger.error("[TRACKER] GPU required for motion detection but not available")
            raise RuntimeError("GPU acceleration required for motion detection")
            
        # Use GPU for background subtraction
        try:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Use CPU for background subtraction since OpenCV CUDA doesn't have a direct equivalent
            # This is the only CPU operation we allow, as it can't be done on GPU with OpenCV CUDA
            fgmask = self.background_model.apply(frame)
            
            # Release GPU resources
            gpu_frame.release()
        except Exception as e:
            logger.error(f"[TRACKER] Background subtraction failed: {str(e)}")
            raise RuntimeError(f"Failed to perform background subtraction: {str(e)}")
        
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
        """Detect and track vehicles in frame, with strict GPU requirement."""
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
                # Check GPU availability but with fallback
                if not self.gpu_available:
                    logger.warning(f"[TRACKER:{self.camera_id}] GPU is not available but will attempt to use CPU")
                
                try:
                    # For Jetson Nano, reduce resolution if frame is large
                    # This helps with inference speed
                    h, w = frame.shape[:2]
                    if h > 720 or w > 1280:
                        try:
                            if self.gpu_available:
                                # GPU-accelerated resize
                                scale_factor = min(720 / h, 1280 / w)
                                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                                gpu_frame = cv2.cuda_GpuMat()
                                gpu_frame.upload(frame)
                                gpu_resized = cv2.cuda.resize(gpu_frame, (new_w, new_h))
                                scaled_frame = gpu_resized.download()
                                gpu_frame.release()
                                gpu_resized.release()
                                logger.debug(f"[TRACKER:{self.camera_id}] Resized frame from {w}x{h} to {new_w}x{new_h} using GPU")
                            else:
                                # CPU fallback resize
                                scale_factor = min(720 / h, 1280 / w)
                                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                                scaled_frame = cv2.resize(frame, (new_w, new_h))
                                logger.debug(f"[TRACKER:{self.camera_id}] Resized frame from {w}x{h} to {new_w}x{new_h} using CPU")
                        except Exception as e:
                            logger.warning(f"[TRACKER:{self.camera_id}] Frame resize failed: {str(e)}, using original size")
                            scaled_frame = frame
                    else:
                        scaled_frame = frame
                except Exception as e:
                    logger.warning(f"[TRACKER:{self.camera_id}] Preprocessing failed: {str(e)}, using original frame")
                    scaled_frame = frame
                
                # Run detection with optimizations for Jetson Nano
                logger.debug(f"[TRACKER:{self.camera_id}] Running YOLO detection and tracking")
                
                try:
                    # Always use half precision for faster inference if GPU is available
                    half_precision = self.gpu_available
                    
                    # Run the model with error handling
                    results = self.model.track(
                        scaled_frame,
                        persist=True,
                        classes=list(VEHICLE_CLASSES.keys()),
                        conf=0.3,
                        iou=0.45,
                        half=half_precision,
                        verbose=False,
                        device=0 if self.gpu_available else 'cpu'  # Explicitly set device
                    )
                    
                    # Process detections if we have any valid results
                    if len(results) > 0:
                        # Check if tracking IDs are available
                        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                            try:
                                # Get detections
                                boxes = results[0].boxes.xyxy.cpu().numpy()
                                track_ids = results[0].boxes.id.int().cpu().tolist()
                                class_ids = results[0].boxes.cls.int().cpu().tolist()
                                
                                # If we used a scaled frame, adjust boxes back to original frame size
                                if scaled_frame is not frame and scaled_frame.shape[:2] != frame.shape[:2]:
                                    h_ratio = frame.shape[0] / scaled_frame.shape[0]
                                    w_ratio = frame.shape[1] / scaled_frame.shape[1]
                                    for i in range(len(boxes)):
                                        boxes[i][0] *= w_ratio  # x1
                                        boxes[i][1] *= h_ratio  # y1
                                        boxes[i][2] *= w_ratio  # x2
                                        boxes[i][3] *= h_ratio  # y2
                                
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
                                                    logger.info(f"[TRACKER:{self.camera_id}] Processing license plate for vehicle {track_id} using best quality frame")
                                                    self._process_plate(best_frame, track_id)
                                                else:
                                                    logger.warning(f"[TRACKER:{self.camera_id}] Could not select best frame for vehicle {track_id}")
                                
                                if vehicles_in_roi > 0:
                                    logger.debug(f"[TRACKER:{self.camera_id}] {vehicles_in_roi} vehicles in ROI, {vehicles_processed_for_plates} processed for plates")
                                
                                # Draw all detections on the visualization frame
                                vis_frame = self.visualize_detection(vis_frame, boxes, track_ids, class_ids)
                            except Exception as e:
                                logger.error(f"[TRACKER:{self.camera_id}] Error processing detection boxes: {str(e)}")
                                # Not critical, continue with visualization
                        else:
                            # No tracking IDs available - model ran but tracking failed
                            logger.warning(f"[TRACKER:{self.camera_id}] Detection succeeded but tracking IDs not available")
                            
                            # Try to still show detections without tracking
                            try:
                                boxes = results[0].boxes.xyxy.cpu().numpy()
                                class_ids = results[0].boxes.cls.int().cpu().tolist()
                                # Use sequential IDs as placeholder
                                track_ids = list(range(len(boxes)))
                                
                                # Draw detections but with warning
                                vis_frame = self.visualize_detection(vis_frame, boxes, track_ids, class_ids)
                                cv2.putText(vis_frame, "TRACKING FAILED", (10, 90),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            except Exception as vis_e:
                                logger.error(f"[TRACKER:{self.camera_id}] Error visualizing detection without tracking: {str(vis_e)}")
                    else:
                        logger.debug(f"[TRACKER:{self.camera_id}] No detections in this frame")
                except Exception as e:
                    logger.error(f"[TRACKER:{self.camera_id}] Model inference failed: {str(e)}")
                    # Show error on visualization
                    cv2.putText(vis_frame, "DETECTION ERROR", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                # Cleanup stale vehicles periodically
                self._cleanup_stale_vehicles()
            else:
                # Still show the last detected plate info when idle
                h, w = vis_frame.shape[:2]
                
                # Draw smaller background rectangle for better aesthetics
                cv2.rectangle(vis_frame, (10, h-70), (350, h-10), (0, 0, 0), -1)
                cv2.rectangle(vis_frame, (10, h-70), (350, h-10), (255, 255, 255), 2)
                
                # Display last plate info or "No plate detected" message
                if hasattr(self, 'last_recognized_plate') and self.last_recognized_plate is not None:
                    plate_text = self.last_recognized_plate
                    auth_status = "Authorized" if self.last_plate_authorized else "Not Authorized"
                    auth_color = (0, 255, 0) if self.last_plate_authorized else (0, 0, 255)
                else:
                    plate_text = "No plate detected"
                    auth_status = "N/A"
                    auth_color = (128, 128, 128)
                
                # Display the plate info on idle frames too
                cv2.putText(vis_frame, f"Last Plate: {plate_text}", 
                           (20, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(vis_frame, f"Last Plate: {plate_text}", 
                           (20, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(vis_frame, f"Status: {auth_status}", 
                           (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(vis_frame, f"Status: {auth_status}", 
                           (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, auth_color, 1)
            
            # Add GPU info to display
            gpu_text = "GPU" if self.gpu_available else "CPU"
            gpu_color = (0, 255, 0) if self.gpu_available else (0, 0, 255)
            cv2.putText(vis_frame, f"Mode: {gpu_text}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(vis_frame, f"Mode: {gpu_text}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, gpu_color, 1)
            
            # If activity was just detected, show an indicator
            if activity_detected:
                h, w = vis_frame.shape[:2]
                cv2.putText(vis_frame, "ACTIVITY DETECTED", (w//2-150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display the visualization frame in a camera-specific window
            # On Jetson, use jetson.utils for faster display if available
            try:
                logger.info(f"[TRACKER:{self.camera_id}] Attempting to display detection window")
                
                if 'jetson.utils' in sys.modules:
                    logger.info(f"[TRACKER:{self.camera_id}] Using jetson.utils for display")
                    # Use jetson.utils for faster display
                    import jetson.utils
                    import jetson.inference
                    
                    # Convert OpenCV BGR to RGBA for jetson.utils
                    frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGBA)
                    cuda_mem = jetson.utils.cudaFromNumpy(frame_rgb)
                    
                    # Display using jetson.utils
                    window_name = f'Detections - {self.camera_id}'
                    logger.debug(f"[TRACKER:{self.camera_id}] Window name: {window_name}")
                    jetson.utils.cudaDeviceSynchronize()
                    jetson.utils.display.render(cuda_mem, width=vis_frame.shape[1], height=vis_frame.shape[0])
                    jetson.utils.cudaDeviceSynchronize()
                    logger.info(f"[TRACKER:{self.camera_id}] Successfully displayed using jetson.utils")
                else:
                    # Use GStreamer display for DeepStream compatibility
                    window_name = f'Detections - {self.camera_id}'
                    logger.info(f"[TRACKER:{self.camera_id}] Using GStreamer display with window name: {window_name}")
                    
                    try:
                        # First time we need to create the display pipeline
                        if not hasattr(self, 'display_pipeline') or self.display_pipeline is None:
                            logger.info(f"[TRACKER:{self.camera_id}] Creating GStreamer display pipeline")
                            import gi
                            gi.require_version('Gst', '1.0')
                            from gi.repository import Gst, GLib
                            
                            # Initialize GStreamer if not already initialized
                            if not Gst.is_initialized():
                                Gst.init(None)
                            
                            # Create display pipeline with nvoverlaysink for hardware acceleration
                            pipeline_str = (
                                "appsrc name=source is-live=true format=time ! "
                                "videoconvert ! video/x-raw,format=RGBA ! "
                                "nvvidconv ! "
                                f"nvoverlaysink window-width={vis_frame.shape[1]} window-height={vis_frame.shape[0]} window-x=20 window-y=20 sync=false"
                            )
                            
                            self.display_pipeline = Gst.parse_launch(pipeline_str)
                            self.appsrc = self.display_pipeline.get_by_name('source')
                            
                            # Configure appsrc
                            caps = Gst.Caps.from_string(f"video/x-raw,format=RGBA,width={vis_frame.shape[1]},height={vis_frame.shape[0]},framerate=30/1")
                            self.appsrc.set_property('caps', caps)
                            
                            # Start the pipeline
                            self.display_pipeline.set_state(Gst.State.PLAYING)
                            logger.info(f"[TRACKER:{self.camera_id}] GStreamer display pipeline started")
                        
                        # Convert frame to RGBA for GStreamer
                        frame_rgba = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGBA)
                        
                        # Create GStreamer buffer from numpy array
                        data = frame_rgba.tobytes()
                        buf = Gst.Buffer.new_allocate(None, len(data), None)
                        buf.fill(0, data)
                        
                        # Push buffer to appsrc
                        self.appsrc.emit('push-buffer', buf)
                        logger.debug(f"[TRACKER:{self.camera_id}] Frame pushed to GStreamer pipeline")
                    except Exception as gst_err:
                        logger.error(f"[TRACKER:{self.camera_id}] GStreamer error: {str(gst_err)}")
                        # Fall back to cv2.imshow as last resort
                        logger.info(f"[TRACKER:{self.camera_id}] Falling back to OpenCV display")
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, vis_frame.shape[1], vis_frame.shape[0])
                        cv2.imshow(window_name, vis_frame)
                        cv2.waitKey(1) # This needs to be called to actually display the window
            except Exception as e:
                logger.error(f"[TRACKER:{self.camera_id}] Display error: {str(e)}")
                # Print detailed stack trace
                import traceback
                logger.error(f"[TRACKER:{self.camera_id}] Display error stack trace: {traceback.format_exc()}")
                # Do not raise an exception here, as it would stop processing
                # Just continue without the display
                
            # Return the visualization frame
            return True, vis_frame
        
        except Exception as e:
            logger.error(f"[TRACKER:{self.camera_id}] Detection error: {str(e)}")
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
            
            # Verify GPU is available
            if not self.gpu_available:
                logger.error("[TRACKER] GPU required for vehicle extraction but not available")
                raise RuntimeError("GPU acceleration required for vehicle extraction")
            
            # Extract using GPU - no fallbacks to CPU
            try:
                # Upload to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Extract ROI on GPU
                gpu_vehicle = gpu_frame.colRange(x1, x2).rowRange(y1, y2)
                
                # Download result
                vehicle_img = gpu_vehicle.download()
                
                gpu_frame.release()
                gpu_vehicle.release()
                return vehicle_img
            except Exception as e:
                logger.error(f"[TRACKER] GPU extraction failed: {str(e)}")
                raise RuntimeError(f"Failed to extract vehicle region using GPU: {str(e)}")
        except Exception as e:
            logger.error(f"[TRACKER] Error extracting vehicle image: {str(e)}")
            raise RuntimeError(f"Vehicle extraction failed: {str(e)}")

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
