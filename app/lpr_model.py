import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import shutil
from PIL import Image, ImageFont, ImageDraw
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import threading
import queue
import sys
from typing import Dict, Tuple, Optional, List, Callable, Any

from .config import (
    ARABIC_MAPPING,
    FONT_PATH,
    LPR_MODEL_PATH,
    logger,
)

# Set environment variables to help with CUDA detection on Jetson platforms
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# For Jetson, try to enable TensorRT acceleration
if os.path.exists('/proc/device-tree/model'):
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'Jetson' in model:
                # Jetson-specific environment variables
                logger.info("[LPR] Setting Jetson-specific environment variables")
                os.environ["TORCH_CUDA_ARCH_LIST"] = "5.3;6.2;7.2"  # For Jetson Nano/TX1/TX2/Xavier
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Help with memory fragmentation
                
                # Try importing Jetson-specific libraries and enabling optimizations
                try:
                    import torch
                    # Disable benchmarking which can cause issues on Jetson
                    torch.backends.cudnn.benchmark = False
                    # Deterministic mode can be slower but more stable
                    torch.backends.cudnn.deterministic = True
                    logger.info("[LPR] Set PyTorch CUDA flags for Jetson")
                except ImportError:
                    pass
    except Exception as e:
        logger.warning(f"[LPR] Error reading Jetson model info: {str(e)}")

# Use a lower resolution for license plate detection to speed up inference
PLATE_DETECTION_SIZE = 480

def preprocess_image(image, use_gpu=True):
    """
    Preprocess the vehicle snapshot to improve license plate recognition.
    Steps:
      1. Convert to grayscale.
      2. Perform histogram equalization for contrast enhancement.
      3. Convert back to BGR.
      4. Apply an unsharp mask to sharpen the image.
      
    Args:
        image: Input image
        use_gpu: Whether to use GPU acceleration (Always True now)
    """
    try:
        # Always verify GPU is available - should be True
        if not cv2.cuda.getCudaEnabledDeviceCount() > 0:
            raise RuntimeError("CUDA device required for image preprocessing but none available")
        
        # Create GPU matrices
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        
        # Convert to grayscale
        gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram to enhance contrast
        gpu_equ = cv2.cuda.equalizeHist(gpu_gray)
        
        # Convert back to BGR format
        gpu_equ_bgr = cv2.cuda.cvtColor(gpu_equ, cv2.COLOR_GRAY2BGR)
        
        # Download for GaussianBlur (not available in CUDA)
        equ_bgr = gpu_equ_bgr.download()
        
        # Release GPU resources
        gpu_image.release()
        gpu_gray.release()
        gpu_equ.release()
        gpu_equ_bgr.release()
        
        # Apply unsharp mask for sharpening
        blurred = cv2.GaussianBlur(equ_bgr, (0, 0), 3)
        sharpened = cv2.addWeighted(equ_bgr, 1.5, blurred, -0.5, 0)
        
        return sharpened
    except Exception as e:
        logger.error(f"[LPR] GPU preprocessing failed: {str(e)}")
        raise RuntimeError(f"Failed to perform GPU image preprocessing: {str(e)}")

# Helper function to check GPU availability
def setup_gpu():
    """Setup and verify GPU availability for Jetson Nano and other CUDA devices"""
    try:
        # Check if CUDA is available for OpenCV
        logger.info("[LPR] Checking for CUDA-enabled GPU...")
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if cuda_available:
            # Try to get CUDA device properties
            try:
                device_count = cv2.cuda.getCudaEnabledDeviceCount()
                logger.info(f"[LPR] Found {device_count} CUDA device(s)")
                for i in range(device_count):
                    try:
                        props = cv2.cuda.getDevice()
                        logger.info(f"[LPR] CUDA device {i} selected: {props}")
                    except Exception as e:
                        logger.warning(f"[LPR] Could not get CUDA device {i} properties: {str(e)}")
            except Exception as e:
                logger.warning(f"[LPR] Error checking CUDA device properties: {str(e)}")
                
            # Initialize CUDA context
            cv2.cuda.setDevice(0)
            logger.info("[LPR] CUDA device set to 0")
            
            # Create a small test operation to verify CUDA is working
            try:
                test_mat = cv2.cuda_GpuMat((10, 10), cv2.CV_8UC3)
                test_mat.upload(np.zeros((10, 10, 3), dtype=np.uint8))
                # Try a simple operation
                cv2.cuda.cvtColor(test_mat, cv2.COLOR_BGR2GRAY)
                test_mat.release()
                logger.info("[LPR] Successfully verified CUDA operation")
            except Exception as e:
                logger.error(f"[LPR] CUDA test operation failed: {str(e)}")
                return False
            
            # Check for Jetson-specific environment
            if os.path.exists('/proc/device-tree/model'):
                try:
                    with open('/proc/device-tree/model', 'r') as f:
                        model = f.read()
                        if 'Jetson' in model:
                            logger.info(f"[LPR] Detected Jetson hardware: {model.strip()}")
                            # Apply Jetson-specific optimizations
                            try:
                                # Try to import torch first to see if it's available
                                import torch
                                logger.info(f"[LPR] PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
                                if torch.cuda.is_available():
                                    logger.info(f"[LPR] PyTorch CUDA device count: {torch.cuda.device_count()}")
                                    logger.info(f"[LPR] Current CUDA device: {torch.cuda.current_device()}")
                                    logger.info(f"[LPR] CUDA device name: {torch.cuda.get_device_name(0)}")
                            except ImportError:
                                logger.info("[LPR] PyTorch not available or not imported")
                            except Exception as e:
                                logger.warning(f"[LPR] Error checking PyTorch CUDA: {str(e)}")
                                
                            try:
                                import jetson.utils
                                logger.info("[LPR] Jetson utils package detected")
                            except ImportError:
                                logger.warning("[LPR] Running on Jetson but jetson.utils not available")
                except Exception as e:
                    logger.debug(f"[LPR] Error reading device model: {str(e)}")
            
            # Try importing CUDA packages
            try:
                import pycuda.driver as cuda
                logger.info("[LPR] PyCUDA available")
                cuda.init()
                logger.info(f"[LPR] PyCUDA device count: {cuda.Device.count()}")
                for i in range(cuda.Device.count()):
                    device = cuda.Device(i)
                    logger.info(f"[LPR] PyCUDA device {i} name: {device.name()}")
            except ImportError:
                logger.info("[LPR] PyCUDA not available")
            except Exception as e:
                logger.warning(f"[LPR] Error initializing PyCUDA: {str(e)}")
            
            logger.info("[LPR] GPU acceleration enabled for OpenCV")
            return True
        else:
            logger.warning("[LPR] No CUDA-capable GPU detected by OpenCV")
            # Additional check using other libraries
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("[LPR] PyTorch detects CUDA is available, but OpenCV doesn't")
                    # Try to enable CUDA for OpenCV
                    os.environ["OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES"] = "1"
                    logger.info("[LPR] Enabled OpenCL for all devices")
                    return True
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"[LPR] Error checking PyTorch CUDA: {str(e)}")
            
            return False
    except Exception as e:
        logger.error(f"[LPR] Error setting up GPU, falling back to CPU: {str(e)}")
        return False

# Helper functions that will be called by ProcessPoolExecutor
def _find_plate_in_image(model_path, image, use_gpu=False):
    try:
        # Load model with explicit device parameter
        device = 0 if use_gpu else 'cpu'
        model = YOLO(model_path)
        
        # Set model to GPU if available
        if use_gpu:
            try:
                # First try string format
                try:
                    model.to('cuda:0')
                except Exception:
                    # Then try integer
                    model.to(0)
                logger.debug("[LPR] Plate detection model set to GPU")
            except Exception as e:
                logger.debug(f"[LPR] Could not set plate detection model to GPU: {str(e)}")
                use_gpu = False

        # Preprocess the image before detection
        preprocessed = preprocess_image(image, use_gpu)
        
        # Use half precision for GPU inference
        half_precision = use_gpu
        
        # Extra debugging for shape
        logger.debug(f"[LPR] Input shape for detection: {preprocessed.shape}")
        
        # Run prediction with explicit device
        results = model.predict(
            preprocessed, 
            conf=0.25,
            iou=0.5,
            verbose=False,
            imgsz=PLATE_DETECTION_SIZE,
            half=half_precision,
            device=device  # Explicit device parameter
        )
        
        # Extra logging to verify prediction ran
        if results and len(results) > 0:
            logger.debug(f"[LPR] Detection results received: {len(results[0].boxes)} boxes")
        else:
            logger.debug("[LPR] No detection results")
            return None, None, None

        plate_boxes = []
        plate_scores = []
        lpr_class_names = model.names
        
        for box, cls, conf in zip(
            results[0].boxes.xyxy, 
            results[0].boxes.cls, 
            results[0].boxes.conf
        ):
            if lpr_class_names[int(cls)] == "License Plate":
                plate_boxes.append(box.cpu().numpy())
                plate_scores.append(float(conf))

        if not plate_boxes:
            logger.debug("[LPR] No license plates detected in boxes")
            return None, None, None

        plate_boxes = np.array(plate_boxes)
        plate_scores = np.array(plate_scores)
        best_idx = np.argmax(plate_scores)
        if plate_scores[best_idx] < 0.6:
            logger.debug(f"[LPR] Best plate has low confidence: {plate_scores[best_idx]:.2f}")
            return None, None, None

        h, w = preprocessed.shape[:2]
        x1, y1, x2, y2 = plate_boxes[best_idx]
        pad_x = (x2 - x1) * 0.1
        pad_y = (y2 - y1) * 0.1
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))
        
        # Use GPU for ROI extraction if available
        if use_gpu:
            try:
                gpu_preprocessed = cv2.cuda_GpuMat()
                gpu_preprocessed.upload(preprocessed)
                gpu_plate = gpu_preprocessed.colRange(x1, x2).rowRange(y1, y2)
                plate_img = gpu_plate.download()
                gpu_preprocessed.release()
                gpu_plate.release()
                logger.debug("[LPR] Plate ROI extracted using GPU")
            except Exception as e:
                logger.debug(f"[LPR] GPU extraction failed, using CPU: {str(e)}")
                plate_img = preprocessed[y1:y2, x1:x2]
        else:
            plate_img = preprocessed[y1:y2, x1:x2]
        
        logger.debug(f"[LPR] Plate detected with confidence: {plate_scores[best_idx]:.2f}")
        return plate_img, plate_scores[best_idx], preprocessed
    except Exception as e:
        logger.error(f"Error in _find_plate_in_image: {str(e)}")
        return None, None, None

def _recognize_plate(model_path, plate_image, use_gpu=False):
    try:
        # Load model with explicit device parameter
        device = 0 if use_gpu else 'cpu'
        model = YOLO(model_path)
        
        # Set model to GPU if available
        if use_gpu:
            try:
                # First try string format
                try:
                    model.to('cuda:0')
                except Exception:
                    # Then try integer
                    model.to(0)
                logger.debug("[LPR] Character recognition model set to GPU")
            except Exception as e:
                logger.debug(f"[LPR] Could not set character recognition model to GPU: {str(e)}")
                use_gpu = False
                
        # Use half precision for GPU inference
        half_precision = use_gpu
        
        # Extra debugging for shape
        logger.debug(f"[LPR] Input shape for recognition: {plate_image.shape}")
        
        # Run prediction with explicit device
        results = model.predict(
            plate_image, 
            conf=0.25,
            iou=0.45,
            imgsz=PLATE_DETECTION_SIZE,
            verbose=False,
            half=half_precision,
            device=device  # Explicit device parameter
        )
        
        if not results or len(results[0].boxes) == 0:
            logger.debug("[LPR] No characters detected on license plate")
            return None

        lpr_class_names = model.names
        boxes_and_classes = [
            (float(box[0]), float(box[2]), lpr_class_names[int(cls)], conf)
            for box, cls, conf in zip(
                results[0].boxes.xyxy,
                results[0].boxes.cls,
                results[0].boxes.conf,
            )
        ]
        boxes_and_classes.sort(key=lambda b: b[0])
        unmapped_chars = [
            cls for _, _, cls, _ in boxes_and_classes if cls in ARABIC_MAPPING
        ]
        license_text = "".join([ARABIC_MAPPING.get(c, c) for c in unmapped_chars if c in ARABIC_MAPPING])
        
        if license_text:
            logger.debug(f"[LPR] Plate recognized: {license_text}")
            return license_text
        else:
            logger.debug("[LPR] No valid characters found on license plate")
            return None
    except Exception as e:
        logger.error(f"Error in _recognize_plate: {str(e)}")
        return None

class PlateProcessor:
    """
    The PlateProcessor class handles license plate recognition from vehicle images.
    It uses YOLO-based object detection to identify license plates and characters.
    """
    
    def __init__(self, max_workers=None):
        logger.info("Initializing license plate recognition model...")
        
        # Configure system
        self.font_path = FONT_PATH
        self.gpu_available = False  # Will be set to True if CUDA is available
        
        # Check for GPU support first - this is mandatory
        try:
            # Setup GPU acceleration
            self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            if not self.gpu_available:
                logger.error("[LPR] No CUDA-capable GPU detected - GPU is required")
                raise RuntimeError("GPU acceleration is required but no CUDA device was found")
                
            # Initialize CUDA context
            cv2.cuda.setDevice(0)
            logger.info("[LPR] CUDA device set to 0")
            
            # Create a small test operation to verify CUDA is working
            test_mat = cv2.cuda_GpuMat((10, 10), cv2.CV_8UC3)
            test_mat.upload(np.zeros((10, 10, 3), dtype=np.uint8))
            # Try a simple operation
            cv2.cuda.cvtColor(test_mat, cv2.COLOR_BGR2GRAY)
            test_mat.release()
            logger.info("[LPR] Successfully verified CUDA operation")
        except Exception as e:
            logger.error(f"[LPR] CUDA initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize GPU acceleration: {str(e)}")
        
        try:
            # Try to check CUDA support on PyTorch side which is used by YOLO
            try:
                import torch
                self.torch_cuda_available = torch.cuda.is_available()
                if not self.torch_cuda_available:
                    logger.error("[LPR] PyTorch CUDA is NOT available - required for model inference")
                    raise RuntimeError("PyTorch CUDA support is required but not available")
                    
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"[LPR] PyTorch detected CUDA device: {device_name}")
                    
                    # Explicitly set the PyTorch device
                    torch.cuda.set_device(0)
                    logger.info(f"[LPR] PyTorch CUDA device set to 0")
                    
                    # Force synchronization to make sure device is initialized
                    torch.cuda.synchronize()
                else:
                    logger.error("[LPR] PyTorch reports no CUDA devices available")
                    raise RuntimeError("PyTorch found no CUDA devices")
            except ImportError:
                logger.error("[LPR] PyTorch not imported - required for model inference")
                raise RuntimeError("PyTorch is required but not installed")
            except Exception as e:
                logger.error(f"[LPR] Error checking PyTorch CUDA: {str(e)}")
                raise RuntimeError(f"PyTorch CUDA error: {str(e)}")
                
            # Download model if needed
            if not os.path.exists(LPR_MODEL_PATH):
                logger.info("Downloading LPR model for the first time...")
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download("omarelsayeed/licence_plates") 
                source_model = os.path.join(model_dir, "license_yolo8s_1024.pt")
                shutil.copy2(source_model, LPR_MODEL_PATH)
                logger.info("LPR model downloaded successfully")
            
            # Initialize a local model for synchronous operations
            # We need to specify the device explicitly
            device_arg = "cuda:0"  # Explicitly specify CUDA device 0
            logger.info(f"[LPR] Setting YOLO device to {device_arg}")
                
            # Load the model with explicit device specification
            self.lpr_model = YOLO(LPR_MODEL_PATH)
            logger.info(f"[LPR] YOLO model loaded, now setting device to {device_arg}")
            
            # Set model to GPU using the to() method
            try:
                # First check the model's current device
                if hasattr(self.lpr_model, 'model') and hasattr(self.lpr_model.model, 'device'):
                    logger.info(f"[LPR] Model's current device: {self.lpr_model.model.device}")
                    
                # Try explicit device setting
                self.lpr_model.to(device_arg)
                logger.info(f"[LPR] Model moved to {device_arg}")
                
                # Check if it worked
                if hasattr(self.lpr_model, 'model') and hasattr(self.lpr_model.model, 'device'):
                    logger.info(f"[LPR] Model's device after moving: {self.lpr_model.model.device}")
                    
                    # Verify it's actually on CUDA
                    if not str(self.lpr_model.model.device).startswith('cuda'):
                        logger.error(f"[LPR] Failed to move model to GPU: {self.lpr_model.model.device}")
                        raise RuntimeError(f"Model failed to move to GPU, current device: {self.lpr_model.model.device}")
            except Exception as e:
                logger.error(f"[LPR] Failed to set model to GPU: {str(e)}")
                raise RuntimeError(f"Failed to set model to GPU: {str(e)}")
            
            # Initialize thread pools for parallel processing
            if max_workers is None:
                # Use fewer workers than CPU cores to avoid overloading the system
                # Especially important on Jetson with limited resources
                max_workers = max(1, multiprocessing.cpu_count() // 2)
                
            logger.info(f"[LPR] Using {max_workers} workers for parallel processing")
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.thread_local = threading.local()
            
            logger.info("License plate recognition model initialized successfully on GPU")
        except Exception as e:
            logger.error(f"Error initializing license plate recognition model: {str(e)}")
            raise

    def __del__(self):
        # Clean up resources
        self.shutdown()

    def shutdown(self):
        """Properly shutdown all resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def find_best_plate_in_image(self, image):
        """
        Synchronous method to find the best license plate in an image.
        Maintains backwards compatibility with existing code.
        """
        if image is None:
            logger.warning("Empty image provided to find_best_plate_in_image")
            return None
            
        try:
            # Preprocess the image before detection
            preprocessed = preprocess_image(image, True)  # Always use GPU
            
            # Always use half precision for GPU inference
            half_precision = True
            
            # Set explicit device
            device = 0  # Always use GPU
            logger.debug(f"[LPR] Running plate detection on device: {device}")
            
            # Make sure model is on the right device before prediction
            try:
                # First try string format
                try:
                    self.lpr_model.to('cuda:0')
                except Exception:
                    # Then try integer
                    self.lpr_model.to(0)
            except Exception as e:
                logger.error(f"[LPR] Failed to move model to GPU before detection: {str(e)}")
                raise RuntimeError(f"Failed to set model to GPU: {str(e)}")
            
            # Get image shape for debugging
            logger.debug(f"[LPR] Plate detection input shape: {preprocessed.shape}")
            
            results = self.lpr_model.predict(
                preprocessed, 
                conf=0.25,
                iou=0.5,
                verbose=False,
                imgsz=PLATE_DETECTION_SIZE,
                half=half_precision,
                device=device  # Explicit device parameter
            )
            
            if not results or len(results[0].boxes) == 0:
                logger.info("No license plate detected in the image")
                return None

            plate_boxes = []
            plate_scores = []
            lpr_class_names = self.lpr_model.names
            
            for box, cls, conf in zip(
                results[0].boxes.xyxy, 
                results[0].boxes.cls, 
                results[0].boxes.conf
            ):
                if lpr_class_names[int(cls)] == "License Plate":
                    plate_boxes.append(box.cpu().numpy())
                    plate_scores.append(float(conf))

            if not plate_boxes:
                logger.info("No license plates found in the detections")
                return None

            plate_boxes = np.array(plate_boxes)
            plate_scores = np.array(plate_scores)
            best_idx = np.argmax(plate_scores)
            if plate_scores[best_idx] < 0.6:
                logger.info(f"Best plate detection has low confidence: {plate_scores[best_idx]:.2f}")
                return None

            h, w = preprocessed.shape[:2]
            x1, y1, x2, y2 = plate_boxes[best_idx]
            pad_x = (x2 - x1) * 0.1
            pad_y = (y2 - y1) * 0.1
            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(w, int(x2 + pad_x))
            y2 = min(h, int(y2 + pad_y))
            
            # Always use GPU for ROI extraction
            try:
                gpu_preprocessed = cv2.cuda_GpuMat()
                gpu_preprocessed.upload(preprocessed)
                gpu_plate = gpu_preprocessed.colRange(x1, x2).rowRange(y1, y2)
                plate_img = gpu_plate.download()
                gpu_preprocessed.release()
                gpu_plate.release()
            except Exception as e:
                logger.error(f"[LPR] GPU extraction failed: {str(e)}")
                raise RuntimeError(f"Failed to extract plate region using GPU: {str(e)}")
                
            logger.info(f"License plate detected with confidence: {plate_scores[best_idx]:.2f}")
            return plate_img

        except Exception as e:
            logger.error(f"Error detecting license plate: {str(e)}")
            raise

    def recognize_plate(self, plate_image):
        """
        Synchronous method to recognize text on a license plate.
        Maintains backwards compatibility with existing code.
        """
        if plate_image is None:
            logger.warning("Empty plate image provided to recognize_plate")
            return None
            
        try:
            # Always use half precision for GPU inference
            half_precision = True
            
            # Set explicit device
            device = 0  # Always use GPU
            logger.debug(f"[LPR] Running plate recognition on device: {device}")
            
            # Make sure model is on the right device before prediction
            try:
                # First try string format
                try:
                    self.lpr_model.to('cuda:0')
                except Exception:
                    # Then try integer
                    self.lpr_model.to(0)
            except Exception as e:
                logger.error(f"[LPR] Failed to move model to GPU before recognition: {str(e)}")
                raise RuntimeError(f"Failed to set model to GPU: {str(e)}")
            
            # Get image shape for debugging
            logger.debug(f"[LPR] Plate recognition input shape: {plate_image.shape}")
            
            results = self.lpr_model.predict(
                plate_image, 
                conf=0.25,
                iou=0.45,
                imgsz=PLATE_DETECTION_SIZE,
                verbose=False,
                half=half_precision,
                device=device  # Explicit device parameter
            )
            
            if not results or len(results[0].boxes) == 0:
                logger.info("No characters detected on license plate")
                return None

            lpr_class_names = self.lpr_model.names
            boxes_and_classes = [
                (float(box[0]), float(box[2]), lpr_class_names[int(cls)], conf)
                for box, cls, conf in zip(
                    results[0].boxes.xyxy,
                    results[0].boxes.cls,
                    results[0].boxes.conf,
                )
            ]
            boxes_and_classes.sort(key=lambda b: b[0])
            unmapped_chars = [
                cls for _, _, cls, _ in boxes_and_classes if cls in ARABIC_MAPPING
            ]
            
            if not unmapped_chars:
                return None
                
            mapped_chars = [ARABIC_MAPPING.get(c, c) for c in unmapped_chars]
            plate_text = "".join(mapped_chars)
            logger.info(f"License plate text: {plate_text}")
            return plate_text

        except Exception as e:
            logger.error(f"Error recognizing license plate: {str(e)}")
            raise

    def add_text_to_image(self, image, text):
        """Add recognized license plate text to the image"""
        if not text or image is None:
            return image
            
        try:
            # Skip expensive text rendering if it's a small image or on Jetson Nano
            # to improve performance, as this is just a visualization function
            image_size = image.shape[0] * image.shape[1]
            is_small_image = image_size < 100000  # Skip for small images
            
            if self.gpu_available and is_small_image:
                # Simplified version for Jetson Nano - just add a generic text indicator
                h, w = image.shape[:2]
                cv2.rectangle(image, (10, h-30), (w-10, h-10), (0, 0, 0), -1)
                cv2.putText(image, "Plate Detected", (15, h-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return image
            
            # Full version with proper text rendering
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            height, _ = image.shape[:2]
            font_size = int(height / 15)
            try:
                font = ImageFont.truetype(self.font_path, font_size) if self.font_path else ImageFont.load_default()
            except Exception as e:
                logger.warning(f"Error loading font: {str(e)}. Using default font.")
                font = ImageFont.load_default()
            draw = ImageDraw.Draw(pil_image)
            separated_text = "-".join(text)
            padding = 20
            text_bbox = draw.textbbox((0, 0), separated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = padding
            y = height - text_height - padding * 2
            background_coords = [x - padding, y - padding, x + text_width + padding, y + text_height + padding]
            draw.rectangle(background_coords, fill=(0, 0, 0, 180))
            draw.text((x, y), separated_text, font=font, fill=(255, 255, 255))
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"Could not add text to image: {str(e)}")
            return image

    def visualize_roi(self, image, roi_polygon=None):
        """
        Draw ROI visualization on the image for gate entry logging.
        If no ROI is available, returns the original image.
        
        Args:
            image (numpy.ndarray): The input image
            roi_polygon (numpy.ndarray, optional): ROI polygon points. If None, returns the original image.
        
        Returns:
            numpy.ndarray: Image with ROI drawn
        """
        if image is None:
            return None
        
        if roi_polygon is None:
            # Return original image if no ROI provided
            return image
        
        try:
            # Create a copy to avoid modifying the original
            result = image.copy()
            
            # Draw the ROI polygon
            cv2.polylines(result, [roi_polygon], True, (0, 255, 0), 3)
            
            # Add some transparency inside the ROI
            overlay = result.copy()
            cv2.fillPoly(overlay, [roi_polygon], (0, 200, 0, 50))
            alpha = 0.15
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            
            return result
        except Exception as e:
            logger.warning(f"Error visualizing ROI: {str(e)}")
            return image

    def process_vehicle_image(self, vehicle_image, save_path=None):
        """
        Synchronous method for processing a vehicle image and extracting the license plate.
        Maintains backwards compatibility with existing code.
        """
        try:
            # Optimize preprocessing with GPU if available
            start_time = time.time()
            if self.gpu_available:
                try:
                    # Upload to GPU for preprocessing
                    gpu_vehicle = cv2.cuda_GpuMat()
                    gpu_vehicle.upload(vehicle_image)
                    
                    # Use OpenCV CUDA for preprocessing if possible
                    gpu_gray = cv2.cuda.cvtColor(gpu_vehicle, cv2.COLOR_BGR2GRAY)
                    gpu_equ = cv2.cuda.equalizeHist(gpu_gray)
                    gpu_equ_bgr = cv2.cuda.cvtColor(gpu_equ, cv2.COLOR_GRAY2BGR)
                    
                    # Download for operations not available in CUDA
                    equ_bgr = gpu_equ_bgr.download()
                    gpu_vehicle.release()
                    gpu_gray.release()
                    gpu_equ.release()
                    gpu_equ_bgr.release()
                    
                    # Continue with standard preprocessing
                    blurred = cv2.GaussianBlur(equ_bgr, (0, 0), 3)
                    preprocessed_vehicle = cv2.addWeighted(equ_bgr, 1.5, blurred, -0.5, 0)
                except Exception as e:
                    logger.debug(f"[LPR] GPU preprocessing failed, using CPU: {str(e)}")
                    preprocessed_vehicle = preprocess_image(vehicle_image, self.gpu_available)
            else:
                preprocessed_vehicle = preprocess_image(vehicle_image, self.gpu_available)
            
            # Find license plate in the image
            plate_image = self.find_best_plate_in_image(preprocessed_vehicle)
            if plate_image is None:
                logger.info("No license plate found in vehicle image")
                return None, None
                
            # Recognize text on the plate
            plate_text = self.recognize_plate(plate_image)
            if plate_text is None:
                logger.info("Could not recognize text on license plate")
                return None, None
                
            # Add text overlay to the plate image
            processed_image = self.add_text_to_image(plate_image, plate_text)
            
            if save_path and processed_image is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, processed_image)
                logger.info(f"Saved processed plate image to {save_path}")
                
            process_time = time.time() - start_time
            logger.debug(f"[LPR] Vehicle image processed in {process_time:.3f}s on {'GPU' if self.gpu_available else 'CPU'}")
            return plate_text, processed_image
        except Exception as e:
            logger.error(f"Error processing vehicle image: {str(e)}")
            return None, None

    def submit_image(self, vehicle_image, save_path=None, callback=None):
        """
        Submit an image for non-blocking, asynchronous processing.
        
        Args:
            vehicle_image: The vehicle image to process
            save_path: Optional path to save the processed plate image
            callback: Optional callback function(plate_text, processed_image) to call when processing completes
            
        Returns:
            task_id: A unique ID for this task, can be used to get results later if no callback provided
        """
        if vehicle_image is None:
            logger.warning("Empty image provided to submit_image")
            if callback:
                callback(None, None)
                return None
            return None
            
        task_id = self._get_next_task_id()
        self.task_queue.put((task_id, vehicle_image, save_path, callback))
        return task_id
        
    def get_result(self, task_id, timeout=None):
        """
        Get the result of a previously submitted image processing task.
        
        Args:
            task_id: The task ID returned from submit_image
            timeout: Maximum time to wait for the result (None = wait forever)
            
        Returns:
            (plate_text, processed_image) or None if timeout or task not found
        """
        if task_id is None:
            return None, None
            
        end_time = time.time() + timeout if timeout else None
        
        while end_time is None or time.time() < end_time:
            with self.results_lock:
                if task_id in self.results:
                    result = self.results.pop(task_id)
                    return result
            
            # Small sleep to prevent tight loop
            time.sleep(0.01)
            
        return None, None
