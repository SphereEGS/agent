import cv2
import os
import time
from .config import CAMERA_URL, logger, ALLOW_FALLBACK, FRAME_WIDTH, FRAME_HEIGHT

class InputStream:
    """
    Camera class to handle video input from different sources (RTSP, webcam, video file)
    """
    
    def __init__(self):
        """
        Initialize camera with the configured input source
        """
        self.cap = None
        self.frame_count = 0
        self.last_frame = None
        self.source = None
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        self.fps = 10  # Default FPS
        self.last_successful_read_time = 0
        
        # Initialize stream with configured source
        logger.info(f"[CAMERA] Initializing input stream with source: {CAMERA_URL}")
        self._connect_to_source()
        
    def _connect_to_source(self):
        """Connect to the video source with proper error handling."""
        try:
            # Clean the source string of any quotes or comments
            source = CAMERA_URL.strip()
            if source.startswith(('"', "'")):
                source = source[1:-1]
            if '#' in source:
                source = source.split('#')[0].strip()
            
            logger.info(f"Attempting to connect to: {source}")
            
            # Set RTSP transport to TCP for better reliability
            if source.startswith('rtsp://'):
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(source)
            
            # Wait for the connection to be established
            if not cap.isOpened():
                logger.error(f"Failed to connect to: {source}")
                if ALLOW_FALLBACK:
                    logger.info("Falling back to local webcam (index 0)")
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        raise RuntimeError("Failed to connect to webcam")
                else:
                    raise RuntimeError(f"Failed to connect to camera source: {source}")
            
            # Get the original frame size
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Connected to camera with resolution: {width}x{height}")
            
            # Use the window size from configuration
            target_width = FRAME_WIDTH
            target_height = FRAME_HEIGHT
            
            # Calculate the aspect ratio preserving dimensions
            aspect_ratio = width / height
            if aspect_ratio > (target_width / target_height):
                # Image is wider than target
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # Image is taller than target
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # Ensure dimensions are even numbers (required by some OpenCV operations)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            logger.info(f"Resizing frames to: {new_width}x{new_height}")
            
            # Store the resize dimensions
            self.resize_dimensions = (new_width, new_height)
            
            return cap
            
        except Exception as e:
            logger.error(f"Error connecting to camera: {str(e)}")
            if ALLOW_FALLBACK:
                logger.info("Attempting fallback to webcam...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise RuntimeError("Failed to connect to webcam")
                return cap
            raise
    
    def read(self):
        """Read a frame from the camera with error handling and reconnection logic."""
        try:
            if not self.cap or not self.cap.isOpened():
                logger.warning("[CAMERA] Camera not initialized, attempting to reconnect")
                self.cap = self._connect_to_source()
            
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("[CAMERA] Failed to read frame, attempting reconnection")
                self.cap = self._connect_to_source()
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Failed to read frame after reconnection")
            
            # Resize the frame to the target dimensions while preserving aspect ratio
            if hasattr(self, 'resize_dimensions'):
                frame = cv2.resize(frame, self.resize_dimensions, interpolation=cv2.INTER_AREA)
            
            self.last_successful_read_time = time.time()
            return ret, frame
            
        except Exception as e:
            logger.error(f"[CAMERA] Error reading frame: {str(e)}")
            if ALLOW_FALLBACK:
                logger.info("[CAMERA] Attempting fallback to webcam...")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise RuntimeError("Failed to connect to webcam")
                return self.read()
            raise
    
    def release(self):
        """
        Release the camera
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("[CAMERA] Released input stream")
