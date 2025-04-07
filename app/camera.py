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
        self.fps = 30  # Default FPS
        self.last_successful_read_time = 0
        
        # Initialize stream with configured source
        logger.info(f"[CAMERA] Initializing input stream with source: {CAMERA_URL}")
        self._connect_to_source(CAMERA_URL)
        
    def _connect_to_source(self, source):
        """
        Connect to a camera source (RTSP, webcam, or video file)
        
        Args:
            source: Camera source (RTSP URL, webcam index, or video file path)
        """
        # Clean the source string - remove quotes and comments that might be in the string
        if isinstance(source, str):
            # Remove quotes and comments if present (from environment variables)
            source = source.strip()
            if source.startswith('"') and source.endswith('"'):
                source = source[1:-1].strip()
            # Handle quotes within the string
            if '"' in source:
                source = source.replace('"', '')
            # Remove any comments
            if '#' in source:
                source = source.split('#')[0].strip()
        
        logger.info(f"[CAMERA] Connecting to cleaned source: {source}")
        
        try:
            logger.info(f"[CAMERA] Connecting to source: {source}")
            
            # Handle different source types
            if isinstance(source, str) and source.startswith('rtsp'):
                # For RTSP streams, use a specific configuration
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                # Set timeout for RTSP connections (in milliseconds)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
                
                # Check if connection was successful
                if not self.cap.isOpened():
                    raise Exception(f"Failed to connect to RTSP stream: {source}")
                    
                logger.info(f"[CAMERA] Successfully connected to RTSP stream")
                self.source = "rtsp"
                
            elif source == "0" or source == 0 or (isinstance(source, str) and source.isdigit()):
                # For webcam - convert string to integer if needed
                index = 0 if source == "0" else int(source)
                self.cap = cv2.VideoCapture(index)
                
                if not self.cap.isOpened():
                    raise Exception(f"Failed to open webcam at index {index}")
                    
                logger.info(f"[CAMERA] Successfully connected to webcam at index {index}")
                self.source = "webcam"
                
            else:
                # For video files
                if isinstance(source, str) and not os.path.exists(source):
                    raise Exception(f"Input file not found: {source}")
                    
                self.cap = cv2.VideoCapture(source)
                
                if not self.cap.isOpened():
                    raise Exception(f"Failed to open video file: {source}")
                    
                logger.info(f"[CAMERA] Successfully opened video file: {source}")
                self.source = "video"
                
            # Set frame size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            
            # Get the first frame
            ret, self.last_frame = self.cap.read()
            if not ret:
                raise Exception("Failed to read frame from source")
                
            # Set dimensions from the actual frame
            self.height, self.width = self.last_frame.shape[:2]
            self.last_successful_read_time = time.time()
            logger.info(f"[CAMERA] Stream initialized with resolution: {self.width}x{self.height}")
            
        except Exception as e:
            logger.error(f"[CAMERA] Error initializing stream from source {source}: {str(e)}")
            self.cap = None
            
            # Try fallback to webcam if enabled and we're not already trying webcam
            if ALLOW_FALLBACK and source != 0 and source != "0":
                logger.warning("[CAMERA] Attempting fallback to default webcam")
                try:
                    self._connect_to_source(0)
                except Exception as fallback_error:
                    logger.error(f"[CAMERA] Fallback to webcam also failed: {str(fallback_error)}")
                    raise Exception("Both primary and fallback camera connections failed")
            else:
                logger.error(f"[CAMERA] No fallback attempted as ALLOW_FALLBACK is {ALLOW_FALLBACK}")
                raise
    
    def read(self):
        """
        Read a frame from the camera
        
        Returns:
            tuple: (success, frame)
        """
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            self.last_frame = frame
            self.last_successful_read_time = time.time()
            return True, frame
        else:
            # If we haven't been able to read a frame for more than 5 seconds, attempt reconnection
            current_time = time.time()
            if current_time - self.last_successful_read_time > 5:
                logger.warning(f"[CAMERA] No frames read for 5 seconds, attempting reconnection")
                try:
                    self._connect_to_source(CAMERA_URL)
                    return self.read()
                except Exception as e:
                    logger.error(f"[CAMERA] Reconnection failed: {str(e)}")
            
            return False, self.last_frame
    
    def release(self):
        """
        Release the camera
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("[CAMERA] Released input stream")
