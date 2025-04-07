import cv2
import os
import time
from .config import INPUT_SOURCE, logger, ALLOW_FALLBACK, FRAME_WIDTH, FRAME_HEIGHT

class InputStream:
    def __init__(self):
        """Initialize video capture from camera source"""
        self.cap = None
        self.is_webcam = False
        self.source_type = "unknown"
        self.frame_count = 0
        
        try:
            input_source = INPUT_SOURCE
            logger.info(f"[CAMERA] Initializing input source: {input_source}")
            
            # Handle webcam sources
            if input_source in ["0", "1"] or input_source == 0 or input_source == 1:
                self._init_webcam(input_source)
            
            # Handle video file sources
            elif isinstance(input_source, str) and os.path.exists(input_source):
                self._init_video_file(input_source)
            
            # Handle CCTV stream sources
            elif isinstance(input_source, str) and (input_source.startswith("rtsp://") or 
                                                   input_source.startswith("http://") or 
                                                   input_source.startswith("https://")):
                self._init_cctv(input_source)
            
            # Handle unknown source types
            else:
                self._init_unknown(input_source)
            
            # Final checks
            if not self.cap or not self.cap.isOpened():
                logger.error(f"[CAMERA] Failed to open any video source")
                raise Exception(f"Failed to open any video source")
            
            # Initialize display window
            cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
            
            # Test frame reading
            self._test_frame_reading()
            
            # Log source details
            self._log_source_details()
            
        except Exception as e:
            logger.error(f"[CAMERA] Initialization error: {str(e)}")
            if self.cap:
                self.cap.release()
            raise
    
    def _init_webcam(self, source):
        """Initialize webcam source"""
        self.is_webcam = True
        self.source_type = "webcam"
        camera_index = int(source) if isinstance(source, str) else source
        logger.info(f"[CAMERA] Opening webcam with index {camera_index}")
        self.cap = cv2.VideoCapture(camera_index)
        
        if self.cap.isOpened():
            logger.info(f"[CAMERA] Webcam opened, setting properties: {FRAME_WIDTH}x{FRAME_HEIGHT}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            time.sleep(1)  # Allow camera to initialize
        else:
            logger.error(f"[CAMERA] Failed to open webcam with index {camera_index}")
            raise Exception(f"Failed to open webcam with index {camera_index}")
    
    def _init_video_file(self, source):
        """Initialize video file source"""
        self.source_type = "video_file"
        logger.info(f"[CAMERA] Opening video file: {source}")
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logger.error(f"[CAMERA] Failed to open video file: {source}")
            raise Exception(f"Failed to open video file: {source}")
    
    def _init_cctv(self, source):
        """Initialize CCTV source"""
        self.source_type = "cctv"
        logger.info(f"[CAMERA] Opening CCTV stream: {source}")
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            logger.error(f"[CAMERA] Failed to connect to CCTV stream: {source}")
            if ALLOW_FALLBACK:
                logger.warning(f"[CAMERA] Attempting fallback to webcam (index 0)")
                self.cap = cv2.VideoCapture(0)
                self.source_type = "webcam (fallback)"
                self.is_webcam = True
                time.sleep(1)
                if not self.cap.isOpened():
                    logger.error(f"[CAMERA] Webcam fallback also failed")
                    raise Exception(f"Failed to connect to CCTV and webcam fallback also failed")
            else:
                logger.error(f"[CAMERA] Fallback disabled, not falling back to webcam")
                raise Exception(f"Failed to connect to CCTV stream and fallback is disabled")
        else:
            logger.info(f"[CAMERA] Successfully connected to CCTV stream")
    
    def _init_unknown(self, source):
        """Initialize unknown source"""
        logger.info(f"[CAMERA] Opening unknown camera source: {source}")
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            logger.error(f"[CAMERA] Failed to open camera source: {source}")
            if ALLOW_FALLBACK:
                logger.warning(f"[CAMERA] Attempting fallback to webcam (index 0)")
                self.cap = cv2.VideoCapture(0)
                self.source_type = "webcam (fallback)"
                self.is_webcam = True
                if not self.cap.isOpened():
                    logger.error(f"[CAMERA] Webcam fallback also failed")
                    raise Exception(f"Failed to open source and webcam fallback also failed")
            else:
                logger.error(f"[CAMERA] Fallback disabled, not falling back to webcam")
                raise Exception(f"Failed to open video source and fallback is disabled")
    
    def _test_frame_reading(self):
        """Test if frames can be read from source"""
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            logger.error(f"[CAMERA] Could not read frames from {self.source_type}")
            raise Exception(f"Could not read frames from {self.source_type}")
    
    def _log_source_details(self):
        """Log details about the video source"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if self.is_webcam:
            logger.info(f"[CAMERA] Webcam initialized: {width}x{height}, {fps} FPS")
        elif self.source_type == "cctv":
            logger.info(f"[CAMERA] CCTV stream initialized: {width}x{height}, {fps} FPS")
        else:
            logger.info(f"[CAMERA] Video source initialized: {width}x{height}, {fps} FPS")
    
    def read(self):
        """Read a frame from the stream"""
        if not self.cap or not self.cap.isOpened():
            logger.error("[CAMERA] Camera is not initialized or has been closed")
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            logger.warning(f"[CAMERA] Failed to read frame from {self.source_type}")
            return None
        
        # Log frame reading periodically
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            logger.debug(f"[CAMERA] Read {self.frame_count} frames from {self.source_type}")
        
        return frame
    
    def isOpened(self):
        """Check if stream is open"""
        is_open = self.cap is not None and self.cap.isOpened()
        if not is_open:
            logger.warning(f"[CAMERA] {self.source_type} is no longer open")
        return is_open
    
    def stop(self):
        """Release the video capture and cleanup"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info(f"[CAMERA] {self.source_type.capitalize()} released and windows closed")
        else:
            logger.info(f"[CAMERA] No active camera to release")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.stop()
