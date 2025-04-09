import cv2
import os
import time
import threading
import queue
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
        self.fps = 100  # Default FPS
        self.last_successful_read_time = 0
        
        # For threaded capture
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer up to 30 frames
        self.stop_event = threading.Event()
        self.capture_thread = None
        
        # Initialize stream with configured source
        logger.info(f"[CAMERA] Initializing input stream with source: {CAMERA_URL}")
        self._connect_to_source()
        self._start_capture_thread()
        
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
            
            self.cap = cap
            self.source = source
            
            return cap
            
        except Exception as e:
            logger.error(f"Error connecting to camera: {str(e)}")
            if ALLOW_FALLBACK:
                logger.info("Attempting fallback to webcam...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise RuntimeError("Failed to connect to webcam")
                self.cap = cap
                self.source = "webcam"
                return cap
            raise

    def _capture_thread_function(self):
        """Thread function to continuously capture frames from the camera."""
        reconnect_delay = 1.0  # Initial reconnect delay in seconds
        max_reconnect_delay = 30.0  # Maximum reconnect delay

        while not self.stop_event.is_set():
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.warning("[CAMERA] Camera not initialized in thread, attempting to reconnect")
                    self.cap = self._connect_to_source()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue

                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("[CAMERA] Failed to read frame in thread, attempting reconnection")
                    self.cap = self._connect_to_source()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue
                
                # Resize the frame to the target dimensions while preserving aspect ratio
                if hasattr(self, 'resize_dimensions'):
                    frame = cv2.resize(frame, self.resize_dimensions, interpolation=cv2.INTER_AREA)
                
                # Reset reconnect delay on successful frame capture
                reconnect_delay = 1.0
                
                # Put the frame in the queue, if queue is full, remove oldest frame
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # Discard oldest frame
                        except queue.Empty:
                            pass
                    self.frame_queue.put((ret, frame), block=False)
                except queue.Full:
                    # If still full, just continue
                    pass
                
                self.last_successful_read_time = time.time()
                
            except Exception as e:
                logger.error(f"[CAMERA] Error in capture thread: {str(e)}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    def _start_capture_thread(self):
        """Start the frame capture thread."""
        if self.capture_thread is not None and self.capture_thread.is_alive():
            return  # Thread already running
            
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_thread_function, daemon=True)
        self.capture_thread.start()
        logger.info("[CAMERA] Started capture thread")
    
    def read(self):
        """Read a frame from the frame queue."""
        try:
            # If capture thread is not running, restart it
            if self.capture_thread is None or not self.capture_thread.is_alive():
                logger.warning("[CAMERA] Capture thread not running, restarting")
                self._start_capture_thread()
                
            # Try to get a frame from the queue with timeout
            try:
                ret, frame = self.frame_queue.get(timeout=1.0)
                return ret, frame
            except queue.Empty:
                logger.warning("[CAMERA] No frames in queue, waiting for capture thread")
                # If no frames available after waiting, try direct capture as fallback
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        return ret, frame
                
                return False, None
                
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
        Release the camera and stop the capture thread
        """
        self.stop_event.set()  # Signal the thread to stop
        
        # Wait for the thread to finish with timeout
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3.0)
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        logger.info("[CAMERA] Released input stream and stopped capture thread")
