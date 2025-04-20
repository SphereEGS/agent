import cv2
import os
import time
import threading
import queue
from .config import logger, ALLOW_FALLBACK, FRAME_WIDTH, FRAME_HEIGHT

class InputStream:
    """
    Camera class to handle video input from different sources (RTSP, webcam, video file)
    """
    
    def __init__(self, camera_url=None, camera_id="main"):
        """
        Initialize camera with the specified input source
        
        Args:
            camera_url: URL/path to camera stream, overrides config if provided
            camera_id: Unique identifier for this camera stream
        """
        self.cap = None
        self.frame_count = 0
        self.last_frame = None
        self.source = None
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        self.fps = 100  # Default FPS
        self.last_successful_read_time = 0
        self.camera_id = camera_id
        self.camera_url = camera_url
        
        # For threaded capture
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer up to 30 frames
        self.stop_event = threading.Event()
        self.capture_thread = None
        
        # Initialize stream with configured source
        logger.info(f"[CAMERA:{self.camera_id}] Initializing input stream with source: {self.camera_url}")
        self._connect_to_source()
        self._start_capture_thread()
        
    def _connect_to_source(self):
        """Connect to the video source with proper error handling."""
        try:
            # Clean the source string of any quotes or comments
            source = self.camera_url.strip() if self.camera_url else ""
            if source.startswith(('"', "'")):
                source = source[1:-1]
            if '#' in source:
                source = source.split('#')[0].strip()
            
            logger.info(f"[CAMERA:{self.camera_id}] Attempting to connect to: {source}")
            
            # Set RTSP transport to TCP for better reliability
            if source.startswith('rtsp://'):
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(source)
            
            # Wait for the connection to be established
            if not cap.isOpened():
                logger.error(f"[CAMERA:{self.camera_id}] Failed to connect to: {source}")
                if ALLOW_FALLBACK:
                    logger.info(f"[CAMERA:{self.camera_id}] Falling back to local webcam (index 0)")
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        raise RuntimeError("Failed to connect to webcam")
                else:
                    raise RuntimeError(f"Failed to connect to camera source: {source}")
            
            # Get the original frame size
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"[CAMERA:{self.camera_id}] Connected to camera with resolution: {width}x{height}")
            
            # Use the exact dimensions from configuration
            target_width = FRAME_WIDTH
            target_height = FRAME_HEIGHT
            
            # Store the resize dimensions - use exact target dimensions
            self.resize_dimensions = (target_width, target_height)
            
            logger.info(f"[CAMERA:{self.camera_id}] Resizing frames to: {target_width}x{target_height}")
            
            self.cap = cap
            self.source = source
            
            return cap
            
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error connecting to camera: {str(e)}")
            if ALLOW_FALLBACK:
                logger.info(f"[CAMERA:{self.camera_id}] Attempting fallback to webcam...")
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
                    logger.warning(f"[CAMERA:{self.camera_id}] Camera not initialized in thread, attempting to reconnect")
                    self.cap = self._connect_to_source()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue

                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f"[CAMERA:{self.camera_id}] Failed to read frame in thread, attempting reconnection")
                    self.cap = self._connect_to_source()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue
                
                # Resize the frame to the target dimensions directly
                if hasattr(self, 'resize_dimensions') and frame is not None:
                    # Check if frame is valid and has expected dimensions
                    if frame.size > 0 and len(frame.shape) == 3:
                        # Use INTER_LINEAR for consistent resizing
                        frame = cv2.resize(frame, self.resize_dimensions, interpolation=cv2.INTER_LINEAR)
                    else:
                        logger.warning(f"[CAMERA:{self.camera_id}] Received invalid frame, skipping resize")
                
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
                logger.error(f"[CAMERA:{self.camera_id}] Error in capture thread: {str(e)}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    def _start_capture_thread(self):
        """Start the frame capture thread."""
        if self.capture_thread is not None and self.capture_thread.is_alive():
            return  # Thread already running
            
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_thread_function, daemon=True)
        self.capture_thread.start()
        logger.info(f"[CAMERA:{self.camera_id}] Started capture thread")
    
    def read(self):
        """Read a frame from the frame queue."""
        try:
            # If capture thread is not running, restart it
            if self.capture_thread is None or not self.capture_thread.is_alive():
                logger.warning(f"[CAMERA:{self.camera_id}] Capture thread not running, restarting")
                self._start_capture_thread()
                
            # Try to get a frame from the queue with timeout
            try:
                ret, frame = self.frame_queue.get(timeout=1.0)
                return ret, frame
            except queue.Empty:
                logger.warning(f"[CAMERA:{self.camera_id}] No frames in queue, waiting for capture thread")
                # If no frames available after waiting, try direct capture as fallback
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        return ret, frame
                
                return False, None
                
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error reading frame: {str(e)}")
            if ALLOW_FALLBACK:
                logger.info(f"[CAMERA:{self.camera_id}] Attempting fallback to webcam...")
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
                
        logger.info(f"[CAMERA:{self.camera_id}] Released input stream and stopped capture thread")