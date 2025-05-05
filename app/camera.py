import cv2
import os
import time
import threading
import queue
import numpy as np
from .config import CAMERA_URLS, CAMERA_URL, logger, ALLOW_FALLBACK, FRAME_WIDTH, FRAME_HEIGHT

class InputStream:
    """
    Production-ready camera class optimized for reliable, real-time video streaming
    with specific fixes for H.264 decoding errors and thread synchronization issues.
    """
    
    def __init__(self, camera_id="main"):
        """
        Initialize camera with the configured input source
        
        Args:
            camera_id (str): The ID of the camera to use (default: "main")
        """
        self.cap = None
        self.frame_count = 0
        self.last_frame = None
        self.source = None
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        self.last_successful_read_time = 0
        self.camera_id = camera_id
        
        # Get the camera URL for the specified camera ID
        if camera_id in CAMERA_URLS:
            self.camera_url = CAMERA_URLS[camera_id]
        else:
            # Fallback to main camera if the specified ID is not found
            logger.warning(f"[CAMERA] Camera ID '{camera_id}' not found, using main camera")
            self.camera_id = "main"
            self.camera_url = CAMERA_URL  # For backward compatibility
        
        # For threaded capture - single frame queue to minimize latency
        self.frame_queue = queue.Queue(maxsize=3)  # Small queue size to minimize buffering
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.frame_lock = threading.Lock()  # Lock for thread safety
        
        # Resize dimensions
        self.resize_dimensions = None
        
        # H.264 error handling
        self.decode_errors = 0
        self.last_error_time = 0
        self.error_threshold = 10  # Max errors before reconnecting
        self.error_window = 5  # Time window for counting errors
        
        # Thread sync fix
        self._thread_initialized = False
        
        # Initialize stream with configured source
        logger.info(f"[CAMERA] Initializing robust input stream for camera {self.camera_id}")
        self._connect_to_source()
        self._start_capture_thread()
        
    def _connect_to_source(self):
        """Connect to the video source with H.264 error prevention"""
        try:
            # Clean the source string
            source = self.camera_url.strip()
            if source.startswith(('"', "'")):
                source = source[1:-1]
            if '#' in source:
                source = source.split('#')[0].strip()
            
            logger.info(f"[CAMERA:{self.camera_id}] Connecting to: {source}")
            
            # Release previous capture if it exists
            if self.cap is not None:
                try:
                    self.cap.release()
                    time.sleep(0.5)  # Give time for proper release
                except Exception:
                    pass
                self.cap = None
            
            # Determine if this is an RTSP source
            is_rtsp = source.startswith('rtsp://')
            
            # Set specific FFMPEG settings to prevent H.264 decoding errors
            if is_rtsp:
                # Critical: Configure FFMPEG to be more error tolerant
                ffmpeg_options = []
                ffmpeg_options.append("rtsp_transport;tcp")  # More reliable than UDP
                ffmpeg_options.append("stimeout;5000000")    # Socket timeout in microseconds
                ffmpeg_options.append("error_concealment;1") # Enable error concealment
                ffmpeg_options.append("enable_er;1")         # Enable error resilience
                ffmpeg_options.append("fflags;discardcorrupt")  # Discard corrupt frames
                ffmpeg_options.append("max_delay;500000")    # Maximum demuxing delay
                ffmpeg_options.append("framedrop;1")         # Enable frame dropping if needed
                
                # Join options with pipe separator
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = '|'.join(ffmpeg_options)
                
                # Use standard FFMPEG backend for RTSP - more stable than GStreamer for error handling
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
                # For webcams or other sources
                cap = cv2.VideoCapture(source)
            
            # Wait for connection
            if not cap.isOpened():
                logger.error(f"[CAMERA:{self.camera_id}] Failed to connect to: {source}")
                if ALLOW_FALLBACK:
                    logger.info(f"[CAMERA:{self.camera_id}] Falling back to local webcam (index 0)")
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        raise RuntimeError("Failed to connect to webcam")
                else:
                    raise RuntimeError(f"Failed to connect to camera source: {source}")
            
            # Configure capture properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer for lowest latency
            
            # Get the original frame size
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"[CAMERA:{self.camera_id}] Connected to camera with resolution: {width}x{height} @ {fps if fps > 0 else 'unknown'}fps")
            
            # Calculate the aspect ratio preserving dimensions
            aspect_ratio = width / height
            if aspect_ratio > (FRAME_WIDTH / FRAME_HEIGHT):
                # Image is wider than target
                new_width = FRAME_WIDTH
                new_height = int(FRAME_WIDTH / aspect_ratio)
            else:
                # Image is taller than target
                new_height = FRAME_HEIGHT
                new_width = int(FRAME_HEIGHT * aspect_ratio)
            
            # Ensure dimensions are even numbers (required by some OpenCV operations)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            logger.info(f"[CAMERA:{self.camera_id}] Resizing frames to: {new_width}x{new_height}")
            
            # Store the resize dimensions
            self.resize_dimensions = (new_width, new_height)
            
            self.cap = cap
            self.source = source
            
            # Reset error counters
            self.decode_errors = 0
            self.last_error_time = 0
            
            return cap
            
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error connecting to camera: {str(e)}")
            if ALLOW_FALLBACK:
                logger.info(f"[CAMERA:{self.camera_id}] Attempting fallback to webcam...")
                try:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        raise RuntimeError("Failed to connect to webcam")
                    self.cap = cap
                    self.source = "webcam"
                    return cap
                except Exception as webcam_e:
                    logger.error(f"[CAMERA:{self.camera_id}] Webcam fallback failed: {str(webcam_e)}")
            raise

    def _capture_thread_function(self):
        """Thread function to continuously capture frames with error handling"""
        reconnect_delay = 1.0  # Initial reconnect delay in seconds
        max_reconnect_delay = 5.0  # Maximum reconnect delay
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        # Set thread initialized flag
        self._thread_initialized = True

        while not self.stop_event.is_set():
            try:
                # Check connection status
                if not self.cap or not self.cap.isOpened():
                    logger.warning(f"[CAMERA:{self.camera_id}] Connection lost, reconnecting...")
                    self.cap = self._connect_to_source()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                    continue

                # Read frame with explicit timeout handling
                frame_read_success = False
                frame = None
                
                # Use lock for thread safety during frame reading
                with self.frame_lock:
                    try:
                        ret, frame = self.cap.read()
                        if ret and frame is not None and len(frame.shape) == 3:
                            frame_read_success = True
                            consecutive_errors = 0
                        else:
                            consecutive_errors += 1
                    except Exception as e:
                        logger.error(f"[CAMERA:{self.camera_id}] Frame read error: {str(e)}")
                        consecutive_errors += 1
                
                # Handle frame read failures
                if not frame_read_success:
                    if "[h264" in str(consecutive_errors) or "error while decoding" in str(consecutive_errors):
                        # H.264 specific error - increment counter
                        current_time = time.time()
                        if current_time - self.last_error_time > self.error_window:
                            # Reset counter for new time window
                            self.decode_errors = 1
                            self.last_error_time = current_time
                        else:
                            self.decode_errors += 1
                        
                        # If too many errors in the window, reconnect
                        if self.decode_errors >= self.error_threshold:
                            logger.warning(f"[CAMERA:{self.camera_id}] Too many H.264 decode errors, reconnecting...")
                            self.cap = self._connect_to_source()
                            time.sleep(reconnect_delay)
                            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                            continue
                    
                    # Too many consecutive errors - reconnect
                    if consecutive_errors >= max_consecutive_errors:
                        logger.warning(f"[CAMERA:{self.camera_id}] {consecutive_errors} consecutive frame failures, reconnecting...")
                        self.cap = self._connect_to_source()
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                        consecutive_errors = 0
                        continue
                    
                    # Skip this iteration but don't reconnect yet
                    time.sleep(0.01)
                    continue
                
                # Reset reconnect delay on success
                reconnect_delay = 1.0
                
                # Resize the frame if needed
                if self.resize_dimensions and frame is not None:
                    try:
                        # Ensure the frame dimensions are valid before resizing
                        h, w = frame.shape[:2]
                        if h > 0 and w > 0:
                            # Crop to even dimensions if needed
                            if h % 2 == 1 or w % 2 == 1:
                                frame = frame[:h - (h % 2), :w - (w % 2)]
                            
                            # Use INTER_NEAREST for fastest resize with acceptable quality
                            frame = cv2.resize(frame, self.resize_dimensions, interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        logger.warning(f"[CAMERA:{self.camera_id}] Error during resize: {str(e)}")
                
                # Put the frame in the queue - prioritize real-time by clearing queue if full
                current_time = time.time()
                try:
                    # Clear queue if full to ensure we only keep the newest frame
                    if self.frame_queue.full():
                        while not self.frame_queue.empty():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                break
                    
                    # Add new frame
                    self.frame_queue.put((ret, frame, current_time), block=False)
                except queue.Full:
                    # If still full (unlikely after clearing), just continue
                    pass
                
                self.last_successful_read_time = current_time
                self.frame_count += 1
                
                # Small sleep to prevent thread from consuming too much CPU
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"[CAMERA:{self.camera_id}] Error in capture thread: {str(e)}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

    def _start_capture_thread(self):
        """Start the frame capture thread with proper synchronization"""
        # Don't start if already running
        if self.capture_thread is not None and self.capture_thread.is_alive():
            return
            
        # Clear stop event and reset thread initialized flag
        self.stop_event.clear()
        self._thread_initialized = False
        
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        # Create and start thread
        self.capture_thread = threading.Thread(
            target=self._capture_thread_function, 
            name=f"Camera-{self.camera_id}-Thread",
            daemon=True
        )
        self.capture_thread.start()
        
        # Wait for thread initialization to complete
        timeout = 5.0  # seconds
        start_time = time.time()
        while not self._thread_initialized and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self._thread_initialized:
            logger.warning(f"[CAMERA:{self.camera_id}] Thread initialization timeout")
        
        logger.info(f"[CAMERA:{self.camera_id}] Started capture thread")
    
    def read(self):
        """Read the most recent frame with error handling"""
        # Default return values
        ret, frame = False, None
        
        try:
            # Check if capture thread is running
            if self.capture_thread is None or not self.capture_thread.is_alive():
                logger.warning(f"[CAMERA:{self.camera_id}] Capture thread not running, restarting")
                self._start_capture_thread()
                time.sleep(0.1)  # Small delay to let thread initialize
            
            # Try to get frame from queue with short timeout
            try:
                # Always get the newest frame for real-time performance
                newest_frame = None
                newest_timestamp = 0
                
                # Empty the queue and find newest frame
                while True:
                    try:
                        current_ret, current_frame, timestamp = self.frame_queue.get(block=False)
                        if timestamp > newest_timestamp and current_frame is not None:
                            newest_ret = current_ret
                            newest_frame = current_frame
                            newest_timestamp = timestamp
                    except queue.Empty:
                        break
                
                # If we found a frame, use it
                if newest_frame is not None:
                    ret, frame = newest_ret, newest_frame
                    self.last_frame = frame  # Cache the frame
                    return ret, frame
            except Exception as e:
                logger.error(f"[CAMERA:{self.camera_id}] Error getting frame from queue: {str(e)}")
            
            # If no frame from queue, try direct capture as fallback
            if self.cap and self.cap.isOpened():
                with self.frame_lock:
                    try:
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            # Resize if needed
                            if self.resize_dimensions:
                                frame = cv2.resize(frame, self.resize_dimensions, interpolation=cv2.INTER_NEAREST)
                            self.last_frame = frame  # Cache the frame
                            return ret, frame
                    except Exception as e:
                        logger.error(f"[CAMERA:{self.camera_id}] Error in direct capture: {str(e)}")
            
            # Last resort: return the last successfully captured frame
            if self.last_frame is not None:
                logger.debug(f"[CAMERA:{self.camera_id}] Returning last cached frame")
                return True, self.last_frame
                
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error in read: {str(e)}")
            
        # Fallback to last frame if available, or return None
        if self.last_frame is not None:
            return True, self.last_frame
            
        return False, None
    
    def get_frame_rate(self):
        """Calculate the effective frame rate based on frame count"""
        current_time = time.time()
        if hasattr(self, 'last_fps_time') and hasattr(self, 'last_frame_count'):
            elapsed = current_time - self.last_fps_time
            if elapsed > 0:
                frames = self.frame_count - self.last_frame_count
                fps = frames / elapsed
                self.last_fps_time = current_time
                self.last_frame_count = self.frame_count
                return fps
        
        # Initialize if first call
        self.last_fps_time = current_time
        self.last_frame_count = self.frame_count
        return 0
    
    def release(self):
        """Safely release all resources"""
        # Set stop event first
        self.stop_event.set()
        
        # Wait for thread to stop
        if self.capture_thread is not None:
            try:
                self.capture_thread.join(timeout=1.0)
            except Exception:
                pass
            
        # Release capture device with lock
        with self.frame_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        
        logger.info(f"[CAMERA:{self.camera_id}] Released all resources")