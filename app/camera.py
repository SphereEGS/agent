import cv2
import os
import time
import threading
import queue
import numpy as np
from .config import CAMERA_URLS, CAMERA_URL, logger, ALLOW_FALLBACK, FRAME_WIDTH, FRAME_HEIGHT

class InputStream:
    """
    Optimized camera class to handle video input from different sources with minimal latency
    """
    
    def __init__(self, camera_id="main"):
        """
        Initialize camera with the configured input source optimized for low latency
        
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
        
        # For threaded capture - reduce queue size for lower latency
        self.frame_queue = queue.Queue(maxsize=5)  # Reduced from 30 to minimize buffering delay
        self.stop_event = threading.Event()
        self.capture_thread = None
        
        # Performance optimization flags
        self.enable_hw_acceleration = True
        self.precomputed_resize = None
        self.resize_map_x = None
        self.resize_map_y = None
        self.frame_timestamp = 0
        
        # Network settings
        self.network_buffer_size = 0  # 0 = minimal buffering
        self.rtsp_transport = 'tcp'  # TCP for more reliable streaming
        self.connection_timeout = 5.0
        self.last_reconnect_attempt = 0
        
        # Initialize stream with configured source
        logger.info(f"[CAMERA] Initializing optimized input stream for camera {self.camera_id} with source: {self.camera_url}")
        self._connect_to_source()
        self._start_capture_thread()
        
    def _setup_hw_acceleration(self, cap):
        """Configure hardware acceleration if available"""
        if not self.enable_hw_acceleration:
            return
            
        # Try to enable hardware acceleration for decoding
        try:
            # NVIDIA GPU acceleration
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info(f"[CAMERA:{self.camera_id}] CUDA-enabled device found, enabling GPU acceleration")
                # Set preference for CUDA
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                cap.set(cv2.CAP_PROP_HW_DEVICE, 0)  # Use first GPU
            else:
                # Other hardware acceleration options
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                logger.info(f"[CAMERA:{self.camera_id}] Requested hardware acceleration")
        except Exception as e:
            logger.warning(f"[CAMERA:{self.camera_id}] Could not enable hardware acceleration: {str(e)}")
            
    def _setup_resize_maps(self, original_width, original_height, target_width, target_height):
        """Precalculate resize mapping for faster frame resizing"""
        try:
            # Calculate aspect ratio preserving dimensions
            aspect_ratio = original_width / original_height
            
            if aspect_ratio > (target_width / target_height):
                # Image is wider than target
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # Image is taller than target
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # Ensure dimensions are even numbers
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            # Store the resize dimensions
            self.resize_dimensions = (new_width, new_height)
            
            # Create remapping matrices for faster resizing
            map_x, map_y = cv2.initUndistortRectifyMap(
                np.eye(3), None, None, np.eye(3), 
                (new_width, new_height), cv2.CV_32FC1)
                
            self.resize_map_x = map_x
            self.resize_map_y = map_y
            
            logger.info(f"[CAMERA:{self.camera_id}] Optimized resize maps created: {new_width}x{new_height}")
            
        except Exception as e:
            logger.warning(f"[CAMERA:{self.camera_id}] Error creating resize maps: {str(e)}")
            self.resize_map_x = None
            self.resize_map_y = None

    def _connect_to_source(self):
        """Connect to the video source with optimizations for low latency"""
        try:
            # Clean the source string
            source = self.camera_url.strip()
            if source.startswith(('"', "'")):
                source = source[1:-1]
            if '#' in source:
                source = source.split('#')[0].strip()
            
            logger.info(f"[CAMERA:{self.camera_id}] Connecting to: {source}")
            
            # Determine if this is an RTSP source
            is_rtsp = source.startswith('rtsp://')
            is_http = source.startswith(('http://', 'https://'))
            
            # Set up connection options to minimize latency
            if is_rtsp:
                # RTSP-specific optimizations
                transport_option = f"rtsp_transport;{self.rtsp_transport}"
                buffer_option = f"buffer_size;{self.network_buffer_size}"
                max_delay_option = "max_delay;0"  # Minimize internal buffering
                
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'{transport_option}|{buffer_option}|{max_delay_option}'
                
                # Additional RTSP-specific settings via GStreamer if available
                try:
                    # Try to use GStreamer pipeline for better performance with RTSP
                    gst_pipeline = (
                        f"rtspsrc location={source} latency=0 buffer-mode=auto ! "
                        f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
                    )
                    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                    
                    if not cap.isOpened():
                        # Fall back to standard OpenCV RTSP handling
                        logger.info(f"[CAMERA:{self.camera_id}] GStreamer pipeline failed, falling back to standard RTSP")
                        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                except Exception:
                    # If GStreamer is not available, use standard OpenCV
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            elif is_http:
                # HTTP-specific optimizations (for IP cameras with HTTP stream)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                # Set minimal buffering
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                # Local camera or other sources
                cap = cv2.VideoCapture(source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Check if opened successfully
            if not cap.isOpened():
                logger.error(f"[CAMERA:{self.camera_id}] Failed to connect to: {source}")
                if ALLOW_FALLBACK:
                    logger.info(f"[CAMERA:{self.camera_id}] Falling back to local webcam (index 0)")
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened():
                        raise RuntimeError("Failed to connect to webcam")
                else:
                    raise RuntimeError(f"Failed to connect to camera source: {source}")
            
            # Try to enable hardware acceleration
            self._setup_hw_acceleration(cap)
            
            # Get and set camera properties for optimal performance
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"[CAMERA:{self.camera_id}] Connected to camera with resolution: {original_width}x{original_height} @ {original_fps}fps")
            
            # Setup optimized resize maps
            self._setup_resize_maps(original_width, original_height, FRAME_WIDTH, FRAME_HEIGHT)
            
            # Minimize frame buffering in OpenCV
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.cap = cap
            self.source = source
            self.original_dimensions = (original_width, original_height)
            
            # Force an initial read to fully establish the connection
            ret, _ = cap.read()
            if not ret:
                logger.warning(f"[CAMERA:{self.camera_id}] Initial frame read failed, but continuing...")
            
            return cap
            
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error connecting to camera: {str(e)}")
            if ALLOW_FALLBACK and time.time() - self.last_reconnect_attempt > 5:
                self.last_reconnect_attempt = time.time()
                logger.info(f"[CAMERA:{self.camera_id}] Attempting fallback to webcam...")
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    raise RuntimeError("Failed to connect to webcam")
                self.cap = cap
                self.source = "webcam"
                return cap
            raise

    def _optimized_resize(self, frame):
        """Perform optimized frame resizing"""
        if frame is None:
            return None
            
        try:
            # Check if frame dimensions match our expected input
            h, w = frame.shape[:2]
            
            # Ensure dimensions are even
            if h % 2 == 1 or w % 2 == 1:
                h_new = h - (h % 2)
                w_new = w - (w % 2)
                frame = frame[:h_new, :w_new]
            
            # If we have precomputed resize maps, use them for faster resizing
            if self.resize_map_x is not None and self.resize_map_y is not None:
                try:
                    # Use remap for potentially faster resizing
                    resized = cv2.remap(frame, self.resize_map_x, self.resize_map_y, cv2.INTER_LINEAR)
                    return resized
                except Exception:
                    # Fall back to standard resize if remap fails
                    pass
            
            # Standard resize as fallback
            return cv2.resize(frame, self.resize_dimensions, interpolation=cv2.INTER_LINEAR)
            
        except Exception as e:
            logger.warning(f"[CAMERA:{self.camera_id}] Error in optimized resize: {str(e)}")
            # Try standard resize as last resort
            try:
                return cv2.resize(frame, self.resize_dimensions, interpolation=cv2.INTER_LINEAR)
            except Exception:
                return frame  # Return original frame if all resize attempts fail

    def _capture_thread_function(self):
        """Thread function to continuously capture frames with minimal latency"""
        reconnect_delay = 1.0  # Initial reconnect delay in seconds
        max_reconnect_delay = 10.0  # Maximum reconnect delay (reduced from 30)
        frame_drop_threshold = 0.5  # Drop frames older than this many seconds

        while not self.stop_event.is_set():
            try:
                # Check if connection is active
                if not self.cap or not self.cap.isOpened():
                    logger.warning(f"[CAMERA:{self.camera_id}] Connection lost, reconnecting...")
                    self.cap = self._connect_to_source()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                    continue

                # Capture frame with timeout monitoring
                start_time = time.time()
                ret, frame = self.cap.read()
                read_time = time.time() - start_time
                
                # Alert if read is taking too long
                if read_time > 0.1:  # More than 100ms is concerning for real-time
                    logger.warning(f"[CAMERA:{self.camera_id}] Slow frame read: {read_time:.3f}s")
                
                if not ret or frame is None:
                    logger.warning(f"[CAMERA:{self.camera_id}] Failed to read frame, reconnecting...")
                    self.cap = self._connect_to_source()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                    continue
                
                # Reset reconnect delay on successful capture
                reconnect_delay = 1.0
                
                # Resize the frame using optimized method
                if hasattr(self, 'resize_dimensions'):
                    start_resize = time.time()
                    frame = self._optimized_resize(frame)
                    resize_time = time.time() - start_resize
                    
                    # Log if resize is taking too long
                    if resize_time > 0.05:
                        logger.warning(f"[CAMERA:{self.camera_id}] Slow resize: {resize_time:.3f}s")
                
                # Store capture timestamp
                current_time = time.time()
                
                # Real-time priority: If queue is not empty and we're running behind, 
                # clear older frames to prioritize newest
                if self.frame_queue.qsize() > 2:
                    while not self.frame_queue.empty():
                        try:
                            # Try to remove all but the most recent frame
                            _, _, frame_time = self.frame_queue.get_nowait()
                            if current_time - frame_time < frame_drop_threshold:
                                # Keep this frame, it's recent enough
                                break
                        except queue.Empty:
                            break
                
                # Put frame in queue with timestamp
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # Remove oldest frame
                        except queue.Empty:
                            pass
                    self.frame_queue.put((ret, frame, current_time), block=False)
                except queue.Full:
                    # If still full, just continue
                    pass
                
                self.last_successful_read_time = current_time
                self.frame_count += 1
                
                # Sleep a tiny amount to prevent CPU overload
                # but small enough not to affect real-time performance
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"[CAMERA:{self.camera_id}] Error in capture thread: {str(e)}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

    def _start_capture_thread(self):
        """Start the frame capture thread with high priority"""
        if self.capture_thread is not None and self.capture_thread.is_alive():
            return  # Thread already running
            
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_thread_function, daemon=True)
        self.capture_thread.name = f"Camera-{self.camera_id}-CaptureThread"
        
        # Start thread
        self.capture_thread.start()
        
        # Try to increase thread priority if possible
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            if hasattr(process, "nice"):
                process.nice(psutil.HIGH_PRIORITY_CLASS)
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"[CAMERA:{self.camera_id}] Could not adjust thread priority: {str(e)}")
            
        logger.info(f"[CAMERA:{self.camera_id}] Started optimized capture thread")
    
    def read(self):
        """Read the most recent frame with minimal latency"""
        try:
            # If capture thread is not running, restart it
            if self.capture_thread is None or not self.capture_thread.is_alive():
                logger.warning(f"[CAMERA:{self.camera_id}] Capture thread not running, restarting")
                self._start_capture_thread()
                
            # For real-time applications, always get the newest frame
            newest_frame = None
            newest_timestamp = 0
            
            # Empty the queue to get to the newest frame
            frames_in_queue = []
            while True:
                try:
                    ret, frame, timestamp = self.frame_queue.get(block=False)
                    frames_in_queue.append((ret, frame, timestamp))
                    if timestamp > newest_timestamp:
                        newest_frame = frame
                        newest_timestamp = timestamp
                except queue.Empty:
                    break
            
            # Put all frames except the newest back in the queue
            # (This preserves frame history if needed)
            if len(frames_in_queue) > 1:
                for ret, frame, timestamp in frames_in_queue:
                    if timestamp != newest_timestamp:
                        try:
                            self.frame_queue.put((ret, frame, timestamp), block=False)
                        except queue.Full:
                            pass
            
            # If we found a frame in the queue, return it
            if newest_frame is not None:
                self.last_frame = newest_frame
                return True, newest_frame
                
            # If queue was empty, try direct capture as immediate fallback
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None and hasattr(self, 'resize_dimensions'):
                    frame = self._optimized_resize(frame)
                    self.last_frame = frame
                    return ret, frame
                    
            # Last resort: return the last successfully captured frame if available
            if self.last_frame is not None:
                logger.warning(f"[CAMERA:{self.camera_id}] Returning last cached frame")
                return True, self.last_frame
                
            return False, None
            
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error reading frame: {str(e)}")
            if self.last_frame is not None:
                return True, self.last_frame
            return False, None
    
    def release(self):
        """Release resources and stop capture thread"""
        self.stop_event.set()
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)  # Shorter timeout for faster shutdown
        if self.cap is not None:
            self.cap.release()
        logger.info(f"[CAMERA:{self.camera_id}] Released camera resources")