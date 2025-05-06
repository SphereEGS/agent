import cv2
import os
import time
import threading
import queue
import numpy as np
from .config import CAMERA_URLS, CAMERA_URL, logger, ALLOW_FALLBACK, FRAME_WIDTH, FRAME_HEIGHT

# Check for Jetson Nano
try:
    import jetson.utils
    import jetson.inference
    JETSON_AVAILABLE = True
    logger.info("[CAMERA] Jetson modules available, enabling GPU acceleration")
except ImportError:
    JETSON_AVAILABLE = False
    logger.info("[CAMERA] Jetson modules not available, using CPU processing")

class InputStream:
    """
    High-performance camera class optimized for minimal-latency video streaming
    with RTSP acceleration and pipeline optimizations for real-time processing.
    Now with Jetson Nano GPU acceleration when available.
    """

    def __init__(self, camera_id="main"):
        """
        Initialize optimized camera with the configured input source
        
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
        self.use_gpu = JETSON_AVAILABLE
        
        # Get the camera URL for the specified camera ID
        if camera_id in CAMERA_URLS:
            self.camera_url = CAMERA_URLS[camera_id]
        else:
            # Fallback to main camera if the specified ID is not found
            logger.warning(f"[CAMERA] Camera ID '{camera_id}' not found, using main camera")
            self.camera_id = "main"
            self.camera_url = CAMERA_URL  # For backward compatibility
        
        # Use a ring buffer instead of queue for lower latency
        self.buffer_size = 2  # Minimum buffer size to reduce latency
        self.frame_buffer = [None] * self.buffer_size
        self.buffer_index = 0
        self.latest_frame_index = -1
        self.buffer_lock = threading.Lock()
        
        # For threaded capture - optimized for minimal latency
        self.stop_event = threading.Event()
        self.capture_thread = None
        
        # Pre-allocated resize dimensions
        self.resize_dimensions = None
        
        # Pre-allocated resize memory buffer for zero-copy operations
        self.resize_buffer = None
        
        # Error handling with fast recovery
        self.decode_errors = 0
        self.last_error_time = 0
        self.error_threshold = 5
        self.error_window = 3
        
        # Thread sync
        self._thread_initialized = threading.Event()
        
        # For Jetson Nano GPU-accelerated processing
        self.jetson_input = None
        
        # Initialize stream with optimized settings
        logger.info(f"[CAMERA] Initializing low-latency input stream for camera {self.camera_id}")
        self._connect_to_source()
        self._start_capture_thread()
    
    def _connect_to_source(self):
        """Connect to the video source with optimized pipeline settings"""
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
                    time.sleep(0.2)  # Reduced release wait time
                except Exception:
                    pass
                self.cap = None
            
            # Release jetson input if exists
            if hasattr(self, 'jetson_input') and self.jetson_input is not None:
                self.jetson_input = None
            
            # Determine if this is an RTSP source
            is_rtsp = source.startswith('rtsp://')
            
            # Try to use Jetson GPU acceleration if available
            if self.use_gpu and JETSON_AVAILABLE:
                try:
                    # For RTSP streams, use Jetson's optimized decoder
                    if is_rtsp:
                        # Set up GPU-accelerated video input using jetson.utils
                        logger.info(f"[CAMERA:{self.camera_id}] Using Jetson GPU acceleration for RTSP")
                        self.jetson_input = jetson.utils.videoSource(source, argv=['--input-codec=h264'])
                        
                        # Get one test frame to verify connection
                        test_frame = self.jetson_input.Capture()
                        if test_frame:
                            logger.info(f"[CAMERA:{self.camera_id}] Jetson GPU connection successful")
                            
                            # Get the dimensions
                            width = self.jetson_input.GetWidth()
                            height = self.jetson_input.GetHeight()
                            logger.info(f"[CAMERA:{self.camera_id}] Connected via Jetson: {width}x{height}")
                            
                            # OPTIMIZATION: Calculate the resize dimensions only once
                            aspect_ratio = width / height
                            if aspect_ratio > (FRAME_WIDTH / FRAME_HEIGHT):
                                # Image is wider than target
                                new_width = FRAME_WIDTH
                                new_height = int(FRAME_WIDTH / aspect_ratio)
                            else:
                                # Image is taller than target
                                new_height = FRAME_HEIGHT
                                new_width = int(FRAME_HEIGHT * aspect_ratio)
                            
                            # Ensure dimensions are even numbers
                            new_width = new_width - (new_width % 2)
                            new_height = new_height - (new_height % 2)
                            
                            logger.info(f"[CAMERA:{self.camera_id}] Resize target: {new_width}x{new_height}")
                            self.resize_dimensions = (new_width, new_height)
                            
                            # Set source and return without initializing OpenCV capture
                            self.source = source
                            return None
                    else:
                        # For local cameras on Jetson, still try GPU acceleration
                        if source.isdigit() or source == "0":
                            logger.info(f"[CAMERA:{self.camera_id}] Using Jetson GPU acceleration for local camera")
                            # Format for CSI cameras on Jetson: csi://0
                            self.jetson_input = jetson.utils.videoSource(f"csi://{source}")
                            
                            # Get one test frame to verify connection
                            test_frame = self.jetson_input.Capture()
                            if test_frame:
                                logger.info(f"[CAMERA:{self.camera_id}] Jetson GPU connection successful for local camera")
                                
                                # Get the dimensions
                                width = self.jetson_input.GetWidth()
                                height = self.jetson_input.GetHeight()
                                
                                # Calculate resize dimensions
                                aspect_ratio = width / height
                                if aspect_ratio > (FRAME_WIDTH / FRAME_HEIGHT):
                                    # Image is wider than target
                                    new_width = FRAME_WIDTH
                                    new_height = int(FRAME_WIDTH / aspect_ratio)
                                else:
                                    # Image is taller than target
                                    new_height = FRAME_HEIGHT
                                    new_width = int(FRAME_HEIGHT * aspect_ratio)
                                
                                # Ensure dimensions are even numbers
                                new_width = new_width - (new_width % 2)
                                new_height = new_height - (new_height % 2)
                                
                                self.resize_dimensions = (new_width, new_height)
                                self.source = source
                                return None
                
                except Exception as e:
                    logger.error(f"[CAMERA:{self.camera_id}] Error initializing Jetson GPU acceleration: {str(e)}")
                    logger.info(f"[CAMERA:{self.camera_id}] Falling back to CPU processing")
                    self.use_gpu = False
                    self.jetson_input = None
            
            # If we couldn't use Jetson GPU acceleration, use OpenCV
            if is_rtsp:
                # CRITICAL OPTIMIZATION: Use GStreamer pipeline for RTSP with ultra-low latency settings
                # This is one of the most important optimizations for reducing latency
                gst_pipeline = (
                    f"rtspsrc location={source} latency=0 buffer-mode=auto ! "
                    f"rtph264depay ! h264parse ! avdec_h264 ! "
                    f"videoconvert ! appsink max-buffers=1 drop=true sync=false"
                )
                
                logger.info(f"[CAMERA:{self.camera_id}] Using optimized GStreamer pipeline")
                cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                
                # Fallback to FFMPEG if GStreamer fails
                if not cap.isOpened():
                    logger.warning(f"[CAMERA:{self.camera_id}] GStreamer failed, using FFMPEG with optimized settings")
                    
                    # Optimized FFMPEG settings for low latency
                    ffmpeg_options = []
                    ffmpeg_options.append("rtsp_transport;tcp")      # TCP is more reliable
                    ffmpeg_options.append("stimeout;2000000")        # Reduced timeout
                    ffmpeg_options.append("fflags;nobuffer+discardcorrupt")  # No buffering + discard corrupt
                    ffmpeg_options.append("flags;low_delay")         # Low delay flag
                    ffmpeg_options.append("max_delay;50000")         # Drastically reduced delay (microseconds)
                    ffmpeg_options.append("analyzeduration;100000")  # Reduced analyze time
                    ffmpeg_options.append("probesize;32000")         # Smaller probe size
                    ffmpeg_options.append("framedrop;1")             # Enable frame dropping
                    
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = '|'.join(ffmpeg_options)
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
                # For webcams or other sources
                cap = cv2.VideoCapture(source)
                if source.isdigit() or source == "0":  # Local webcam
                    # Optimized webcam settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Wait for connection
            if not cap.isOpened():
                logger.error(f"[CAMERA:{self.camera_id}] Failed to connect to: {source}")
                if ALLOW_FALLBACK:
                    logger.info(f"[CAMERA:{self.camera_id}] Falling back to local webcam")
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        raise RuntimeError("Failed to connect to webcam")
                else:
                    raise RuntimeError(f"Failed to connect to camera source: {source}")
            
            # Configure for low latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer
            
            # Get the original frame size
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"[CAMERA:{self.camera_id}] Connected: {width}x{height} @ {fps if fps > 0 else 'unknown'}fps")
            
            # OPTIMIZATION: Calculate the resize dimensions only once
            aspect_ratio = width / height
            if aspect_ratio > (FRAME_WIDTH / FRAME_HEIGHT):
                # Image is wider than target
                new_width = FRAME_WIDTH
                new_height = int(FRAME_WIDTH / aspect_ratio)
            else:
                # Image is taller than target
                new_height = FRAME_HEIGHT
                new_width = int(FRAME_HEIGHT * aspect_ratio)
            
            # Ensure dimensions are even numbers
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            logger.info(f"[CAMERA:{self.camera_id}] Resize target: {new_width}x{new_height}")
            
            # OPTIMIZATION: Pre-allocate resize buffer for zero-copy operations
            self.resize_dimensions = (new_width, new_height)
            self.resize_buffer = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            
            self.cap = cap
            self.source = source
            
            # Reset error counters
            self.decode_errors = 0
            self.last_error_time = 0
            
            # Clear buffer
            with self.buffer_lock:
                for i in range(self.buffer_size):
                    self.frame_buffer[i] = None
                self.latest_frame_index = -1
            
            return cap
                
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error connecting to camera: {str(e)}")
            if ALLOW_FALLBACK:
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
        """Thread function to continuously capture frames and update the buffer"""
        try:
            # Signal that the thread is initialized
            self._thread_initialized.set()
            
            # Use the appropriate capture method based on whether we're using Jetson GPU
            if self.use_gpu and self.jetson_input is not None:
                logger.info(f"[CAMERA:{self.camera_id}] Starting GPU-accelerated capture thread")
                self._gpu_capture_thread_function()
            else:
                logger.info(f"[CAMERA:{self.camera_id}] Starting CPU capture thread")
                self._cpu_capture_thread_function()
                
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error in capture thread: {str(e)}")
        finally:
            logger.info(f"[CAMERA:{self.camera_id}] Capture thread stopped")
            
    def _cpu_capture_thread_function(self):
        """CPU-based frame capture thread function"""
        consecutive_failures = 0
        while not self.stop_event.is_set():
            try:
                if self.cap is None:
                    time.sleep(0.1)
                    continue
                
                # Read frame with timeout to prevent blocking indefinitely
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    
                    # Log a warning every 30 failed attempts
                    if consecutive_failures % 30 == 1:
                        logger.warning(f"[CAMERA:{self.camera_id}] Failed to read frame ({consecutive_failures} consecutive failures)")
                    
                    # After too many consecutive failures, try to reconnect
                    if consecutive_failures > 300:  # ~10 seconds at 30fps
                        logger.error(f"[CAMERA:{self.camera_id}] Too many consecutive failures, reconnecting...")
                        self._connect_to_source()
                        consecutive_failures = 0
                    
                    time.sleep(0.033)  # ~30fps sleep time
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                
                # Update frame count and last successful read time
                self.frame_count += 1
                self.last_successful_read_time = time.time()
                
                # Resize frame if needed - use zero-copy operations for performance
                if self.resize_dimensions:
                    resized_frame = cv2.resize(
                        frame, self.resize_dimensions, 
                        dst=self.resize_buffer if self.resize_buffer is not None else None,
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    resized_frame = frame
                
                # Store the frame in the buffer
                with self.buffer_lock:
                    self.latest_frame_index = (self.latest_frame_index + 1) % self.buffer_size
                    self.frame_buffer[self.latest_frame_index] = resized_frame
                
            except Exception as e:
                self.decode_errors += 1
                curr_time = time.time()
                
                # Only log the error if not too frequent to avoid log spam
                if curr_time - self.last_error_time > 5.0:
                    logger.error(f"[CAMERA:{self.camera_id}] Error reading frame: {str(e)}")
                    self.last_error_time = curr_time
                
                # Attempt to reconnect if too many errors in a short time
                if self.decode_errors > self.error_threshold and curr_time - self.last_error_time < self.error_window:
                    logger.warning(f"[CAMERA:{self.camera_id}] Too many decode errors, reconnecting...")
                    self._connect_to_source()
                    self.decode_errors = 0
                
                time.sleep(0.033)  # ~30fps sleep time

    def _gpu_capture_thread_function(self):
        """GPU-accelerated frame capture thread function using Jetson Nano"""
        consecutive_failures = 0
        while not self.stop_event.is_set():
            try:
                if self.jetson_input is None:
                    time.sleep(0.1)
                    continue
                
                # Capture frame with Jetson GPU acceleration
                cuda_frame = self.jetson_input.Capture()
                
                if cuda_frame is None:
                    consecutive_failures += 1
                    
                    # Log a warning every 30 failed attempts
                    if consecutive_failures % 30 == 1:
                        logger.warning(f"[CAMERA:{self.camera_id}] Failed to read GPU frame ({consecutive_failures} consecutive failures)")
                    
                    # After too many consecutive failures, try to reconnect
                    if consecutive_failures > 300:  # ~10 seconds at 30fps
                        logger.error(f"[CAMERA:{self.camera_id}] Too many consecutive failures, reconnecting...")
                        self._connect_to_source()
                        consecutive_failures = 0
                    
                    time.sleep(0.033)  # ~30fps sleep time
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                
                # Convert CUDA frame to numpy array (CPU)
                # This is necessary for compatibility with the rest of the pipeline
                cpu_frame = jetson.utils.cudaToNumpy(cuda_frame)
                
                # Convert from RGB to BGR (OpenCV format)
                cpu_frame = cv2.cvtColor(cpu_frame, cv2.COLOR_RGB2BGR)
                
                # Update frame count and last successful read time
                self.frame_count += 1
                self.last_successful_read_time = time.time()
                
                # Resize frame if needed
                if self.resize_dimensions:
                    resized_frame = cv2.resize(
                        cpu_frame, self.resize_dimensions, 
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    resized_frame = cpu_frame
                
                # Store the frame in the buffer
                with self.buffer_lock:
                    self.latest_frame_index = (self.latest_frame_index + 1) % self.buffer_size
                    self.frame_buffer[self.latest_frame_index] = resized_frame
                
            except Exception as e:
                self.decode_errors += 1
                curr_time = time.time()
                
                # Only log the error if not too frequent to avoid log spam
                if curr_time - self.last_error_time > 5.0:
                    logger.error(f"[CAMERA:{self.camera_id}] Error reading GPU frame: {str(e)}")
                    self.last_error_time = curr_time
                
                # Attempt to reconnect if too many errors in a short time
                if self.decode_errors > self.error_threshold and curr_time - self.last_error_time < self.error_window:
                    logger.warning(f"[CAMERA:{self.camera_id}] Too many GPU decode errors, reconnecting...")
                    self._connect_to_source()
                    self.decode_errors = 0
                
                time.sleep(0.033)  # ~30fps sleep time

    def _start_capture_thread(self):
        """Start the frame capture thread"""
        # Stop existing thread if running
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.stop_event.set()
            self.capture_thread.join(timeout=1.0)
            self.stop_event.clear()
            
        # Ensure previous thread is not blocking
        if hasattr(self, '_thread_initialized'):
            self._thread_initialized.clear()

        # Start a new capture thread
        logger.info(f"[CAMERA:{self.camera_id}] Starting capture thread")
        self.capture_thread = threading.Thread(
            target=self._capture_thread_function,
            name=f"FrameCapture-{self.camera_id}",
            daemon=True
        )
        self.capture_thread.start()
        
        # Wait for thread to initialize (max 5 seconds)
        initialized = self._thread_initialized.wait(timeout=5.0)
        if not initialized:
            logger.warning(f"[CAMERA:{self.camera_id}] Capture thread initialization timeout")

    def read(self):
        """
        Read the latest frame from the buffer
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the image
        """
        # Get the latest frame from the buffer
        with self.buffer_lock:
            if self.latest_frame_index >= 0 and self.frame_buffer[self.latest_frame_index] is not None:
                success = True
                frame = self.frame_buffer[self.latest_frame_index].copy()
                self.last_frame = frame
            else:
                success = False
                frame = None

        # If no frames are available in buffer but we have a previous frame, use that
        if not success and self.last_frame is not None:
            success = True
            frame = self.last_frame
            
        # If all else fails, return a blank frame
        if not success:
            logger.warning(f"[CAMERA:{self.camera_id}] No frames available yet")
            success = True
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        return success, frame

    def get_frame_rate(self):
        """
        Estimate the current frame rate based on recent captures
        
        Returns:
            float: Estimated frames per second
        """
        # If we're using Jetson GPU, try to get the frame rate directly
        if self.use_gpu and self.jetson_input is not None:
            try:
                return self.jetson_input.GetFrameRate()
            except:
                pass
        
        # Otherwise, try to get it from OpenCV
        if self.cap is not None:
            try:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    return fps
            except:
                pass
            
        # Last resort: default to 30fps
        return 30.0

    def release(self):
        """Release resources and stop capture thread"""
        logger.info(f"[CAMERA:{self.camera_id}] Releasing resources")
        
        # Stop the capture thread
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.stop_event.set()
            try:
                self.capture_thread.join(timeout=2.0)
            except:
                pass
        
        # Release OpenCV capture
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        # Release Jetson resources
        if hasattr(self, 'jetson_input') and self.jetson_input is not None:
            self.jetson_input = None
        
        logger.info(f"[CAMERA:{self.camera_id}] Resources released")