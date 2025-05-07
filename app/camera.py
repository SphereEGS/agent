import cv2
import os
import time
import threading
import queue
import numpy as np
from .config import CAMERA_URLS, CAMERA_URL, logger, ALLOW_FALLBACK, FRAME_WIDTH, FRAME_HEIGHT

class InputStream:
    """
    High-performance camera class optimized for minimal-latency video streaming
    with RTSP acceleration and pipeline optimizations for real-time processing.
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
            
            # Determine if this is an RTSP source
            is_rtsp = source.startswith('rtsp://')
            
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
        """Ultra-optimized thread function to minimize frame latency"""
        reconnect_delay = 0.5  # Reduced initial delay
        max_reconnect_delay = 2.0  # Reduced max delay
        consecutive_errors = 0
        max_consecutive_errors = 3
        last_frame_time = 0
        
        # Signal thread initialization
        self._thread_initialized.set()

        while not self.stop_event.is_set():
            try:
                # Check connection status
                if not self.cap or not self.cap.isOpened():
                    logger.warning(f"[CAMERA:{self.camera_id}] Connection lost, reconnecting...")
                    self.cap = self._connect_to_source()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                    continue

                # OPTIMIZATION: Direct read without lock for speed
                ret, frame = self.cap.read()
                
                current_time = time.time()
                
                # Check if we got a valid frame
                if not ret or frame is None or len(frame.shape) != 3:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.warning(f"[CAMERA:{self.camera_id}] {consecutive_errors} consecutive failures, reconnecting...")
                        self.cap = self._connect_to_source()
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                        consecutive_errors = 0
                    time.sleep(0.001)  # Minimal sleep
                    continue
                
                # Successfully got frame
                consecutive_errors = 0
                reconnect_delay = 0.5  # Reset reconnect delay
                
                # OPTIMIZATION: Resize using pre-allocated buffer for better performance
                if self.resize_dimensions:
                    h, w = frame.shape[:2]
                    if h > 0 and w > 0:
                        try:
                            # OPTIMIZATION: Use nearest neighbor for fastest resize
                            cv2.resize(frame, self.resize_dimensions, dst=self.resize_buffer, 
                                      interpolation=cv2.INTER_NEAREST)
                            frame = self.resize_buffer
                        except Exception:
                            # Fallback to standard resize if pre-allocation fails
                            frame = cv2.resize(frame, self.resize_dimensions, 
                                              interpolation=cv2.INTER_NEAREST)
                
                # OPTIMIZATION: Use ring buffer instead of queue for lower overhead
                with self.buffer_lock:
                    # Update buffer with new frame
                    self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                    self.frame_buffer[self.buffer_index] = frame
                    self.latest_frame_index = self.buffer_index
                
                self.last_successful_read_time = current_time
                self.frame_count += 1
                
                # OPTIMIZATION: Dynamic sleep based on frame rate
                # If we're getting frames too quickly, sleep a tiny bit to prevent CPU overload
                elapsed = current_time - last_frame_time
                if elapsed < 0.016:  # targeting 60fps max (1/60 ≈ 0.016)
                    sleep_time = max(0.001, 0.016 - elapsed)
                    time.sleep(sleep_time)
                
                last_frame_time = current_time
                
            except Exception as e:
                logger.error(f"[CAMERA:{self.camera_id}] Error in capture thread: {str(e)}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

    def _start_capture_thread(self):
        """Start the frame capture thread with fast initialization"""
        # Don't start if already running
        if self.capture_thread is not None and self.capture_thread.is_alive():
            return
        
        # Clear stop event and reset initialization event
        self.stop_event.clear()
        self._thread_initialized.clear()
        
        # Create and start thread with high priority
        self.capture_thread = threading.Thread(
            target=self._capture_thread_function, 
            name=f"Camera-{self.camera_id}-Thread",
            daemon=True
        )
        
        # OPTIMIZATION: Set thread to higher priority if possible
        try:
            self.capture_thread.start()
            # Try to set thread priority higher on platforms that support it
            if hasattr(os, "sched_setaffinity") and os.name == 'posix':
                # On Linux, try to set CPU affinity to the first CPU
                try:
                    import ctypes
                    libc = ctypes.cdll.LoadLibrary('libc.so.6')
                    SYS_gettid = 186
                    tid = libc.syscall(SYS_gettid)
                    os.sched_setaffinity(tid, {0})  # Set affinity to CPU 0
                except:
                    pass
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Failed to start thread: {str(e)}")
            return
        
        # Wait for thread initialization but with shorter timeout
        if not self._thread_initialized.wait(timeout=2.0):
            logger.warning(f"[CAMERA:{self.camera_id}] Thread initialization timeout")
        
        logger.info(f"[CAMERA:{self.camera_id}] Started high-priority capture thread")

    def read(self):
        """Ultra low-latency frame reading directly from the latest frame"""
        try:
            # OPTIMIZATION: Short-circuit for speed - direct access to latest frame
            with self.buffer_lock:
                if self.latest_frame_index >= 0:
                    frame = self.frame_buffer[self.latest_frame_index]
                    if frame is not None:
                        self.last_frame = frame  # Cache it
                        return True, frame
            
            # Check if capture thread is running, restart if needed
            if self.capture_thread is None or not self.capture_thread.is_alive():
                logger.warning(f"[CAMERA:{self.camera_id}] Capture thread not running, restarting")
                self._start_capture_thread()
                time.sleep(0.05)  # Reduced delay
            
            # OPTIMIZATION: Direct capture as fallback but only if really needed
            # This reduces excess buffer reads when we already have a thread
            if self.cap and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        # Resize if needed
                        if self.resize_dimensions:
                            # Try to use pre-allocated buffer
                            try:
                                cv2.resize(frame, self.resize_dimensions, dst=self.resize_buffer,
                                          interpolation=cv2.INTER_NEAREST)
                                frame = self.resize_buffer
                            except:
                                frame = cv2.resize(frame, self.resize_dimensions, 
                                                 interpolation=cv2.INTER_NEAREST)
                                
                        self.last_frame = frame
                        return True, frame
                except Exception as e:
                    logger.debug(f"[CAMERA:{self.camera_id}] Error in direct capture: {str(e)}")
            
            # Last resort: use cached frame
            if self.last_frame is not None:
                return True, self.last_frame
                
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error in read: {str(e)}")
            
        # Final fallback
        if self.last_frame is not None:
            return True, self.last_frame
            
        return False, None

    def get_frame_rate(self):
        """Calculate the effective frame rate with minimal overhead"""
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
        """Safely release all resources with minimal waiting"""
        # Set stop event
        self.stop_event.set()
        
        # Wait briefly for thread to stop
        if self.capture_thread is not None:
            try:
                self.capture_thread.join(timeout=0.5)  # Reduced timeout
            except Exception:
                pass
            
        # Release capture device
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        
        logger.info(f"[CAMERA:{self.camera_id}] Released resources")