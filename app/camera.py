import cv2
import os
import time
import threading
import numpy as np
import gi
import logging
from .config import CAMERA_URLS, CAMERA_URL, logger, ALLOW_FALLBACK, FRAME_WIDTH, FRAME_HEIGHT

# Required GStreamer imports for DeepStream
try:
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
except Exception as e:
    logger.error(f"Failed to import GStreamer: {str(e)}")
    logger.error("Make sure DeepStream SDK is installed properly")
    raise ImportError(f"DeepStream dependencies not satisfied: {str(e)}")

class InputStream:
    """
    High-performance camera class optimized for minimal-latency video streaming
    with NVIDIA DeepStream acceleration for Jetson Nano.
    """

    def __init__(self, camera_id="main"):
        """
        Initialize DeepStream-accelerated camera with the configured input source
        
        Args:
            camera_id (str): The ID of the camera to use (default: "main")
        """
        # Initialize GStreamer
        Gst.init(None)
        
        # Check CUDA availability for optimizations
        self.cuda_available = False
        try:
            # Attempt to detect CUDA/GPU availability on Jetson
            if os.path.exists('/dev/nvhost-ctrl'):
                self.cuda_available = True
                logger.info(f"[CAMERA:{camera_id}] CUDA acceleration available")
            elif os.path.exists('/usr/local/cuda'):
                self.cuda_available = True
                logger.info(f"[CAMERA:{camera_id}] CUDA installation detected")
        except Exception as e:
            logger.warning(f"[CAMERA:{camera_id}] Error checking CUDA: {str(e)}")
        
        self.frame_count = 0
        self.last_frame = None
        self.source = None
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        self.last_successful_read_time = 0
        self.camera_id = camera_id
        
        # DeepStream-specific attributes
        self.pipeline = None
        self.loop = None
        self.bus = None
        self.appsink = None
        self.mainloop_thread = None
        
        # Hardware acceleration is always enabled with DeepStream
        self.use_hw_accel = True
        
        # Get the camera URL for the specified camera ID
        if camera_id in CAMERA_URLS:
            self.camera_url = CAMERA_URLS[camera_id]
            # Strip quotes if present in the URL from config
            if isinstance(self.camera_url, str):
                self.camera_url = self.camera_url.strip('"\'')
        else:
            # Fallback to main camera if the specified ID is not found
            logger.warning(f"[CAMERA] Camera ID '{camera_id}' not found, using main camera")
            self.camera_id = "main"
            self.camera_url = CAMERA_URL  # For backward compatibility
            # Strip quotes if present
            if isinstance(self.camera_url, str):
                self.camera_url = self.camera_url.strip('"\'')
        
        # Use a ring buffer instead of queue for lower latency
        self.buffer_size = 2  # Minimum buffer size to reduce latency
        self.frame_buffer = [None] * self.buffer_size
        self.buffer_index = 0
        self.latest_frame_index = -1
        self.buffer_lock = threading.Lock()
        
        # For thread management
        self.stop_event = threading.Event()
        self.capture_thread = None
        
        # Pre-allocated resize dimensions
        self.resize_dimensions = None
        
        # Error handling with fast recovery
        self.decode_errors = 0
        self.last_error_time = 0
        self.error_threshold = 5
        self.error_window = 3
        
        # Thread sync
        self._thread_initialized = threading.Event()
        
        # Set up CPU affinity for better performance
        self._setup_cpu_affinity()
        
        # Initialize stream with DeepStream pipeline
        logger.info(f"[CAMERA] Initializing DeepStream input stream for camera {self.camera_id} on Jetson Nano")
        self._connect_to_source()
        self._start_capture_thread()
    
    def _setup_cpu_affinity(self):
        """Set up CPU affinity for better performance on Jetson"""
        try:
            if os.name == 'posix':
                # Use the last CPU core for camera thread to avoid competing with
                # the main processing thread which will use the GPU
                import multiprocessing
                self.num_cpus = multiprocessing.cpu_count()
                logger.info(f"[CAMERA:{self.camera_id}] System has {self.num_cpus} CPU cores")
                # We'll set actual affinity when starting the thread
        except Exception as e:
            logger.warning(f"[CAMERA:{self.camera_id}] Could not determine CPU count: {str(e)}")
    
    def _create_deepstream_pipeline(self, source):
        """Create a DeepStream pipeline optimized for Jetson Nano"""
        try:
            # Get original dimensions for resizing calculations
            width = FRAME_WIDTH
            height = FRAME_HEIGHT
            
            # Strip quotes from source if present
            if isinstance(source, str):
                source = source.strip('"\'')
            
            # Determine the source type and build appropriate pipeline
            if isinstance(source, str) and (
                source.startswith("rtsp://") or 
                source.startswith("rtmp://") or
                source.startswith("http://") or
                source.startswith("https://")
            ):
                # Streaming source (RTSP/RTMP/HTTP)
                if source.startswith("rtsp://"):
                    # RTSP-specific optimized pipeline for DeepStream
                    # FIX: Properly quote the URL and avoid using it directly in the pipeline string
                    source_element = f'rtspsrc location="{source}" latency=0 buffer-mode=auto drop-on-latency=true ! '
                    source_element += 'rtph264depay ! h264parse ! '
                    # Add hardware decoding if CUDA is available
                    if self.cuda_available:
                        source_element += "nvv4l2decoder enable-max-performance=1 ! nvvidconv"
                    else:
                        source_element += "avdec_h264 ! videoconvert"
                        
                elif source.startswith("http") and source.endswith((".mp4", ".mkv", ".avi")):
                    # HTTP video file
                    source_element = f'souphttpsrc location="{source}" ! decodebin'
                else:
                    # Generic streaming source
                    source_element = f'uridecodebin uri="{source}"'
            
            elif source.isdigit() or source == "0":
                # Local camera (V4L2)
                source_element = f"v4l2src device=/dev/video{source} ! video/x-raw, width=640, height=480, framerate=30/1"
            else:
                # Try one more time to handle RTSP URLs that may have formatting issues
                if isinstance(source, str) and "rtsp://" in source:
                    # Extract the RTSP URL part
                    rtsp_part = source[source.find("rtsp://"):]
                    end_markers = [" ", '"', "'"]
                    for marker in end_markers:
                        if marker in rtsp_part:
                            rtsp_part = rtsp_part[:rtsp_part.find(marker)]
                    
                    logger.warning(f"[CAMERA:{self.camera_id}] Attempting to parse malformed RTSP URL: {rtsp_part}")
                    source_element = f'rtspsrc location="{rtsp_part}" latency=0 buffer-mode=auto drop-on-latency=true ! '
                    source_element += 'rtph264depay ! h264parse ! '
                    if self.cuda_available:
                        source_element += "nvv4l2decoder enable-max-performance=1 ! nvvidconv"
                    else:
                        source_element += "avdec_h264 ! videoconvert"
                else:
                    # Unknown source type
                    logger.error(f"[CAMERA:{self.camera_id}] Unsupported source: {source}")
                    raise ValueError(f"Unsupported camera source: {source}")
            
            # Build optimized DeepStream pipeline
            pipeline_str = (
                f"{source_element} ! "
                f"nvvidconv ! "  # NVIDIA video converter
                f"video/x-raw(memory:NVMM), format=NV12 ! "  # NVIDIA memory format
                f"nvvidconv ! "  # Another converter for format conversion
                f"video/x-raw, format=BGRx ! "  # Format that's compatible with OpenCV
                f"videoconvert ! "  # Convert to BGR for OpenCV
                f"video/x-raw, format=BGR ! "  # Final format for OpenCV
                f"appsink name=appsink max-buffers=1 drop=true sync=false emit-signals=true"  # Output to our app
            )
            
            logger.info(f"[CAMERA:{self.camera_id}] Creating DeepStream pipeline")
            logger.debug(f"[CAMERA:{self.camera_id}] Pipeline: {pipeline_str}")
            
            # Create the pipeline
            pipeline = Gst.parse_launch(pipeline_str)
            
            # Get the appsink element for frame retrieval
            appsink = pipeline.get_by_name("appsink")
            appsink.set_property("emit-signals", True)
            appsink.set_property("max-buffers", 1)
            appsink.set_property("drop", True)
            appsink.set_property("sync", False)
            
            # Connect to new-sample signal
            appsink.connect("new-sample", self._on_new_sample)
            
            # Calculate the resize dimensions only once
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
            
            # Set up bus for pipeline messages
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            
            return pipeline, appsink, bus
        
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error creating DeepStream pipeline: {str(e)}")
            raise RuntimeError(f"Failed to create DeepStream pipeline: {str(e)}")

    def _on_new_sample(self, appsink):
        try:
            sample = appsink.emit("pull-sample")
            if sample:
                buf = sample.get_buffer()
                caps = sample.get_caps()
                success, map_info = buf.map(Gst.MapFlags.READ)
                if success:
                    # Get frame dimensions from caps
                    structure = caps.get_structure(0)
                    width = structure.get_value("width")
                    height = structure.get_value("height")
                    
                    # Add safety check for dimensions
                    if width <= 0 or height <= 0:
                        logger.error(f"[CAMERA:{self.camera_id}] Invalid dimensions: {width}x{height}")
                        buf.unmap(map_info)
                        return Gst.FlowReturn.ERROR
                    
                    # Add buffer size safety check
                    expected_size = width * height * 3  # 3 channels for BGR
                    if len(map_info.data) < expected_size:
                        logger.error(f"[CAMERA:{self.camera_id}] Buffer too small: {len(map_info.data)} < {expected_size}")
                        buf.unmap(map_info)
                        return Gst.FlowReturn.ERROR
                    
                    # Convert to numpy array with appropriate shape limiting
                    try:
                        frame = np.ndarray(
                            shape=(height, width, 3),
                            dtype=np.uint8,
                            buffer=map_info.data
                        )
                        
                        # Make a deep copy before releasing buffer
                        frame = frame.copy()
                        buf.unmap(map_info)
                        
                        # Update frame buffer
                        with self.buffer_lock:
                            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                            self.frame_buffer[self.buffer_index] = frame
                            self.latest_frame_index = self.buffer_index
                        
                        self.last_successful_read_time = time.time()
                        self.frame_count += 1
                        
                        return Gst.FlowReturn.OK
                    except Exception as e:
                        logger.error(f"[CAMERA:{self.camera_id}] Array creation error: {str(e)}")
                        buf.unmap(map_info)
                        return Gst.FlowReturn.ERROR
                else:
                    logger.warning(f"[CAMERA:{self.camera_id}] Failed to map buffer")
            
            return Gst.FlowReturn.ERROR
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error in new-sample handler: {str(e)}")
            return Gst.FlowReturn.ERROR

    def _on_bus_message(self, bus, message):
        """Handle GStreamer pipeline messages"""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[CAMERA:{self.camera_id}] Pipeline error: {err.message}")
            # Try to recover
            self._handle_pipeline_error()
        
        elif msg_type == Gst.MessageType.EOS:
            logger.info(f"[CAMERA:{self.camera_id}] End of stream")
            # Restart for stream sources that might terminate
            if self.camera_url.startswith(("rtsp://", "http://", "https://")):
                self._restart_pipeline()
        
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                if new_state == Gst.State.PLAYING:
                    logger.info(f"[CAMERA:{self.camera_id}] Pipeline is now playing")
        
        return True

    def _handle_pipeline_error(self):
        """Handle pipeline errors with recovery logic"""
        current_time = time.time()
        # Reset if errors are too frequent
        if current_time - self.last_error_time < self.error_window:
            self.decode_errors += 1
        else:
            self.decode_errors = 1
        
        self.last_error_time = current_time
        
        if self.decode_errors >= self.error_threshold:
            logger.warning(f"[CAMERA:{self.camera_id}] Too many errors, restarting pipeline")
            self._restart_pipeline()
            self.decode_errors = 0

    def _restart_pipeline(self):
        """Restart the DeepStream pipeline"""
        logger.info(f"[CAMERA:{self.camera_id}] Restarting pipeline")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            # Small delay to ensure cleanup
            time.sleep(0.5)
        
        self._connect_to_source()

    def _connect_to_source(self):
        """Connect to the video source with DeepStream pipeline"""
        try:
            # Determine source from URL or ID
            source = self.camera_url
            
            # Clean up existing pipeline if any
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            
            # Strip quotes if present for logging
            log_source = source
            if isinstance(log_source, str):
                log_source = log_source.strip('"\'')
                
            logger.info(f"[CAMERA:{self.camera_id}] Connecting to: {log_source}")
            
            # Create new pipeline
            try:
                self.pipeline, self.appsink, self.bus = self._create_deepstream_pipeline(source)
            except Exception as e:
                logger.error(f"[CAMERA:{self.camera_id}] DeepStream pipeline creation failed: {str(e)}")
                
                # If ALLOW_FALLBACK is enabled, try OpenCV as a fallback
                if ALLOW_FALLBACK:
                    logger.warning(f"[CAMERA:{self.camera_id}] Attempting fallback to OpenCV capture")
                    self._connect_with_opencv_fallback(source)
                    return
                else:
                    raise  # Re-raise the exception if fallback is not allowed
            
            # Start the pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error(f"[CAMERA:{self.camera_id}] Failed to start pipeline")
                
                # Try fallback if enabled
                if ALLOW_FALLBACK:
                    logger.warning(f"[CAMERA:{self.camera_id}] Pipeline start failed, trying OpenCV fallback")
                    self._connect_with_opencv_fallback(source)
                    return
                else:
                    raise RuntimeError(f"Failed to start DeepStream pipeline for {source}")
            
            self.source = source
            
            # Reset error counters
            self.decode_errors = 0
            self.last_error_time = 0
            
            logger.info(f"[CAMERA:{self.camera_id}] DeepStream pipeline started")
            
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error connecting to source: {str(e)}")
            
            # Last chance fallback
            if ALLOW_FALLBACK:
                try:
                    logger.warning(f"[CAMERA:{self.camera_id}] Last attempt fallback to OpenCV")
                    self._connect_with_opencv_fallback(source)
                    return
                except Exception as fallback_e:
                    logger.error(f"[CAMERA:{self.camera_id}] OpenCV fallback also failed: {str(fallback_e)}")
                    
            raise RuntimeError(f"Failed to connect to camera source: {str(e)}")

    def _connect_with_opencv_fallback(self, source):
        """Fallback to OpenCV for camera connection when DeepStream fails"""
        import cv2
        
        logger.info(f"[CAMERA:{self.camera_id}] Attempting OpenCV fallback connection")
        
        # Clean source string if needed
        if isinstance(source, str):
            source = source.strip('"\'')
        
        # Try to convert to integer if it's a number
        try:
            if source.isdigit():
                source = int(source)
        except (AttributeError, ValueError):
            pass
        
        # Create OpenCV capture
        self.cap = cv2.VideoCapture(source)
        
        # Check if successfully opened
        if not self.cap.isOpened():
            logger.error(f"[CAMERA:{self.camera_id}] OpenCV fallback failed to open source: {source}")
            raise RuntimeError(f"OpenCV fallback failed for source: {source}")
        
        # Set buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Record that we're using OpenCV fallback
        self.using_opencv_fallback = True
        self.source = source
        
        logger.info(f"[CAMERA:{self.camera_id}] Successfully connected using OpenCV fallback")

    def _glib_mainloop_thread(self):
        """Thread function for GLib main loop"""
        try:
            # Set CPU affinity if possible
            if os.name == 'posix' and hasattr(self, 'num_cpus') and self.num_cpus > 1:
                try:
                    import ctypes
                    libc = ctypes.cdll.LoadLibrary('libc.so.6')
                    SYS_gettid = 186
                    tid = libc.syscall(SYS_gettid)
                    # Use the last CPU core for the GLib loop
                    target_cpu = self.num_cpus - 1
                    os.sched_setaffinity(tid, {target_cpu})
                    logger.info(f"[CAMERA:{self.camera_id}] Set GLib loop thread affinity to CPU core {target_cpu}")
                except Exception as e:
                    logger.warning(f"[CAMERA:{self.camera_id}] Could not set CPU affinity: {str(e)}")
            
            # Create GLib main loop
            self.loop = GLib.MainLoop()
            
            # Signal that thread is initialized
            self._thread_initialized.set()
            
            # Run the loop
            self.loop.run()
            
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error in GLib main loop: {str(e)}")

    def _start_capture_thread(self):
        """Start the GLib main loop thread for DeepStream capture"""
        # Don't start if already running
        if self.mainloop_thread is not None and self.mainloop_thread.is_alive():
            return
        
        # Clear stop event and reset initialization event
        self.stop_event.clear()
        self._thread_initialized.clear()
        
        # Create and start main loop thread
        self.mainloop_thread = threading.Thread(
            target=self._glib_mainloop_thread,
            name=f"Camera-{self.camera_id}-GLib-Thread",
            daemon=True
        )
        
        try:
            self.mainloop_thread.start()
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Failed to start GLib thread: {str(e)}")
            return
        
        # Wait for thread initialization but with shorter timeout
        if not self._thread_initialized.wait(timeout=2.0):
            logger.warning(f"[CAMERA:{self.camera_id}] GLib thread initialization timeout")
        
        logger.info(f"[CAMERA:{self.camera_id}] Started GLib main loop thread")

    def read(self):
        """Ultra low-latency frame reading directly from the latest frame"""
        try:
            # Check if we're using OpenCV fallback
            if hasattr(self, 'using_opencv_fallback') and self.using_opencv_fallback:
                if hasattr(self, 'cap') and self.cap is not None:
                    ret, frame = self.cap.read()
                    if ret:
                        self.last_frame = frame
                        self.frame_count += 1
                        return True, frame
            else:
                # Direct access to latest frame from DeepStream
                with self.buffer_lock:
                    if self.latest_frame_index >= 0:
                        frame = self.frame_buffer[self.latest_frame_index]
                        if frame is not None:
                            self.last_frame = frame  # Cache it
                            return True, frame
            
                # Check if the pipeline is running
                if self.pipeline:
                    state = self.pipeline.get_state(0)[1]
                    if state != Gst.State.PLAYING:
                        logger.warning(f"[CAMERA:{self.camera_id}] Pipeline not in PLAYING state, restarting")
                        self._restart_pipeline()
            
                # Check if GLib thread is running
                if self.mainloop_thread is None or not self.mainloop_thread.is_alive():
                    logger.warning(f"[CAMERA:{self.camera_id}] GLib thread not running, restarting")
                    self._start_capture_thread()
                    time.sleep(0.05)  # Small delay
            
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
        """Safely release all DeepStream resources"""
        # Check if using OpenCV fallback
        if hasattr(self, 'using_opencv_fallback') and self.using_opencv_fallback:
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.warning(f"[CAMERA:{self.camera_id}] Error releasing OpenCV capture: {str(e)}")
            logger.info(f"[CAMERA:{self.camera_id}] Released OpenCV resources")
            return
            
        # Set stop event
        self.stop_event.set()
        
        # Stop GLib main loop
        if hasattr(self, 'loop') and self.loop is not None:
            try:
                self.loop.quit()
            except Exception as e:
                logger.warning(f"[CAMERA:{self.camera_id}] Error stopping GLib loop: {str(e)}")
        
        # Wait briefly for thread to stop
        if self.mainloop_thread is not None:
            try:
                self.mainloop_thread.join(timeout=0.5)
            except Exception as e:
                logger.warning(f"[CAMERA:{self.camera_id}] Error joining thread: {str(e)}")
            
        # Stop pipeline
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception as e:
                logger.warning(f"[CAMERA:{self.camera_id}] Error stopping pipeline: {str(e)}")
            self.pipeline = None
        
        logger.info(f"[CAMERA:{self.camera_id}] Released DeepStream resources")