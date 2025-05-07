import os
import time
import threading
import numpy as np
import gi
from .config import CAMERA_URLS, CAMERA_URL, logger, FRAME_WIDTH, FRAME_HEIGHT

# Required GStreamer imports for DeepStream
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

class InputStream:
    """
    DeepStream-accelerated camera for minimal-latency video streaming on Jetson Nano.
    """

    def __init__(self, camera_id="main"):
        """
        Initialize DeepStream camera with the configured input source
        
        Args:
            camera_id (str): The ID of the camera to use (default: "main")
        """
        # Initialize GStreamer
        Gst.init(None)
        
        self.frame_count = 0
        self.last_frame = None
        self.camera_id = camera_id
        
        # Get the camera URL for the specified camera ID
        if camera_id in CAMERA_URLS:
            self.camera_url = CAMERA_URLS[camera_id].strip('"\'')
        else:
            # Fallback to main camera if the specified ID is not found
            logger.warning(f"[CAMERA] Camera ID '{camera_id}' not found, using main camera")
            self.camera_id = "main"
            self.camera_url = CAMERA_URL.strip('"\'')
        
        # Buffer for storing the latest frame
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # For thread management
        self.stop_event = threading.Event()
        
        # Initialize stream
        logger.info(f"[CAMERA] Initializing DeepStream input stream for camera {self.camera_id}")
        self._create_pipeline()
        self._start_stream()
    
    def _create_pipeline(self):
        """Create a simple DeepStream pipeline based on the working test script"""
        try:
            # Use dimensions from config
            width = FRAME_WIDTH
            height = FRAME_HEIGHT
            
            # Create pipeline for RTSP source
            if self.camera_url.startswith("rtsp://"):
                pipeline_str = (
                    f"rtspsrc location={self.camera_url} latency=0 ! "
                    "rtph264depay ! h264parse ! nvv4l2decoder ! "
                    f"nvvidconv ! video/x-raw,format=NV12,width={width},height={height} ! "
                    "videoconvert ! video/x-raw,format=BGR ! "
                    "appsink name=appsink max-buffers=1 drop=true sync=false emit-signals=true"
                )
            else:
                # For non-RTSP sources (fallback to generic uridecodebin)
                pipeline_str = (
                    f"uridecodebin uri={self.camera_url} ! "
                    f"videoconvert ! video/x-raw,format=BGR,width={width},height={height} ! "
                    "appsink name=appsink max-buffers=1 drop=true sync=false emit-signals=true"
                )
            
            logger.info(f"[CAMERA:{self.camera_id}] Creating pipeline")
            
            # Parse and create the pipeline
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # Get the appsink element
            self.appsink = self.pipeline.get_by_name("appsink")
            self.appsink.connect("new-sample", self._on_new_sample)
            
            # Set up bus for messages
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect("message", self._on_bus_message)
            
            return True
            
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error creating pipeline: {str(e)}")
            return False

    def _on_new_sample(self, appsink):
        """Handle new video frames from the pipeline"""
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
                    
                    # Convert to numpy array
                    frame = np.ndarray(
                        shape=(height, width, 3),
                        dtype=np.uint8,
                        buffer=map_info.data
                    )
                    
                    # Make a copy and update frame
                    frame = frame.copy()
                    buf.unmap(map_info)
                    
                    with self.frame_lock:
                        self.current_frame = frame
                    
                    self.frame_count += 1
                    return Gst.FlowReturn.OK
                else:
                    buf.unmap(map_info)
            
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
            logger.debug(f"[CAMERA:{self.camera_id}] Debug info: {debug}")
            self._restart_pipeline()
        
        elif msg_type == Gst.MessageType.EOS:
            logger.info(f"[CAMERA:{self.camera_id}] End of stream")
            self._restart_pipeline()
        
        return True

    def _start_stream(self):
        """Start the GLib main loop in a separate thread"""
        # Create GLib main loop
        self.loop = GLib.MainLoop()
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error(f"[CAMERA:{self.camera_id}] Failed to start pipeline")
            return False
        
        # Start GLib main loop in a thread
        self.mainloop_thread = threading.Thread(
            target=self._mainloop_thread,
            daemon=True
        )
        self.mainloop_thread.start()
        
        logger.info(f"[CAMERA:{self.camera_id}] Pipeline started successfully")
        return True

    def _mainloop_thread(self):
        """GLib main loop thread function"""
        try:
            self.loop.run()
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error in GLib main loop: {str(e)}")

    def _restart_pipeline(self):
        """Restart the pipeline when needed"""
        logger.info(f"[CAMERA:{self.camera_id}] Restarting pipeline")
        
        # Stop current pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        # Recreate pipeline
        if self._create_pipeline():
            self._start_stream()

    def read(self):
        """Read the latest frame"""
        try:
            with self.frame_lock:
                if self.current_frame is not None:
                    self.last_frame = self.current_frame.copy()
                    return True, self.last_frame
            
            # Return last frame if available
            if self.last_frame is not None:
                return True, self.last_frame
                
        except Exception as e:
            logger.error(f"[CAMERA:{self.camera_id}] Error reading frame: {str(e)}")
            
        return False, None

    def get_frame_rate(self):
        """Calculate the current frame rate"""
        if not hasattr(self, 'last_fps_time'):
            self.last_fps_time = time.time()
            self.last_frame_count = self.frame_count
            return 0
            
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed > 0:
            fps = (self.frame_count - self.last_frame_count) / elapsed
            self.last_fps_time = current_time
            self.last_frame_count = self.frame_count
            return fps
            
        return 0

    def release(self):
        """Safely release all resources"""
        # Stop GLib main loop
        if hasattr(self, 'loop') and self.loop is not None:
            try:
                self.loop.quit()
            except Exception as e:
                logger.warning(f"[CAMERA:{self.camera_id}] Error stopping GLib loop: {str(e)}")
        
        # Wait for thread to stop
        if hasattr(self, 'mainloop_thread') and self.mainloop_thread is not None:
            try:
                self.mainloop_thread.join(timeout=0.5)
            except Exception:
                pass
            
        # Stop pipeline
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        
        logger.info(f"[CAMERA:{self.camera_id}] Released resources")