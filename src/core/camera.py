import cv2
import numpy as np
from queue import Queue
import threading
import time


class CameraManager:
    def __init__(self):
        self.rtsp_url = "rtsp://admin:skam%40123456@192.168.0.242/media/video1"
        self.cap = None
        self.frame_queue = Queue(maxsize=2)
        self.processing = True
        self.capture_thread = None

    def start(self):
        self.connect_to_stream()
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def connect_to_stream(self) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.rtsp_url)
            return self.cap.isOpened()
        except Exception:
            return False

    def _capture_frames(self):
        while self.processing:
            if not self.cap.isOpened():
                if not self.connect_to_stream():
                    time.sleep(1)
                    continue

            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put(frame, block=False)
                except Queue.Full:
                    pass
            time.sleep(0.1)

    def get_frame(self) -> np.ndarray:
        try:
            return self.frame_queue.get_nowait()
        except Queue.Empty:
            return None

    def cleanup(self):
        self.processing = False
        if self.cap:
            self.cap.release()
