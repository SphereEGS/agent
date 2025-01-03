import queue
from threading import Thread

import cv2


class CameraStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.queue = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                return

            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True

    def release(self):
        self.stopped = True
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()
