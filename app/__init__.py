#!/usr/bin/env python3
import os
import signal
import sys
import threading
import numpy as np
from time import sleep, time
from datetime import datetime

import cv2
import requests

from app.camera            import InputStream
from app.config            import (
    API_BASE_URL, CAMERA_URLS, CAMERA_TYPES,
    GATE, PROCESS_EVERY, logger, FRAME_WIDTH, FRAME_HEIGHT
)
from app.gate              import GateControl
from app.lpr_model         import PlateProcessor
from app.sync              import SyncManager
from app.vehicle_tracking  import VehicleTracker


class CameraManager:
    """Manages multiple camera streams and their associated trackers."""
    def __init__(self):
        self.streams      = {}
        self.trackers     = {}
        self.frame_counts = {}
        for cam_id in CAMERA_URLS:
            self.frame_counts[cam_id] = 0

    def initialize_streams(self):
        """Instantiate InputStream + VehicleTracker for each camera."""
        for cam_id in CAMERA_URLS:
            try:
                logger.info(f"[MANAGER] Initializing camera '{cam_id}'")
                self.streams[cam_id]  = InputStream(cam_id)
                self.trackers[cam_id] = VehicleTracker(cam_id)
                logger.info(f"[MANAGER] '{cam_id}' ready")
            except Exception as e:
                logger.error(f"[MANAGER] Failed to init '{cam_id}': {e}")
        return bool(self.streams)

    def get_camera_ids(self):
        return list(self.streams.keys())

    def read_frame(self, cam_id):
        """Grabs one frame + increments counter."""
        stream = self.streams.get(cam_id)
        if not stream:
            return False, None, 0

        ok, frame = stream.read()
        if ok and frame is not None:
            self.frame_counts[cam_id] += 1
            return True, frame, self.frame_counts[cam_id]
        return False, None, self.frame_counts[cam_id]

    def release_all(self):
        for cam_id, stream in self.streams.items():
            try:
                stream.release()
                logger.info(f"[MANAGER] Released '{cam_id}'")
            except Exception as e:
                logger.warning(f"[MANAGER] Error releasing '{cam_id}': {e}")


class SpherexAgent:
    def __init__(self):
        signal.signal(signal.SIGINT, self._on_sigint)

        logger.info("[AGENT] Initializing SpherexAgent")
        self.gate        = GateControl()
        self.processor   = PlateProcessor()
        self.cache       = SyncManager()
        self.cam_mgr     = CameraManager()
        self.is_running  = True
        self.last_detect = 0.0
        self.cooldown    = 2.0   # seconds between logs
        logger.info("[AGENT] Initialization complete")

    def _on_sigint(self, sig, frame):
        logger.info("[AGENT] SIGINT received, shutting down.")
        self.is_running = False

    def log_gate_entry(self, plate, frame, is_authorized, cam_id):
        """Uploads an image + metadata whenever a plate is seen."""
        try:
            # overlay plate text & ROI
            img = self.processor.add_text_to_image(frame, plate)
            tracker = self.cam_mgr.trackers.get(cam_id)
            if tracker and tracker.roi_polygon is not None:
                img = self.processor.visualize_roi(img, tracker.roi_polygon)

            temp = f"gate_entry_{cam_id}.jpg"
            cv2.imwrite(temp, img)

            # upload file
            with open(temp, "rb") as f:
                resp = requests.post(
                    f"{API_BASE_URL}/api/method/spherex.api.upload_file",
                    files={"file": f}
                )
            file_url = resp.json()["message"]["file_url"]

            # log entry
            data = {
                "gate": GATE,
                "license_plate": plate,
                "authorized": is_authorized,
                "image": file_url,
                "access_type": CAMERA_TYPES.get(cam_id, "Entry"),
                "camera": cam_id
            }
            requests.post(f"{API_BASE_URL}/api/resource/Gate Entry Log", data=data)
        except Exception as e:
            logger.error(f"[AGENT:{cam_id}] Error logging entry: {e}")
        finally:
            if os.path.exists(temp):
                os.remove(temp)

    def process_camera(self, cam_id):
        """Thread target: pull frames, run tracker, handle plates, and display."""
        tracker = self.cam_mgr.trackers[cam_id]
        window  = f"Detections - {cam_id}"

        while self.is_running:
            ok, frame, count = self.cam_mgr.read_frame(cam_id)
            if not ok or frame is None:
                sleep(0.05)
                continue

            # draw frame counter
            cv2.putText(frame, f"{cam_id} #{count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # only every Nth frame
            if count % PROCESS_EVERY == 0:
                now = time()
                if now - self.last_detect > self.cooldown:
                    detected, vis = tracker.detect_vehicles(frame)
                    if detected and vis is not None:
                        cv2.imshow(window, vis)
                        # handle newly detected plates
                        new = set(tracker.detected_plates.items()) - getattr(tracker, "_seen_plates", set())
                        for tid, plate in new:
                            auth = self.cache.is_authorized(plate)
                            tracker.last_recognized_plate = plate
                            tracker.last_plate_authorized = auth

                            logger.info(f"[AGENT:{cam_id}] Plate {plate} (ID {tid}) -> {'AUTHORIZED' if auth else 'DENIED'}")
                            if auth:
                                # open gate
                                typ = CAMERA_TYPES.get(cam_id, "Entry").lower()
                                if typ == "entry":
                                    self.gate.open_entry(); sleep(5); self.gate.close_entry()
                                else:
                                    self.gate.open_exit();  sleep(5); self.gate.close_exit()
                            # log
                            self.log_gate_entry(plate, vis, int(auth), cam_id)

                        # record seen plates
                        tracker._seen_plates = set(tracker.detected_plates.items())
                        self.last_detect = now
                    else:
                        # no detection: still show ROI outline
                        roi_vis = tracker.visualize_detection(frame, [], [], [])
                        cv2.imshow(window, roi_vis)

            # refresh window & handle 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break

        logger.info(f"[AGENT] Camera loop for '{cam_id}' exiting")

    def start(self):
        logger.info("[AGENT] Starting streams…")
        if not self.cam_mgr.initialize_streams():
            logger.error("[AGENT] No cameras initialized, exiting.")
            return

        # create windows in main thread
        for cam_id in self.cam_mgr.get_camera_ids():
            cv2.namedWindow(f"Detections - {cam_id}", cv2.WINDOW_NORMAL)

        # start one thread per camera
        threads = []
        for cam_id in self.cam_mgr.get_camera_ids():
            t = threading.Thread(target=self.process_camera,
                                 args=(cam_id,), daemon=True)
            t.start()
            threads.append(t)

        # wait for shutdown
        while self.is_running:
            sleep(0.2)

        # teardown
        logger.info("[AGENT] Shutting down threads and releasing resources")
        self.cam_mgr.release_all()
        cv2.destroyAllWindows()
        logger.info("[AGENT] Shutdown complete.")


if __name__ == "__main__":
    SpherexAgent().start()
