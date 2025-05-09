from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, Generator, List

import cv2
import numpy as np
from ultralytics import YOLO

from .backend_sync import BackendSync
from .config import config
from .gate_control import GateControl
from .logging import logger
from .lpr import LPR

MAX_DISPLAY_HEIGHT = 720


class Tracker:
    def __init__(
        self,
        gate_type: str,
        camera_url: str,
        roi_points: List[List[int]],
        model_path: str = "yolo11n.pt",
    ) -> None:
        self.gate_type = gate_type
        model_path = "resources/" + model_path
        tensorrt_path = "resources/yolo11n.engine"
        if os.path.exists(tensorrt_path):
            logger.info(
                f"Gate {config.gate} ({gate_type}): Loading TensorRT YOLO model for GPU acceleration..."
            )
            self.model = YOLO(tensorrt_path, task="detect")
        else:
            logger.info(
                f"Gate {config.gate} ({gate_type}): Exporting YOLO model to TensorRT for Jetson Nano GPU..."
            )
            try:
                self.model = YOLO(model_path, task="detect")
                self.model.export(format="engine", device="cuda", half=True)
                self.model = YOLO(tensorrt_path, task="detect")
                logger.info(
                    f"Gate {config.gate} ({gate_type}): TensorRT YOLO model exported and loaded successfully"
                )
            except Exception as e:
                logger.warning(
                    f"Gate {config.gate} ({gate_type}): Failed to export to TensorRT: {e}. Falling back to PyTorch model."
                )
                self.model = YOLO(model_path, task="detect")
        self.roi_points: List[List[int]] = roi_points
        self.source: str = camera_url
        self.lpr = LPR()
        self.backend_sync = BackendSync()
        self.gate_control = GateControl()
        self.tracked_vehicles: Dict[int, Dict[str, Any]] = {}
        self.max_attempts = 20

    def draw_roi(self) -> List[List[int]]:
        class RoiState:
            def __init__(self):
                self.points: List[List[int]] = []
                self.drawing: bool = False

            def mouse_callback(
                self, event: int, x: int, y: int, flags: int, param: Any
            ) -> None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.points.append([x, y])
                    self.drawing = True
                elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) > 2:
                    self.drawing = False
                    cv2.setMouseCallback(
                        f"Draw ROI ({self.gate_type})", lambda *args: None
                    )

        roi_state = RoiState()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(
                f"Gate {config.gate} ({self.gate_type}): Could not open video stream"
            )
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(
                f"Gate {config.gate} ({self.gate_type}): Could not read frame"
            )

        # Resize frame to fit within MAX_DISPLAY_HEIGHT while preserving aspect ratio
        orig_height, orig_width = frame.shape[:2]
        scale_factor = min(MAX_DISPLAY_HEIGHT / orig_height, 1.0)
        display_height = int(orig_height * scale_factor)
        display_width = int(orig_width * scale_factor)
        display_frame = cv2.resize(
            frame,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )

        cv2.namedWindow(f"Draw ROI ({self.gate_type})")
        cv2.setMouseCallback(
            f"Draw ROI ({self.gate_type})", roi_state.mouse_callback
        )
        logger.info(
            f"Gate {config.gate} ({self.gate_type}): Left-click to add ROI points, right-click to close polygon (minimum 3 points)."
        )

        while roi_state.drawing or len(roi_state.points) < 3:
            temp_frame = display_frame.copy()
            if roi_state.points:
                cv2.polylines(
                    temp_frame,
                    [np.array(roi_state.points)],
                    False,
                    (0, 255, 0),
                    2,
                )
                for point in roi_state.points:
                    cv2.circle(temp_frame, tuple(point), 5, (0, 0, 255), -1)
            cv2.imshow(f"Draw ROI ({self.gate_type})", temp_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyWindow(f"Draw ROI ({self.gate_type})")
        cv2.destroyAllWindows()

        if len(roi_state.points) < 3:
            raise ValueError(
                f"Gate {config.gate} ({self.gate_type}): ROI must have at least 3 points"
            )

        # Scale ROI points back to original resolution
        scaled_points = [
            [int(x / scale_factor), int(y / scale_factor)]
            for x, y in roi_state.points
        ]
        self.roi_points = scaled_points
        return scaled_points

    def track_and_capture(
        self,
    ) -> Generator[tuple[Any, List[tuple[int, np.ndarray]]], None, None]:
        results = self.model.track(
            source=self.source,
            stream=True,
            persist=True,
            classes=[2],  # Car class (COCO ID 2)
            verbose=False,
        )
        roi_poly = (
            np.array(self.roi_points, np.int32)
            if self.roi_points and len(self.roi_points) > 2
            else None
        )

        for result in results:
            original_frame = result.orig_img
            display_frame = result.plot()
            if roi_poly is not None:
                cv2.polylines(display_frame, [roi_poly], True, (0, 255, 0), 2)

            current_track_ids = set()
            if result.boxes and roi_poly is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id) if box.id is not None else -1
                    current_track_ids.add(track_id)
                    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                    in_roi = any(
                        cv2.pointPolygonTest(roi_poly, corner, False) >= 0
                        for corner in corners
                    )

                    if in_roi:
                        if track_id not in self.tracked_vehicles:
                            self.tracked_vehicles[track_id] = {
                                "status": "pending",
                                "readings": [],
                                "attempts": 0,
                                "plate": None,
                                "authorized": None,
                            }
                            logger.info(
                                f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} entered ROI - Starting recognition"
                            )

                        vehicle = self.tracked_vehicles[track_id]
                        if vehicle["status"] == "pending":
                            car_crop = original_frame[y1:y2, x1:x2]
                            vehicle["attempts"] += 1
                            logger.info(
                                f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} - Recognition attempt {vehicle['attempts']}/{self.max_attempts}"
                            )
                            license_text = (
                                self.lpr.recognize_plate(car_crop)
                                if car_crop.size > 0
                                else None
                            )

                            if license_text:
                                vehicle["readings"].append(license_text)
                                logger.info(
                                    f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} - Plate reading: {license_text}"
                                )

                                if self.backend_sync.is_authorized(
                                    license_text
                                ):
                                    vehicle["status"] = "authorized"
                                    vehicle["authorized"] = True
                                    vehicle["plate"] = license_text
                                    logger.info(
                                        f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} authorized with plate {license_text} - Opening gate"
                                    )
                                    self.gate_control.open(self.gate_type)
                                    self.backend_sync.log_to_backend(
                                        self.gate_type,
                                        license_text,
                                        True,
                                        original_frame,
                                        track_id,
                                    )
                                elif vehicle["attempts"] >= self.max_attempts:
                                    self._handle_unauthorized(
                                        track_id, original_frame
                                    )
                            elif vehicle["attempts"] >= self.max_attempts:
                                logger.info(
                                    f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} - No plate detected after {self.max_attempts} attempts"
                                )
                                vehicle["status"] = "no_plate"
                                vehicle["plate"] = "No plate found"
                                self.backend_sync.log_to_backend(
                                    self.gate_type,
                                    "No plate found",
                                    False,
                                    original_frame,
                                    track_id,
                                )

                        elif vehicle["status"] in [
                            "authorized",
                            "unauthorized",
                            "no_plate",
                        ]:
                            status = (
                                "authorized"
                                if vehicle["authorized"]
                                else "unauthorized"
                            )
                            logger.info(
                                f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} with plate {vehicle['plate']} has been logged and is {status}"
                            )
                            if vehicle["authorized"]:
                                logger.info(
                                    f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} still in ROI - Gate remains open"
                                )

                    elif track_id in self.tracked_vehicles:
                        vehicle = self.tracked_vehicles[track_id]
                        if vehicle["authorized"]:
                            logger.info(
                                f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} with plate {vehicle['plate']} left ROI - Closing gate"
                            )
                            self.gate_control.close(self.gate_type)
                        else:
                            logger.info(
                                f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} with plate {vehicle['plate']} left ROI - No action taken"
                            )
                        del self.tracked_vehicles[track_id]

            for track_id in list(self.tracked_vehicles.keys()):
                if track_id not in current_track_ids:
                    vehicle = self.tracked_vehicles[track_id]
                    if vehicle["authorized"]:
                        logger.info(
                            f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} with plate {vehicle['plate']} no longer detected - Closing gate"
                        )
                        self.gate_control.close(self.gate_type)
                    else:
                        logger.info(
                            f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} with plate {vehicle['plate']} no longer detected - No action taken"
                        )
                    del self.tracked_vehicles[track_id]

            yield display_frame, []

    def _handle_unauthorized(self, track_id: int, frame: np.ndarray):
        vehicle = self.tracked_vehicles[track_id]
        readings = vehicle["readings"]
        if readings:
            most_common = Counter(readings).most_common(1)[0]
            plate, count = most_common
            total_attempts = len(readings)
            probability = count / total_attempts
            logger.info(
                f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} unauthorized after {self.max_attempts} attempts - Final plate: {plate} "
                f"(Readings: {count}/{total_attempts}, Probability: {probability:.2f})"
            )
            vehicle["status"] = "unauthorized"
            vehicle["authorized"] = False
            vehicle["plate"] = plate
            self.backend_sync.log_to_backend(
                self.gate_type, plate, False, frame, track_id
            )
        else:
            logger.info(
                f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} unauthorized after {self.max_attempts} attempts - No plate readings detected"
            )
            vehicle["status"] = "no_plate"
            vehicle["plate"] = "No plate found"
            self.backend_sync.log_to_backend(
                self.gate_type, "No plate found", False, frame, track_id
            )
