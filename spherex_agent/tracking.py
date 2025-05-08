from typing import Any, Generator, List, Dict
import cv2
import numpy as np
from ultralytics import YOLO
from .config import config
from .lpr import LPR
from .backend_sync import BackendSync
from .gate_control import GateControl
from .logging import logger
import os
from collections import Counter


class Tracker:
    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        model_path = "resources/" + model_path
        tensorrt_path = "resources/yolo11n.engine"
        if os.path.exists(tensorrt_path):
            logger.info("Loading TensorRT YOLO model for GPU acceleration...")
            self.model = YOLO(tensorrt_path, task="detect")
        else:
            logger.info(
                "Exporting YOLO model to TensorRT for Jetson Nano GPU..."
            )
            try:
                self.model = YOLO(model_path, task="detect")
                self.model.export(
                    format="engine", device="0", half=True
                )  # GPU (device=0) with FP16
                self.model = YOLO(tensorrt_path, task="detect")
                logger.info(
                    "TensorRT YOLO model exported and loaded successfully"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to export to TensorRT: {e}. Falling back to PyTorch model."
                )
                self.model = YOLO(model_path, task="detect")
        self.roi_points: List[List[int]] = config.roi
        self.source: str = config.camera_url
        self.lpr = LPR()
        self.backend_sync = BackendSync()
        self.gate_control = GateControl()
        self.tracked_vehicles: Dict[int, Dict[str, Any]] = (
            {}
        )  # track_id: {status, readings, attempts, plate, authorized}
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
                    cv2.setMouseCallback("Draw ROI", lambda *args: None)

        roi_state = RoiState()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError("Could not open video stream")
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Could not read frame")

        cv2.namedWindow("Draw ROI")
        cv2.setMouseCallback("Draw ROI", roi_state.mouse_callback)
        logger.info(
            "Left-click to add ROI points, right-click to close polygon (minimum 3 points)."
        )

        while roi_state.drawing or len(roi_state.points) < 3:
            temp_frame = frame.copy()
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
            cv2.imshow("Draw ROI", temp_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(roi_state.points) < 3:
            raise ValueError("ROI must have at least 3 points")
        self.roi_points = roi_state.points
        return roi_state.points

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
                                f"Vehicle {track_id} entered ROI - Starting recognition"
                            )

                        vehicle = self.tracked_vehicles[track_id]
                        if vehicle["status"] == "pending":
                            car_crop = original_frame[y1:y2, x1:x2]
                            vehicle["attempts"] += 1
                            logger.info(
                                f"Vehicle {track_id} - Recognition attempt {vehicle['attempts']}/{self.max_attempts}"
                            )
                            license_text = (
                                self.lpr.recognize_plate(car_crop)
                                if car_crop.size > 0
                                else None
                            )

                            if license_text:
                                vehicle["readings"].append(license_text)
                                logger.info(
                                    f"Vehicle {track_id} - Plate reading: {license_text}"
                                )

                                if self.backend_sync.is_authorized(
                                    license_text
                                ):
                                    vehicle["status"] = "authorized"
                                    vehicle["authorized"] = True
                                    vehicle["plate"] = license_text
                                    logger.info(
                                        f"Vehicle {track_id} authorized with plate {license_text} - Opening gate"
                                    )
                                    self.gate_control.open()
                                    self.backend_sync.log_to_backend(
                                        license_text, True, track_id
                                    )
                                elif vehicle["attempts"] >= self.max_attempts:
                                    self._handle_unauthorized(track_id)
                            elif vehicle["attempts"] >= self.max_attempts:
                                logger.info(
                                    f"Vehicle {track_id} - No plate detected after {self.max_attempts} attempts"
                                )
                                vehicle["status"] = "no_plate"
                                vehicle["plate"] = "No plate found"
                                self.backend_sync.log_to_backend(
                                    "No plate found", False, track_id
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
                                f"Vehicle {track_id} with plate {vehicle['plate']} has been logged and is {status}"
                            )
                            if vehicle["authorized"]:
                                logger.info(
                                    f"Vehicle {track_id} still in ROI - Gate remains open"
                                )

                    elif track_id in self.tracked_vehicles:
                        vehicle = self.tracked_vehicles[track_id]
                        if vehicle["authorized"]:
                            logger.info(
                                f"Vehicle {track_id} with plate {vehicle['plate']} left ROI - Closing gate"
                            )
                            self.gate_control.close()
                        else:
                            logger.info(
                                f"Vehicle {track_id} with plate {vehicle['plate']} left ROI - No action taken"
                            )
                        del self.tracked_vehicles[track_id]

            # Clean up vehicles no longer detected
            for track_id in list(self.tracked_vehicles.keys()):
                if track_id not in current_track_ids:
                    vehicle = self.tracked_vehicles[track_id]
                    if vehicle["authorized"]:
                        logger.info(
                            f"Vehicle {track_id} with plate {vehicle['plate']} no longer detected - Closing gate"
                        )
                        self.gate_control.close()
                    else:
                        logger.info(
                            f"Vehicle {track_id} with plate {vehicle['plate']} no longer detected - No action taken"
                        )
                    del self.tracked_vehicles[track_id]

            yield display_frame, []

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def _handle_unauthorized(self, track_id: int):
        vehicle = self.tracked_vehicles[track_id]
        readings = vehicle["readings"]
        if readings:
            most_common = Counter(readings).most_common(1)[0]
            plate, count = most_common
            total_attempts = len(readings)
            probability = count / total_attempts
            logger.info(
                f"Vehicle {track_id} unauthorized after {self.max_attempts} attempts - Final plate: {plate} "
                f"(Readings: {count}/{total_attempts}, Probability: {probability:.2f})"
            )
            vehicle["status"] = "unauthorized"
            vehicle["authorized"] = False
            vehicle["plate"] = plate
            self.backend_sync.log_to_backend(plate, False, track_id)
        else:
            logger.info(
                f"Vehicle {track_id} unauthorized after {self.max_attempts} attempts - No plate readings detected"
            )
            vehicle["status"] = "no_plate"
            vehicle["plate"] = "No plate found"
            self.backend_sync.log_to_backend("No plate found", False, track_id)
