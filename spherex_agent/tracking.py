from __future__ import annotations
from collections import Counter
from typing import Any, Dict, Generator, List
import cv2
from cv2.typing import MatLike
import numpy as np
from ultralytics import YOLO
from .backend_sync import BackendSync
from .config import config
from .gate_control import GateControl
from .logging import logger
from .lpr import LPR
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
import os
import arabic_reshaper
from bidi.algorithm import get_display
import time

MAX_DISPLAY_HEIGHT = 720
FONT_PATH = "resources/NotoSansArabic-Regular.ttf"

class Tracker:
    def __init__(
        self,
        gate_type: str,
        camera_url: str,
        roi_points: List[List[int]],
        backend_sync: BackendSync,
        gate_control: GateControl,
        model_path: str = "yolo11n.pt",
    ) -> None:
        self.gate_type = gate_type
        model_path = "resources/" + model_path
        tensorrt_path = "resources/yolo11n.engine"

        self.roi_points: List[List[int]] = roi_points
        self.source: str = camera_url
        self.lpr = LPR()
        self.backend_sync = backend_sync
        self.gate_control = gate_control
        self.tracked_vehicles: Dict[int, Dict[str, Any]] = {}
        self.max_attempts = 20

        if config.gpu and os.path.exists(tensorrt_path):
            logger.info(
                f"Gate {config.gate} ({gate_type}): Loading TensorRT YOLO model for GPU acceleration..."
            )
            try:
                self.model = YOLO(tensorrt_path, task="detect")
                logger.info(
                    f"Gate {config.gate} ({gate_type}): TensorRT model loaded successfully"
                )
            except Exception as e:
                logger.warning(
                    f"Gate {config.gate} ({gate_type}): Failed to load TensorRT: {e}. Falling back to CPU model."
                )
                self.model = YOLO(model_path, task="detect")
        elif config.gpu:
            logger.info(
                f"Gate {config.gate} ({gate_type}): Exporting YOLO model to TensorRT..."
            )
            try:
                self.model = YOLO(model_path, task="detect")
                self.model.export(
                    format="engine", device=config.gpu, half=True
                )
                self.model = YOLO(tensorrt_path, task="detect")
                logger.info(
                    f"Gate {config.gate} ({gate_type}): TensorRT model exported and loaded successfully"
                )
            except Exception as e:
                logger.warning(
                    f"Gate {config.gate} ({gate_type}): Failed to export to TensorRT: {e}. Falling back to CPU model."
                )
                self.model = YOLO(model_path, task="detect")
        else:
            logger.info(
                f"Gate {config.gate} ({gate_type}): Loading standard YOLO model for CPU..."
            )
            self.model = YOLO(model_path, task="detect")

    def draw_roi(self) -> List[List[int]]:
        class RoiState:
            def __init__(self, gate_type: str) -> None:
                self.points: List[List[int]] = []
                self.drawing: bool = False
                self.gate_type = gate_type

            def mouse_callback(
                self, event: int, x: int, y: int, flags: int, param: Any
            ) -> None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.points.append([x, y])
                    self.drawing = True
                elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) > 2:
                    self.drawing = False
                    cv2.setMouseCallback(
                        f"Draw ROI {config.gate} ({self.gate_type})",
                        lambda event, x, y, flags, param: None,
                    )

        roi_state = RoiState(self.gate_type)

        max_retries = 5
        for attempt in range(max_retries):
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                break
            logger.warning(
                f"Gate {config.gate} ({self.gate_type}): Failed to open video stream (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(1)
        else:
            raise ValueError(
                f"Gate {config.gate} ({self.gate_type}): Could not open video stream after {max_retries} attempts"
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

    def _render_arabic_text(
        self, text: str, font_size: int, img_shape: tuple[int, int, int]
    ) -> tuple[MatLike, Any]:
        # Reshape and reorder Arabic text
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        # Add spaces between characters for clarity
        spaced_text = " ".join(bidi_text)

        # Create a blank PIL image for text rendering
        pil_img = Image.new(
            "RGB", (img_shape[1], img_shape[0]), color=(0, 0, 0)
        )
        draw = ImageDraw.Draw(pil_img)

        # Load the Arabic font
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except Exception as e:
            logger.error(
                f"Gate {config.gate} ({self.gate_type}): Failed to load font {FONT_PATH}: {e}"
            )
            raise ValueError(f"Font {FONT_PATH} not found or invalid")

        # Calculate text size and position
        text_bbox = draw.textbbox((0, 0), spaced_text, font=font)
        _, text_height = (
            text_bbox[2] - text_bbox[0],
            text_bbox[3] - text_bbox[1],
        )

        # Position text in bottom-left corner with padding
        padding = 10
        text_position = (padding, img_shape[0] - text_height - padding)

        # Draw text (white color)
        draw.text(text_position, spaced_text, font=font, fill=(255, 255, 255))

        # Convert PIL image to OpenCV format
        text_img = np.array(pil_img)
        text_img = cv2.cvtColor(text_img, cv2.COLOR_RGB2BGR)

        # Create a mask for non-black pixels
        mask = np.any(text_img != [0, 0, 0], axis=2)

        return text_img, mask

    def track_and_capture(
        self,
    ) -> Generator[tuple[Any, List[tuple[int, NDArray[Any]]]], None, None]:
        max_retries = 10  # Increased retries for robustness
        for attempt in range(max_retries):
            try:
                # Use GStreamer pipeline with software decoding
                gst_pipeline = (
                    f"rtspsrc location={self.source} latency=200 ! "
                    f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
                    f"appsink"
                )
                cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    break
                logger.warning(
                    f"Gate {config.gate} ({self.gate_type}): Failed to open video stream (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(2)  # Increased delay for network stability
            except Exception as e:
                logger.warning(
                    f"Gate {config.gate} ({self.gate_type}): Failed to start tracking (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(2)
        else:
            logger.error(
                f"Gate {config.gate} ({self.gate_type}): Failed to start tracking after {max_retries} attempts"
            )
            return

        roi_poly = (
            np.array(self.roi_points, np.int32)
            if self.roi_points and len(self.roi_points) > 2
            else None
        )

        for result in self.model.track(
            source=cap,
            stream=True,
            persist=True,
            classes=[2, 3, 5, 7],
            verbose=False,
        ):
            original_frame = result.orig_img
            # Comment out display-related code as per previous request
            # display_frame = result.plot()
            # if roi_poly is not None:
            #     cv2.polylines(display_frame, [roi_poly], True, (0, 255, 0), 2)

            current_track_ids = set()
            text_to_display = []
            if result.boxes and roi_poly is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id) if box.id is not None else -1
                    if track_id == -1:
                        logger.debug(
                            f"Gate {config.gate} ({self.gate_type}): Skipping invalid track_id -1"
                        )
                        continue
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
                                "first_frame": original_frame.copy(),
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
                                text_to_display.append(
                                    f"Vehicle {track_id}: {license_text}"
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
                                    if vehicle["first_frame"] is not None:
                                        self.backend_sync.log_to_backend(
                                            self.gate_type,
                                            license_text,
                                            True,
                                            vehicle["first_frame"],
                                            track_id,
                                        )
                                    else:
                                        logger.warning(
                                            f"Gate {config.gate} ({self.gate_type}): No first frame available for authorized vehicle {track_id}"
                                        )
                            elif vehicle["attempts"] >= self.max_attempts:
                                self._handle_unauthorized(
                                    track_id, vehicle["first_frame"]
                                )
                            else:
                                text_to_display.append(
                                    f"Vehicle {track_id}: Detecting..."
                                )

                        if vehicle["status"] in ["authorized", "unauthorized"]:
                            status = (
                                "Authorized"
                                if vehicle["authorized"]
                                else "Unauthorized"
                            )
                            text_to_display.append(
                                f"Vehicle {track_id}: {vehicle['plate']} - {status}"
                            )

                    elif track_id in self.tracked_vehicles:
                        vehicle = self.tracked_vehicles[track_id]
                        if vehicle["status"] == "pending" and vehicle["readings"]:
                            # Vehicle left ROI early; decide based on readings
                            self._handle_unauthorized(track_id, vehicle["first_frame"])
                            vehicle = self.tracked_vehicles[track_id]  # Refresh vehicle data
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
                    if vehicle["status"] == "pending" and vehicle["readings"]:
                        # Vehicle no longer detected; decide based on readings
                        self._handle_unauthorized(track_id, vehicle["first_frame"])
                        vehicle = self.tracked_vehicles[track_id]  # Refresh vehicle data
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

            # Comment out display-related code
            # if text_to_display:
            #     font_size = 24  # Suitable for resized frame
            #     text_y_offset = display_frame.shape[0] - 10  # Bottom padding
            #     for text in reversed(text_to_display):  # Bottom to top
            #         text_img, mask = self._render_arabic_text(
            #             text, font_size, display_frame.shape
            #         )
            #         text_y_offset -= (
            #             text_img.shape[0] // len(text_to_display)
            #         ) + 5  # Space between lines
            #         display_frame[mask] = text_img[mask]

            # yield display_frame, []
            yield original_frame, []  # Use original_frame since display is disabled

    def _handle_unauthorized(self, track_id: int, frame: NDArray[Any] | None) -> None:
        vehicle = self.tracked_vehicles[track_id]
        readings = vehicle["readings"]
        if readings:
            most_common = Counter(readings).most_common(1)[0]
            plate, count = most_common
            total_valid_readings = len(readings)
            probability = count / total_valid_readings
            logger.info(
                f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} unauthorized after {vehicle['attempts']} attempts - Final plate: {plate} "
                f"(Valid readings: {count}/{total_valid_readings}, Probability: {probability:.2f})"
            )
            vehicle["status"] = "unauthorized"
            vehicle["authorized"] = False
            vehicle["plate"] = plate
            if frame is not None:
                self.backend_sync.log_to_backend(
                    self.gate_type, plate, False, frame, track_id
                )
            else:
                logger.warning(
                    f"Gate {config.gate} ({self.gate_type}): No first frame available for unauthorized vehicle {track_id}"
                )
        else:
            logger.info(
                f"Gate {config.gate} ({self.gate_type}): Vehicle {track_id} unauthorized after {vehicle['attempts']} attempts - No valid plate readings detected"
            )
            vehicle["status"] = "no_plate"
            vehicle["plate"] = "No plate found"
            if frame is not None:
                self.backend_sync.log_to_backend(
                    self.gate_type, "No plate found", False, frame, track_id
                )
            else:
                logger.warning(
                    f"Gate {config.gate} ({self.gate_type}): No first frame available for vehicle {track_id} with no plate"
                )