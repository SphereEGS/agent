from __future__ import annotations

import os
from typing import Any, List, Optional
import cv2
from numpy.typing import NDArray
from ultralytics import YOLO  # type: ignore[import]
from .config import config
from .logging import logger
import time

ARABIC_MAPPING = {
    "0": "٠",
    "1": "١",
    "2": "٢",
    "3": "٣",
    "4": "٤",
    "5": "٥",
    "6": "٦",
    "7": "٧",
    "8": "٨",
    "9": "٩",
    "Beh": "ب",
    "Daad": "ض",
    "Een": "ع",
    "F": "ف",
    "Heeh": "H",
    "Kaaf": "ك",
    "Laam": "ل",
    "License Plate": "",
    "Meem": "م",
    "Noon": "ن",
    "Q": "ق",
    "R": "ر",
    "Saad": "ص",
    "Seen": "س",
    "Taa": "ط",
    "Wow": "و",
    "Yeeh": "ي",
    "Zah": "ظ",
    "Zeen": "ز",
    "alef": "أ",
    "car": "",
    "daal": "د",
    "geem": "ج",
    "ghayn": "غ",
    "khaa": "خ",
    "sheen": "ش",
    "teh": "ت",
    "theh": "ث",
    "zaal": "ذ",
    "7aah": "ح",
}

PLATE_DETECTION_SIZE = 480
QUANTIZED_PATH = "resources/lpr_nano_openvino_model"
TENSORRT_PATH = "resources/lpr_nano.engine"


class LPR:
    def __init__(self) -> None:
        model_path = "resources/" + config.lpr_model

        # Cache class names to avoid repeated lookups
        self.lpr_class_names: List[str] = []

        if config.gpu and os.path.exists(TENSORRT_PATH):
            logger.info("Loading TensorRT LPR model for GPU acceleration...")
            try:
                self.model = YOLO(TENSORRT_PATH, task="detect")
                logger.info("TensorRT LPR model loaded successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to load TensorRT model: {e}. Falling back to CPU model."
                )
                self._load_cpu_model(model_path)
        elif config.gpu:
            logger.info(
                "Exporting LPR model to TensorRT for GPU acceleration..."
            )
            try:
                self.model = YOLO(model_path, task="detect")
                self.model.export(
                    format="engine", device=config.gpu, half=True
                )
                self.model = YOLO(TENSORRT_PATH, task="detect")
                logger.info(
                    "TensorRT LPR model exported and loaded successfully"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to export to TensorRT: {e}. Falling back to CPU model."
                )
                self._load_cpu_model(model_path)
        else:
            self._load_cpu_model(model_path)

        # Cache class names after model is loaded
        self.lpr_class_names = self.model.names

    def _load_cpu_model(self, model_path: str) -> None:
        if os.path.exists(os.path.join(QUANTIZED_PATH, "lpr_nano.bin")):
            logger.info("Loading OpenVINO LPR model for CPU...")
            self.model = YOLO(QUANTIZED_PATH, task="detect")
        else:
            logger.info(
                "Exporting LPR model to OpenVINO for CPU optimization..."
            )
            try:
                self.model = YOLO(model_path, task="detect")
                self.model.export(format="openvino", dynamic=True, half=True)
                self.model = YOLO(QUANTIZED_PATH, task="detect")
                logger.info(
                    "OpenVINO LPR model exported and loaded successfully"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to export to OpenVINO: {e}. Using standard PyTorch model."
                )
                self.model = YOLO(model_path, task="detect")

    def preprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        start_time = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        equ_bgr = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
        blurred = cv2.GaussianBlur(equ_bgr, (0, 0), 3)
        sharpened = cv2.addWeighted(equ_bgr, 1.5, blurred, -0.5, 0)
        logger.debug(
            f"Preprocessing time: {(time.time() - start_time) * 1000:.2f} ms"
        )
        return sharpened

    def find_plate(self, image: NDArray[Any]) -> Optional[NDArray[Any]]:
        start_time = time.time()
        preprocessed = self.preprocess_image(image)
        results: List[Any] = self.model.predict(
            preprocessed,
            conf=0.25,
            iou=0.5,
            verbose=False,
            imgsz=PLATE_DETECTION_SIZE,
        )

        if not results or len(results[0].boxes) == 0:
            logger.debug(
                f"Plate detection failed - No boxes found. Time: {(time.time() - start_time) * 1000:.2f} ms"
            )
            return None

        plate_boxes = [
            box.cpu().numpy()
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls)
            if self.lpr_class_names[int(cls)] == "License Plate"
        ]
        if not plate_boxes:
            logger.debug(
                f"Plate detection failed - No license plate class found. Time: {(time.time() - start_time) * 1000:.2f} ms"
            )
            return None

        best_box = plate_boxes[0]
        x1, y1, x2, y2 = best_box
        h, w = preprocessed.shape[:2]
        pad_x = (x2 - x1) * 0.1
        pad_y = (y2 - y1) * 0.1
        x1, y1, x2, y2 = (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(w, int(x2 + pad_x)),
            min(h, int(y2 + pad_y)),
        )
        logger.debug(
            f"Plate detected and cropped. Time: {(time.time() - start_time) * 1000:.2f} ms"
        )
        return preprocessed[y1:y2, x1:x2]

    def recognize_plate(self, image: NDArray[Any]) -> Optional[str]:
        start_time = time.time()
        logger.info("Starting license plate recognition...")

        plate_img = self.find_plate(image)
        if plate_img is None:
            logger.info(
                f"Recognition failed - No plate detected. Total time: {(time.time() - start_time) * 1000:.2f} ms"
            )
            return None

        results: List[Any] = self.model.predict(
            plate_img,
            conf=0.25,
            iou=0.45,
            imgsz=PLATE_DETECTION_SIZE,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            logger.info(
                f"Recognition failed - No characters detected on plate. Total time: {(time.time() - start_time) * 1000:.2f} ms"
            )
            return None

        boxes_and_classes = sorted(
            [
                (float(box[0]), self.lpr_class_names[int(cls)])
                for box, cls in zip(
                    results[0].boxes.xyxy, results[0].boxes.cls
                )
            ],
            key=lambda x: x[0],
        )
        unmapped_chars = [
            cls for _, cls in boxes_and_classes if cls in ARABIC_MAPPING
        ]
        if not unmapped_chars:
            logger.info(
                f"Recognition failed - No valid characters detected. Total time: {(time.time() - start_time) * 1000:.2f} ms"
            )
            return None

        mapped_chars = [ARABIC_MAPPING.get(c, c) for c in unmapped_chars]
        license_text = "".join(mapped_chars)
        logger.info(
            f"Recognition completed - Plate: {license_text}. Total time: {(time.time() - start_time) * 1000:.2f} ms"
        )
        return license_text
