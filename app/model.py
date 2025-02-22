import os
import shutil

import cv2
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from app.config import ARABIC_MAPPING, FONT_PATH, MODEL_PATH, logger


class PlateDetector:
    def __init__(self):
        os.makedirs("models", exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            logger.info("Downloading model for the first time...")
            model_dir = snapshot_download("omarelsayeed/licence_plates")
            source_model = os.path.join(model_dir, "license_yolo8s_1024.pt")
            shutil.copy2(source_model, MODEL_PATH)
            logger.info("Model downloaded successfully")

        self.model = YOLO(MODEL_PATH)
        self.font_path = FONT_PATH

    def detect_and_crop_plate(self, image):
        results = self.model.predict(
            image, conf=0.25, iou=0.45, verbose=False, imgsz=1024
        )
        if not results or len(results[0].boxes) == 0:
            return None

        plate_boxes = []
        plate_scores = []
        for box, cls, conf in zip(
            results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
        ):
            if self.model.names[int(cls)] == "License Plate":
                plate_boxes.append(box.cpu().numpy())
                plate_scores.append(float(conf))

        if not plate_boxes:
            return None

        plate_boxes = np.array(plate_boxes)
        plate_scores = np.array(plate_scores)
        best_idx = np.argmax(plate_scores)

        if plate_scores[best_idx] < 0.6:
            return None

        h, w = image.shape[:2]
        x1, y1, x2, y2 = plate_boxes[best_idx]
        pad_x = (x2 - x1) * 0.1
        pad_y = (y2 - y1) * 0.1

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        return image[y1:y2, x1:x2]

    def recognize_plate(self, cropped_plate):
        results = self.model.predict(
            cropped_plate, conf=0.25, iou=0.45, imgsz=1024, verbose=False
        )
        if not results:
            return None

        boxes_and_classes = [
            (float(box[0]), float(box[2]), self.model.names[int(cls)], conf)
            for box, cls, conf in zip(
                results[0].boxes.xyxy,
                results[0].boxes.cls,
                results[0].boxes.conf,
            )
        ]

        boxes_and_classes.sort(key=lambda b: b[0])
        unmapped_chars = [
            cls for _, _, cls, _ in boxes_and_classes if cls in ARABIC_MAPPING
        ]

        license_text = "".join(
            [
                ARABIC_MAPPING.get(c, c)
                for c in unmapped_chars
                if c in ARABIC_MAPPING
            ]
        )
        return "".join(license_text) if license_text else None

    def add_text_to_image(self, image, text):
        if not text:
            return image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        height, _ = image.shape[:2]
        font_size = int(height / 15)

        try:
            font = (
                ImageFont.truetype(self.font_path, font_size)
                if self.font_path
                else ImageFont.load_default()
            )
            draw = ImageDraw.Draw(pil_image)

            separated_text = "-".join(text)
            padding = 20

            text_bbox = draw.textbbox((0, 0), separated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = padding
            y = height - text_height - padding * 2

            background_coords = [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding,
            ]
            draw.rectangle(background_coords, fill=(0, 0, 0, 180))
            draw.text((x, y), separated_text, font=font, fill=(255, 255, 255))

            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.warning(f"Warning: Could not add text to image: {str(e)}")
            return image
