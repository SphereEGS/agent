import json
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
        self.roi_polygon = self._load_roi_polygon("config.json")
        
        if self.roi_polygon is not None:
            logger.info(f"ROI polygon loaded successfully with {len(self.roi_polygon)} points")
            logger.info(f"ROI polygon points: {self.roi_polygon.tolist()}")

    def _load_roi_polygon(self, config_path):
        """Load ROI polygon from config.json."""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            # Check if config data is a list of points
            if isinstance(config_data, list) and len(config_data) >= 3:
                # Convert to numpy array and reshape if needed
                roi_polygon = np.array(config_data, dtype=np.int32)
                # Ensure shape is correct for pointPolygonTest
                if len(roi_polygon.shape) == 2 and roi_polygon.shape[1] == 2:
                    logger.info(f"Valid ROI polygon loaded with {len(roi_polygon)} points")
                    return roi_polygon
                else:
                    logger.error(f"Invalid ROI polygon shape: {roi_polygon.shape}")
                    return None
            else:
                logger.error(f"Invalid ROI format in config: {config_data}")
                return None
        except FileNotFoundError:
            logger.warning(f"ROI config file not found: {config_path}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in: {config_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading ROI polygon: {str(e)}")
            return None

    def _is_point_inside_roi(self, point):
        """Check if a point is inside the ROI polygon."""
        if self.roi_polygon is None:
            return True
            
        # Make sure point is a tuple of two integers
        point = (int(point[0]), int(point[1]))
        
        try:
            # pointPolygonTest returns positive for inside, negative for outside, 0 for on the boundary
            result = cv2.pointPolygonTest(self.roi_polygon, point, False)
            return result >= 0
        except Exception as e:
            logger.error(f"Error checking if point {point} is in ROI: {str(e)}")
            # Return True as a fallback to not block detections
            return True

    def _is_box_inside_roi(self, box):
        """Check if a bounding box is inside or overlaps with the ROI polygon.
        This is more strict than just checking the center point."""
        if self.roi_polygon is None:
            return True
            
        x1, y1, x2, y2 = map(int, box)
        
        # Check all four corners of the box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return any(self._is_point_inside_roi(corner) for corner in corners)

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
            x1, y1, x2, y2 = box.cpu().numpy()
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Debug ROI check
            is_inside = self._is_box_inside_roi((x1, y1, x2, y2))
            logger.debug(f"Box at ({center_x}, {center_y}), is inside ROI: {is_inside}")
            
            if is_inside and int(cls) == 0:  # Check if class is "License Plate"
                plate_boxes.append(box.cpu().numpy())
                plate_scores.append(float(conf))

        if not plate_boxes:
            logger.debug("No plates found inside ROI")
            return None

        plate_boxes = np.array(plate_boxes)
        plate_scores = np.array(plate_scores)
        best_idx = np.argmax(plate_scores)

        if plate_scores[best_idx] < 0.6:
            logger.debug(f"Best plate score {plate_scores[best_idx]} below threshold")
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
        if cropped_plate is None:
            return None
            
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

    def visualize_roi(self, image):
        """Draw the ROI polygon on the image for debugging"""
        if self.roi_polygon is None:
            return image
            
        img_copy = image.copy()
        cv2.polylines(img_copy, [self.roi_polygon], True, (0, 255, 0), 2)
        return img_copy
