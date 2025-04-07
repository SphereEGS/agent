import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import shutil
from PIL import Image, ImageDraw, ImageFont

from .config import (
    ARABIC_MAPPING,
    FONT_PATH,
    LPR_MODEL_PATH,
    logger,
)

# Use a lower resolution for license plate detection to speed up inference
PLATE_DETECTION_SIZE = 2048

def preprocess_image(image):
    """
    Preprocess the vehicle snapshot to improve license plate recognition.
    Steps:
      1. Convert to grayscale.
      2. Perform histogram equalization for contrast enhancement.
      3. Convert back to BGR.
      4. Apply an unsharp mask to sharpen the image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Equalize histogram to enhance contrast
    equ = cv2.equalizeHist(gray)
    # Convert back to BGR format
    equ_bgr = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
    # Apply unsharp mask for sharpening
    blurred = cv2.GaussianBlur(equ_bgr, (0, 0), 3)
    sharpened = cv2.addWeighted(equ_bgr, 1.5, blurred, -0.5, 0)
    return sharpened

class PlateProcessor:
    """
    Processes vehicle images to detect and recognize license plates.
    """
    def __init__(self):
        logger.info("Initializing license plate recognition model...")
        os.makedirs("models", exist_ok=True)
        os.makedirs("output/plates", exist_ok=True)
        try:
            if not os.path.exists(LPR_MODEL_PATH):
                logger.info("Downloading LPR model for the first time...")
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download("omarelsayeed/licence_plates") 
                source_model = os.path.join(model_dir, "license_yolo8s_1024.pt")
                shutil.copy2(source_model, LPR_MODEL_PATH)
                logger.info("LPR model downloaded successfully")
            
            self.lpr_model = YOLO(LPR_MODEL_PATH)
            logger.info("LPR model loaded successfully")
            self.font_path = FONT_PATH
            
        except Exception as e:
            logger.error(f"Error initializing license plate model: {str(e)}")
            raise

    def find_best_plate_in_image(self, image):
        if image is None:
            logger.warning("Empty image provided to find_best_plate_in_image")
            return None
            
        try:
            # Preprocess the image before detection
            preprocessed = preprocess_image(image)
            
            results = self.lpr_model.predict(
                preprocessed, 
                conf=0.25,
                iou=0.5,
                verbose=False,
                #imgsz=PLATE_DETECTION_SIZE
            )
            
            if not results or len(results[0].boxes) == 0:
                logger.info("No license plate detected in the image")
                return None

            plate_boxes = []
            plate_scores = []
            lpr_class_names = self.lpr_model.names
            
            for box, cls, conf in zip(
                results[0].boxes.xyxy, 
                results[0].boxes.cls, 
                results[0].boxes.conf
            ):
                if lpr_class_names[int(cls)] == "License Plate":
                    plate_boxes.append(box.cpu().numpy())
                    plate_scores.append(float(conf))

            if not plate_boxes:
                logger.info("No license plates found in the detections")
                return None

            plate_boxes = np.array(plate_boxes)
            plate_scores = np.array(plate_scores)
            best_idx = np.argmax(plate_scores)
            if plate_scores[best_idx] < 0.6:
                logger.info(f"Best plate detection has low confidence: {plate_scores[best_idx]:.2f}")
                return None

            h, w = preprocessed.shape[:2]
            x1, y1, x2, y2 = plate_boxes[best_idx]
            pad_x = (x2 - x1) * 0.1
            pad_y = (y2 - y1) * 0.1
            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(w, int(x2 + pad_x))
            y2 = min(h, int(y2 + pad_y))
            plate_img = preprocessed[y1:y2, x1:x2]
            logger.info(f"License plate detected with confidence: {plate_scores[best_idx]:.2f}")
            return plate_img

        except Exception as e:
            logger.error(f"Error detecting license plate: {str(e)}")
            return None

    def recognize_plate(self, plate_image):
        if plate_image is None:
            logger.warning("Empty plate image provided to recognize_plate")
            return None
            
        try:
            results = self.lpr_model.predict(
                plate_image, 
                conf=0.25,
                iou=0.45,
                #imgs=PLATE_DETECTION_SIZE,
                verbose=False
            )
            
            if not results or len(results[0].boxes) == 0:
                logger.info("No characters detected on license plate")
                return None

            lpr_class_names = self.lpr_model.names
            boxes_and_classes = [
                (float(box[0]), float(box[2]), lpr_class_names[int(cls)], conf)
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
            license_text = "".join([ARABIC_MAPPING.get(c, c) for c in unmapped_chars if c in ARABIC_MAPPING])
            if license_text:
                logger.info(f"License plate recognized: {license_text}")
                return license_text
            else:
                logger.info("No valid characters found on license plate")
                return None

        except Exception as e:
            logger.error(f"Error recognizing license plate: {str(e)}")
            return None

    def add_text_to_image(self, image, text):
        if not text or image is None:
            return image
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            height, _ = image.shape[:2]
            font_size = int(height / 15)
            try:
                font = ImageFont.truetype(self.font_path, font_size) if self.font_path else ImageFont.load_default()
            except Exception as e:
                logger.warning(f"Error loading font: {str(e)}. Using default font.")
                font = ImageFont.load_default()
            draw = ImageDraw.Draw(pil_image)
            separated_text = "-".join(text)
            padding = 20
            text_bbox = draw.textbbox((0, 0), separated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = padding
            y = height - text_height - padding * 2
            background_coords = [x - padding, y - padding, x + text_width + padding, y + text_height + padding]
            draw.rectangle(background_coords, fill=(0, 0, 0, 180))
            draw.text((x, y), separated_text, font=font, fill=(255, 255, 255))
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"Could not add text to image: {str(e)}")
            return image

    def process_vehicle_image(self, vehicle_image, save_path=None):
        try:
            # Preprocess the entire vehicle image before processing the license plate
            preprocessed_vehicle = preprocess_image(vehicle_image)
            plate_image = self.find_best_plate_in_image(preprocessed_vehicle)
            if plate_image is None:
                logger.info("No license plate found in vehicle image")
                return None, None
            plate_text = self.recognize_plate(plate_image)
            if plate_text is None:
                logger.info("Could not recognize text on license plate")
                return None, None
            processed_image = self.add_text_to_image(plate_image, plate_text)
            if save_path and processed_image is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, processed_image)
                logger.info(f"Saved processed plate image to {save_path}")
            return plate_text, processed_image
        except Exception as e:
            logger.error(f"Error processing vehicle image: {str(e)}")
            return None, None
