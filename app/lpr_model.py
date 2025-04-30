import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import shutil
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

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
        logger.info("Initializing license plate recognition system...")
        os.makedirs("models", exist_ok=True)
        os.makedirs("output/plates", exist_ok=True)
        try:
            # Initialize YOLO model for plate detection
            if not os.path.exists(LPR_MODEL_PATH):
                logger.info("Downloading LPR model for the first time...")
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download("omarelsayeed/licence_plates") 
                source_model = os.path.join(model_dir, "license_yolo8s_1024.pt")
                shutil.copy2(source_model, LPR_MODEL_PATH)
                logger.info("LPR model downloaded successfully")
            
            # Keep YOLO for plate detection
            self.lpr_model = YOLO(LPR_MODEL_PATH)
            logger.info("YOLO plate detection model loaded successfully")
            
            # Initialize PaddleOCR for Arabic text recognition
            logger.info("Initializing PaddleOCR for text recognition...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=True)
            logger.info("PaddleOCR initialized successfully")
            
            self.font_path = FONT_PATH
            
        except Exception as e:
            logger.error(f"Error initializing license plate system: {str(e)}")
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
            # 1. Enhance the plate image for better OCR
            enhanced = cv2.detailEnhance(plate_image, sigma_s=10, sigma_r=0.15)

            # 2. Convert to RGB (PaddleOCR expects RGB)
            rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            # 3. Write to a temporary file
            temp_path = "temp_plate.jpg"
            cv2.imwrite(temp_path, enhanced)

            # 4. Run PaddleOCR
            result = self.ocr.ocr(temp_path, cls=True)

            # 5. Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # 6. Check for no result
            if not result or len(result) == 0 or not result[0]:
                logger.info("No text detected on license plate by PaddleOCR")
                return None

            # 7. Collect _all_ high-confidence segments
            text_segments = []
            for line in result[0]:
                # line = [box, (text, confidence), ...]
                if len(line) >= 2 and isinstance(line[1], tuple):
                    text, confidence = line[1]
                    if confidence > 0.5:
                        text_segments.append(text)

            if not text_segments:
                logger.info("No confident text detected on license plate")
                return None

            # 8. Concatenate into one raw string
            raw_text = ''.join(text_segments)

            # 9. Remove non-alphanumerics
            cleaned = ''.join(ch for ch in raw_text if ch.isalnum())

            # 10. Map Arabic‐Indic digits/letters to Latin if needed
            processed_text = ''.join(ARABIC_MAPPING.get(c, c) for c in cleaned)

            if processed_text:
                logger.info(f"License plate recognized: {processed_text}")
                return processed_text
            else:
                logger.info("No valid characters found on license plate")
                return None

        except Exception as e:
            logger.error(f"Error recognizing license plate with PaddleOCR: {e}")
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
            # For Arabic license plates, we might want to display right-to-left
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

    def visualize_roi(self, image, roi_polygon=None):
        """
        Draw ROI visualization on the image for gate entry logging.
        If no ROI is available, returns the original image.
        
        Args:
            image (numpy.ndarray): The input image
            roi_polygon (numpy.ndarray, optional): ROI polygon points. If None, returns the original image.
        
        Returns:
            numpy.ndarray: Image with ROI drawn
        """
        if image is None:
            return None
        
        if roi_polygon is None:
            # Return original image if no ROI provided
            return image
        
        try:
            # Create a copy to avoid modifying the original
            result = image.copy()
            
            # Draw the ROI polygon
            cv2.polylines(result, [roi_polygon], True, (0, 255, 0), 3)
            
            # Add some transparency inside the ROI
            overlay = result.copy()
            cv2.fillPoly(overlay, [roi_polygon], (0, 200, 0, 50))
            alpha = 0.15
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            
            return result
        except Exception as e:
            logger.warning(f"Error visualizing ROI: {str(e)}")
            return image

    def process_vehicle_image(self, vehicle_image, save_path=None):
        try:
            # Preprocess the entire vehicle image before processing the license plate
            preprocessed_vehicle = preprocess_image(vehicle_image)
            
            # First detect the license plate using YOLO (unchanged)
            plate_image = self.find_best_plate_in_image(preprocessed_vehicle)
            if plate_image is None:
                logger.info("No license plate found in vehicle image")
                return None, None
                
            # Now use PaddleOCR to recognize the text
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