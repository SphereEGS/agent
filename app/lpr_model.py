import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import shutil
from PIL import Image, ImageFont, ImageDraw
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import threading
import queue
from typing import Dict, Tuple, Optional, List, Callable, Any

from .config import (
    ARABIC_MAPPING,
    FONT_PATH,
    LPR_MODEL_PATH,
    logger,
)

# Use a lower resolution for license plate detection to speed up inference
PLATE_DETECTION_SIZE = 480

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

# Helper functions that will be called by ProcessPoolExecutor
def _find_plate_in_image(model_path, image):
    try:
        model = YOLO(model_path)
        # Preprocess the image before detection
        preprocessed = preprocess_image(image)
        
        results = model.predict(
            preprocessed, 
            conf=0.25,
            iou=0.5,
            verbose=False,
            imgsz=PLATE_DETECTION_SIZE
        )
        
        if not results or len(results[0].boxes) == 0:
            return None, None, None

        plate_boxes = []
        plate_scores = []
        lpr_class_names = model.names
        
        for box, cls, conf in zip(
            results[0].boxes.xyxy, 
            results[0].boxes.cls, 
            results[0].boxes.conf
        ):
            if lpr_class_names[int(cls)] == "License Plate":
                plate_boxes.append(box.cpu().numpy())
                plate_scores.append(float(conf))

        if not plate_boxes:
            return None, None, None

        plate_boxes = np.array(plate_boxes)
        plate_scores = np.array(plate_scores)
        best_idx = np.argmax(plate_scores)
        if plate_scores[best_idx] < 0.6:
            return None, None, None

        h, w = preprocessed.shape[:2]
        x1, y1, x2, y2 = plate_boxes[best_idx]
        pad_x = (x2 - x1) * 0.1
        pad_y = (y2 - y1) * 0.1
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))
        plate_img = preprocessed[y1:y2, x1:x2]
        
        return plate_img, plate_scores[best_idx], preprocessed
    except Exception as e:
        logger.error(f"Error in _find_plate_in_image: {str(e)}")
        return None, None, None

def _recognize_plate(model_path, plate_image):
    try:
        model = YOLO(model_path)
        results = model.predict(
            plate_image, 
            conf=0.25,
            iou=0.45,
            imgsz=PLATE_DETECTION_SIZE,
            verbose=False
        )
        
        if not results or len(results[0].boxes) == 0:
            return None

        lpr_class_names = model.names
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
        
        return license_text if license_text else None
    except Exception as e:
        logger.error(f"Error in _recognize_plate: {str(e)}")
        return None

class PlateProcessor:
    """
    Processes vehicle images to detect and recognize license plates.
    """
    def __init__(self, max_workers=None):
        logger.info("Initializing license plate recognition model...")
        os.makedirs("models", exist_ok=True)
        os.makedirs("output/plates", exist_ok=True)
        
        # Set default max_workers to number of CPUs
        if max_workers is None:
            max_workers = max(2, multiprocessing.cpu_count() - 1)
        self.max_workers = max_workers
        
        try:
            if not os.path.exists(LPR_MODEL_PATH):
                logger.info("Downloading LPR model for the first time...")
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download("omarelsayeed/licence_plates") 
                source_model = os.path.join(model_dir, "license_yolo8s_1024.pt")
                shutil.copy2(source_model, LPR_MODEL_PATH)
                logger.info("LPR model downloaded successfully")
            
            # Initialize a local model for synchronous operations
            self.lpr_model = YOLO(LPR_MODEL_PATH)
            logger.info("LPR model loaded successfully")
            self.font_path = FONT_PATH
            
            # Initialize thread pool for lighter tasks
            self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers*2)
            
            # Initialize the process pool executor
            self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
            
            # Initialize a task queue for non-blocking operation
            self.task_queue = queue.Queue()
            self.results = {}
            self.results_lock = threading.Lock()
            self.next_task_id = 0
            self.task_id_lock = threading.Lock()
            
            # Start worker thread to process tasks
            self.stop_event = threading.Event()
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()
            
            logger.info(f"Initialized plate processor with {self.max_workers} workers")
            
        except Exception as e:
            logger.error(f"Error initializing license plate model: {str(e)}")
            raise

    def __del__(self):
        # Clean up resources
        self.shutdown()

    def shutdown(self):
        """Properly shutdown all resources"""
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
            
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            
        if hasattr(self, 'process_executor'):
            self.process_executor.shutdown(wait=False)
            
        if hasattr(self, 'thread_executor'):
            self.thread_executor.shutdown(wait=False)
            
        if hasattr(self, 'task_queue'):
            # Clear queue
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break

    def _get_next_task_id(self):
        """Generate a unique task ID"""
        with self.task_id_lock:
            task_id = self.next_task_id
            self.next_task_id += 1
        return task_id

    def _process_queue(self):
        """Background thread that processes the task queue"""
        while not self.stop_event.is_set():
            try:
                # Get task with timeout to allow checking stop_event periodically
                try:
                    task_id, image, save_path, callback = self.task_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                try:
                    # Process the image
                    plate_text, processed_image = self._process_image_worker(image, save_path)
                    
                    # Store or callback with result
                    if callback:
                        try:
                            callback(plate_text, processed_image)
                        except Exception as cb_error:
                            logger.error(f"Error in callback for task {task_id}: {str(cb_error)}")
                    else:
                        with self.results_lock:
                            self.results[task_id] = (plate_text, processed_image)
                            
                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {str(e)}")
                    # Store error result
                    if callback:
                        try:
                            callback(None, None)
                        except Exception:
                            pass
                    else:
                        with self.results_lock:
                            self.results[task_id] = (None, None)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing queue: {str(e)}")
                # Small sleep to prevent tight loop in case of recurring errors
                time.sleep(0.1)
    
    def _process_image_worker(self, image, save_path=None):
        """Worker function that processes images in the background thread"""
        try:
            # Need to ensure image is not None
            if image is None:
                logger.warning("Empty image provided to process_image_worker")
                return None, None
            
            # Use synchronous processing but in background thread
            plate_image, _, _ = _find_plate_in_image(LPR_MODEL_PATH, image)
            if plate_image is None:
                return None, None
                
            plate_text = _recognize_plate(LPR_MODEL_PATH, plate_image)
            if plate_text is None:
                return None, None
                
            processed_image = self.add_text_to_image(plate_image, plate_text)
            
            # Save the image if requested
            if save_path and processed_image is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, processed_image)
                
            return plate_text, processed_image
            
        except Exception as e:
            logger.error(f"Error in process_image_worker: {str(e)}")
            return None, None

    def find_best_plate_in_image(self, image):
        """
        Synchronous method to find the best license plate in an image.
        Maintains backwards compatibility with existing code.
        """
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
                imgsz=PLATE_DETECTION_SIZE
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
        """
        Synchronous method to recognize text on a license plate.
        Maintains backwards compatibility with existing code.
        """
        if plate_image is None:
            logger.warning("Empty plate image provided to recognize_plate")
            return None
            
        try:
            results = self.lpr_model.predict(
                plate_image, 
                conf=0.25,
                iou=0.45,
                imgsz=PLATE_DETECTION_SIZE,
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
        """Add recognized license plate text to the image"""
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
        """
        Synchronous method for processing a vehicle image and extracting the license plate.
        Maintains backwards compatibility with existing code.
        """
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

    def submit_image(self, vehicle_image, save_path=None, callback=None):
        """
        Submit an image for non-blocking, asynchronous processing.
        
        Args:
            vehicle_image: The vehicle image to process
            save_path: Optional path to save the processed plate image
            callback: Optional callback function(plate_text, processed_image) to call when processing completes
            
        Returns:
            task_id: A unique ID for this task, can be used to get results later if no callback provided
        """
        if vehicle_image is None:
            logger.warning("Empty image provided to submit_image")
            if callback:
                callback(None, None)
                return None
            return None
            
        task_id = self._get_next_task_id()
        self.task_queue.put((task_id, vehicle_image, save_path, callback))
        return task_id
        
    def get_result(self, task_id, timeout=None):
        """
        Get the result of a previously submitted image processing task.
        
        Args:
            task_id: The task ID returned from submit_image
            timeout: Maximum time to wait for the result (None = wait forever)
            
        Returns:
            (plate_text, processed_image) or None if timeout or task not found
        """
        if task_id is None:
            return None, None
            
        end_time = time.time() + timeout if timeout else None
        
        while end_time is None or time.time() < end_time:
            with self.results_lock:
                if task_id in self.results:
                    result = self.results.pop(task_id)
                    return result
            
            # Small sleep to prevent tight loop
            time.sleep(0.01)
            
        return None, None
