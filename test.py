import cv2
import numpy as np
import os
import time
import json
from collections import defaultdict
from ultralytics import YOLO
import argparse
from dotenv import load_dotenv
import threading

# Import the plate processor
from app.lpr_model import PlateProcessor

# Vehicle classes from COCO dataset
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Processing parameters for optimization
TARGET_WIDTH = 2048      # Resize frames to this width for faster processing
PROCESS_EVERY = 4      # Process every N-th frame

class VehicleTracker:
    def __init__(self, model_path, roi_config_path=None):
        """
        Initialize the vehicle tracker with YOLO model and ROI.
        """
        self.model = YOLO(model_path)
        self.original_roi = self._load_roi_polygon(roi_config_path)
        self.roi_polygon = None  # will be computed based on resized frame
        self.roi_lock = threading.Lock()  # lock for accessing ROI polygon
        
        # Initialize plate processor
        self.plate_processor = PlateProcessor()
        
        # Vehicle tracking state
        self.tracked_vehicles = {}         # Track seen vehicles (track_id -> timestamp)
        self.plate_attempts = defaultdict(int)  # Count attempts per vehicle
        self.detected_plates = {}          # Store detected plate numbers
        self.max_attempts = 3              # Maximum attempts to detect a plate
        
        # Create output directories
        os.makedirs("output/vehicles", exist_ok=True)
        os.makedirs("output/plates", exist_ok=True)
        
        print(f"Vehicle tracker initialized with model: {model_path}")
        if self.original_roi is not None:
            print(f"ROI loaded with {len(self.original_roi)} points")
        else:
            print("No ROI defined. Using full frame.")

    def _load_roi_polygon(self, config_path):
        """Load ROI polygon from config file."""
        if config_path is None or not os.path.exists(config_path):
            return None
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            if isinstance(config_data, list) and len(config_data) >= 3:
                roi_polygon = np.array(config_data, dtype=np.int32)
                if roi_polygon.ndim == 2 and roi_polygon.shape[1] == 2:
                    return roi_polygon
            print(f"Invalid ROI format in config: {config_path}")
            return None
        except Exception as e:
            print(f"Error loading ROI polygon: {str(e)}")
            return None

    def _scale_roi(self, frame_width, frame_height, orig_width, orig_height):
        """
        Scale the original ROI polygon to match the resized frame.
        """
        if self.original_roi is None:
            return None
        scale_x = frame_width / orig_width
        scale_y = frame_height / orig_height
        scaled_roi = np.copy(self.original_roi)
        scaled_roi[:, 0] = (scaled_roi[:, 0] * scale_x).astype(np.int32)
        scaled_roi[:, 1] = (scaled_roi[:, 1] * scale_y).astype(np.int32)
        return scaled_roi

    def _is_point_inside_roi(self, point):
        """Check if a point is inside the ROI polygon."""
        with self.roi_lock:
            if self.roi_polygon is None:
                return True
            pt = (int(point[0]), int(point[1]))
            try:
                result = cv2.pointPolygonTest(self.roi_polygon, pt, False)
                return result >= 0
            except Exception as e:
                print(f"Error checking ROI point: {str(e)}")
                return True

    def _is_vehicle_in_roi(self, box):
        """Check if the center of the vehicle is inside the ROI."""
        if self.roi_polygon is None:
            return True
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return self._is_point_inside_roi((center_x, center_y))
    
    def _is_vehicle_front_in_roi(self, box):
        """
        Check if the front of the vehicle is inside the ROI.
        Here we define the "front" as the midpoint of the bottom edge of the bounding box.
        """
        if self.roi_polygon is None:
            return True
        x1, y1, x2, y2 = map(int, box)
        front_point = ((x1 + x2) // 2, y2)
        return self._is_point_inside_roi(front_point)
        
    def _capture_vehicle_snapshot(self, frame, box, track_id):
        """
        Capture a snapshot of the vehicle for license plate detection.
        """
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            pad_x = (x2 - x1) * 0.1
            pad_y = (y2 - y1) * 0.1
            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(w, int(x2 + pad_x))
            y2 = min(h, int(y2 + pad_y))
            vehicle_img = frame[y1:y2, x1:x2].copy()
            timestamp = int(time.time())
            image_path = f"output/vehicles/vehicle_{track_id}_{timestamp}.jpg"
            cv2.imwrite(image_path, vehicle_img)
            print(f"Saved vehicle snapshot: {image_path}")
            return vehicle_img
        except Exception as e:
            print(f"Error capturing vehicle snapshot: {str(e)}")
            return None

    def _process_plate_async(self, frame, box, track_id):
        """Thread target to process license plate detection."""
        # Only process if the front of the vehicle is inside the ROI
        if not self._is_vehicle_front_in_roi(box):
            print(f"Vehicle {track_id} front is outside ROI. Skipping snapshot.")
            return
        vehicle_img = self._capture_vehicle_snapshot(frame, box, track_id)
        if vehicle_img is None:
            return
        timestamp = int(time.time())
        save_path = f"output/plates/plate_{track_id}_{timestamp}.jpg"
        plate_text, _ = self.plate_processor.process_vehicle_image(vehicle_img, save_path)
        if plate_text:
            self.detected_plates[track_id] = plate_text
            print(f"Detected plate: {plate_text} for vehicle {track_id}")

    def detect_and_track(self, frame, orig_dims):
        """
        Detect and track vehicles in a frame.
        Args:
            frame: Resized video frame for processing.
            orig_dims: Tuple (orig_width, orig_height) of the full-resolution frame.
        Returns:
            Frame with visualizations.
        """
        vis_frame = frame.copy()
        with self.roi_lock:
            if self.original_roi is not None:
                resized_height, resized_width = frame.shape[:2]
                orig_width, orig_height = orig_dims
                self.roi_polygon = self._scale_roi(resized_width, resized_height, orig_width, orig_height)
                if self.roi_polygon is not None:
                    cv2.polylines(vis_frame, [self.roi_polygon], True, (0, 255, 0), 2)

        results = self.model.track(
            frame, 
            persist=True, 
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.3,
            iou=0.45,
            verbose=False
        )
        
        if results and len(results) > 0 and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id not in VEHICLE_CLASSES:
                    continue
                class_name = VEHICLE_CLASSES[class_id]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, f"ID: {track_id} {class_name}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Only process if the vehicle's center is inside ROI
                if self._is_vehicle_in_roi(box):
                    if (track_id not in self.tracked_vehicles or 
                        (track_id not in self.detected_plates and self.plate_attempts[track_id] < self.max_attempts)):
                        self.tracked_vehicles[track_id] = time.time()
                        
                        # If already detected, just draw the plate text
                        if track_id in self.detected_plates:
                            plate_text = self.detected_plates[track_id]
                            cv2.putText(vis_frame, f"Plate: {plate_text}", (x1, y2+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            continue
                        
                        self.plate_attempts[track_id] += 1
                        print(f"Attempt {self.plate_attempts[track_id]} for vehicle {track_id}")
                        # Process license plate in a separate thread
                        threading.Thread(target=self._process_plate_async, args=(frame, box, track_id), daemon=True).start()
                        
                        if track_id in self.detected_plates:
                            plate_text = self.detected_plates[track_id]
                            cv2.putText(vis_frame, f"Plate: {plate_text}", (x1, y2+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.putText(vis_frame, f"Vehicles in ROI: {len(self.tracked_vehicles)}", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_frame, f"Plates Detected: {len(self.detected_plates)}", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vis_frame

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vehicle and License Plate Detection")
    parser.add_argument("--model", type=str, default="models/yolo11n.pt",
                        help="Path to YOLO model file")
    parser.add_argument("--source", type=str, default="input/test_video3.mov",
                        help="Path to video file or camera URL")
    parser.add_argument("--roi", type=str, default="config.json",
                        help="Path to ROI configuration file")
    parser.add_argument("--output", type=str, default="output/processed_video.mp4",
                        help="Path to output video file")
    return parser.parse_args()

def main():
    """Main function."""
    load_dotenv()
    args = parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
        
    if not args.source.startswith(('rtsp://', 'http://', 'https://')):
        if not os.path.exists(args.source):
            print(f"Error: Video file not found: {args.source}")
            return
            
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return
        
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ratio = TARGET_WIDTH / float(orig_width)
    new_height = int(orig_height * ratio)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (TARGET_WIDTH, new_height))
    
    tracker = VehicleTracker(args.model, args.roi)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % PROCESS_EVERY != 0:
                continue
            
            resized_frame = cv2.resize(frame, (TARGET_WIDTH, new_height))
            processed_frame = tracker.detect_and_track(resized_frame, (orig_width, orig_height))
            
            out.write(processed_frame)
            cv2.imshow("Vehicle and License Plate Detection", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.2f} fps)")
                
    except KeyboardInterrupt:
        print("Processing interrupted")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print("\nProcessing complete:")
        print(f"Total frames processed: {frame_count}")
        print(f"Total processing time: {total_time:.2f}s")
        plates = tracker.detected_plates
        print(f"Total plates detected: {len(plates)}")
        for track_id, plate in plates.items():
            print(f"Vehicle ID {track_id}: Plate {plate}")
        print(f"Output video saved to: {args.output}")

if __name__ == "__main__":
    main()
