import time
import os
import numpy as np
from ultralytics import YOLO

# Configuration
CAMERA_URL = "./tracking.mp4"  # Replace with your camera URL
MODEL_PATH = "resources/yolo11n.pt"
TENSORRT_PATH = "resources/yolo11n.engine"
GPU_ENABLED = False  # Set to False if you don't want to use GPU

def main():
    print(f"Testing YOLO detection FPS on camera: {CAMERA_URL}")

    # Initialize YOLO model
    if GPU_ENABLED and os.path.exists(TENSORRT_PATH):
        print("Loading TensorRT YOLO model for GPU acceleration...")
        try:
            model = YOLO(TENSORRT_PATH, task="detect")
            print("TensorRT model loaded successfully")
        except Exception as e:
            print(f"Failed to load TensorRT: {e}. Falling back to CPU model.")
            model = YOLO(MODEL_PATH, task="detect")
    elif GPU_ENABLED:
        print("Exporting YOLO model to TensorRT...")
        try:
            model = YOLO(MODEL_PATH, task="detect")
            model.export(format="engine", device=0, half=True)
            model = YOLO(TENSORRT_PATH, task="detect")
            print("TensorRT model exported and loaded successfully")
        except Exception as e:
            print(f"Failed to export to TensorRT: {e}. Falling back to CPU model.")
            model = YOLO(MODEL_PATH, task="detect")
    else:
        print("Loading standard YOLO model for CPU...")
        model = YOLO(MODEL_PATH, task="detect")

    # Initialize video stream
    max_retries = 5
    for attempt in range(max_retries):
        try:
            results = model.track(
                source=CAMERA_URL,
                stream=True,
                persist=True,
                classes=[2, 3, 5, 7],  # Detect vehicles (cars, buses, trucks, etc.)
                verbose=False,
            )
            print(f"Successfully connected to camera stream on attempt {attempt + 1}")
            break
        except Exception as e:
            print(f"Failed to start stream (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(1)
    else:
        print(f"Could not open video stream after {max_retries} attempts")
        return

    # Measure FPS and vehicle detections
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    interval = 1.0  # Log every 1 second

    for result in results:
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - start_time

        # Count vehicle detections
        if result.boxes and len(result.boxes) > 0:
            detection_count += len(result.boxes)

        if elapsed >= interval:
            fps = frame_count / elapsed
            detections_per_second = detection_count / elapsed
            print(f"Frames per second (FPS): {fps:.2f}")
            print(f"Vehicle detections per second: {detections_per_second:.2f}")
            # Reset counters
            frame_count = 0
            detection_count = 0
            start_time = current_time

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"An error occurred: {e}")