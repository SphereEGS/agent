import cv2
import json
from dotenv import load_dotenv
import os
import sys
import argparse
from app.config import FRAME_WIDTH, FRAME_HEIGHT

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='ROI selection tool for SpherexAgent')
parser.add_argument('--camera', type=str, default="main", help='Camera ID to configure (e.g., "main", "entry", "exit")')
args = parser.parse_args()

camera_id = args.camera
print(f"Configuring ROI for camera: {camera_id}")

# Get camera URL from environment variables based on camera ID
if camera_id == "main":
    camera_env_var = "CAMERA_URL"
else:
    # Check if camera_id starts with "camera_" and extract the number
    if camera_id.startswith("camera_"):
        camera_number = camera_id.split("_")[1]
        camera_env_var = f"CAMERA_URL_{camera_number}"
    else:
        # Fallback to the original format
        camera_env_var = f"CAMERA_{camera_id.upper()}_URL"

# Get video source from environment variables or use default
CAMERA_URL = os.getenv(camera_env_var, "0")  # Default to local webcam
VIDEO_PATH = os.getenv("VIDEO_PATH", "input/test_video3.mov")

# Clean any quotes from CAMERA_URL
if isinstance(CAMERA_URL, str):
    CAMERA_URL = CAMERA_URL.strip()
    if CAMERA_URL.startswith('"') and CAMERA_URL.endswith('"'):
        CAMERA_URL = CAMERA_URL[1:-1].strip()
    if '#' in CAMERA_URL:
        CAMERA_URL = CAMERA_URL.split('#')[0].strip()

print(f"Camera URL: {CAMERA_URL}")

# Use appropriate source based on CAMERA_URL
if CAMERA_URL.lower().endswith(('.mp4', '.mov', '.avi')):
    source_path = CAMERA_URL
    print(f"Using video file: {source_path}")
    if not os.path.exists(source_path):
        print(f"Error: Video file {source_path} not found.")
        exit(1)
elif CAMERA_URL == "0" or CAMERA_URL.isdigit():
    # Convert string to integer if needed for webcam
    source_path = 0 if CAMERA_URL == "0" else int(CAMERA_URL)
    print(f"Using webcam with index: {source_path}")
else:
    # Assume RTSP or other camera URL
    source_path = CAMERA_URL
    print(f"Attempting to connect to camera stream: {source_path}")

# Set RTSP transport to TCP for better reliability to match camera.py settings
if isinstance(source_path, str) and source_path.startswith('rtsp://'):
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
else:
    # Connect to the source
    cap = cv2.VideoCapture(source_path)

if not cap.isOpened():
    print(f"Failed to connect to: {source_path}")
    print("Falling back to local webcam (index 0)")
    source_path = 0
    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        print("Error: Unable to open any camera source")
        exit(1)

print(f"Successfully connected to: {'CCTV/File' if source_path != 0 else 'Local Webcam'}")

# Read a frame for ROI selection
ret, original_frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read frame from the source.")
    exit(1)

# Get original frame dimensions
original_width = original_frame.shape[1]
original_height = original_frame.shape[0]
print(f"Original frame dimensions: {original_width}x{original_height}")

# Calculate the target dimensions using the same method as in camera.py
target_width = FRAME_WIDTH
target_height = FRAME_HEIGHT

# Calculate aspect ratio preserving dimensions exactly as in camera.py
aspect_ratio = original_width / original_height
if aspect_ratio > (target_width / target_height):
    # Image is wider than target
    new_width = target_width
    new_height = int(target_width / aspect_ratio)
else:
    # Image is taller than target
    new_height = target_height
    new_width = int(target_height * aspect_ratio)

# Ensure dimensions are even numbers (required by some OpenCV operations)
new_width = new_width - (new_width % 2)
new_height = new_height - (new_height % 2)

print(f"Target display dimensions: {new_width}x{new_height}")

# Calculate scale ratios
scale_width_ratio = new_width / original_width
scale_height_ratio = new_height / original_height

# Resize the frame using the dimensions calculated exactly as in camera.py
display_frame = cv2.resize(original_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

polygon_points = []
polygon_finished = False

def mouse_callback(event, x, y, flags, param):
    global polygon_points, polygon_finished
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(polygon_points) >= 3:
            polygon_finished = True
        else:
            print("Need at least 3 points to form a polygon.")

window_name = "Draw ROI - Left Click: add point, Right Click: finish, Esc: exit"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("Instructions:")
print("  - Left-click to add polygon points.")
print("  - Right-click to complete the polygon (minimum 3 points).")
print("  - Press 'Esc' to exit without saving.")
print(f"Original frame: {original_width}x{original_height}, Display frame: {display_frame.shape[1]}x{display_frame.shape[0]}")

while True:
    frame_display = display_frame.copy()

    if polygon_points:
        for pt in polygon_points:
            cv2.circle(frame_display, pt, 4, (0, 0, 255), -1)
        for i in range(1, len(polygon_points)):
            cv2.line(frame_display, polygon_points[i - 1], polygon_points[i], (0, 255, 0), 2)
        if polygon_finished:
            cv2.line(frame_display, polygon_points[-1], polygon_points[0], (0, 255, 0), 2)

    cv2.imshow(window_name, frame_display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("Exiting without saving ROI.")
        break
    if polygon_finished:
        # Convert display coordinates back to original coordinates
        original_polygon_points = []
        for x, y in polygon_points:
            # Use the same scale ratios to convert back
            original_x = int(x / scale_width_ratio)
            original_y = int(y / scale_height_ratio)
            original_polygon_points.append((original_x, original_y))
        
        # Print information for debugging
        print(f"Original frame dimensions: {original_width}x{original_height}")
        print(f"Display frame dimensions: {display_frame.shape[1]}x{display_frame.shape[0]}")
        print(f"Scale width ratio: {scale_width_ratio}, height ratio: {scale_height_ratio}")
        print(f"Display polygon points: {polygon_points}")
        print(f"Original polygon points: {original_polygon_points}")
        
        # Also save the display polygon points for reference
        config_data = {
            "camera_id": camera_id,
            "original_points": original_polygon_points,
            "display_points": polygon_points,
            "original_dimensions": [original_width, original_height],
            "display_dimensions": [display_frame.shape[1], display_frame.shape[0]],
            "scale_ratios": [scale_width_ratio, scale_height_ratio]
        }
        
        # Save to config file with camera_id in the filename
        os.makedirs("configs", exist_ok=True)
        save_path = f"configs/roi_{camera_id}.json"
        with open(save_path, "w") as f:
            json.dump(config_data, f, indent=2)
        print(f"ROI configuration saved to {save_path}")
        print("ROI has been saved successfully!")
        cv2.waitKey(1000)  # Wait a bit longer to show success
        break

cv2.destroyAllWindows()
