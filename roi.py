import cv2
import json
from dotenv import load_dotenv
import os
import sys
from app.config import FRAME_WIDTH, FRAME_HEIGHT

load_dotenv()

# Get video source from environment variables or use default
CAMERA_URL = os.getenv("CAMERA_URL", "0")  # Default to local webcam
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

original_width = original_frame.shape[1]
original_height = original_frame.shape[0]

# Use the same target dimensions as the main application
target_width = FRAME_WIDTH
scale_ratio = target_width / original_frame.shape[1]
display_frame = cv2.resize(original_frame, (target_width, int(original_frame.shape[0] * scale_ratio)))

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
        original_polygon_points = []
        for x, y in polygon_points:
            original_x = int(x / scale_ratio)
            original_y = int(y / scale_ratio)
            original_polygon_points.append((original_x, original_y))
        
        # Save to config.json
        save_path = "config.json"
        with open(save_path, "w") as f:
            json.dump(original_polygon_points, f)
        print("Display polygon points:", polygon_points)
        print(f"Original polygon points saved to {save_path}:", original_polygon_points)
        cv2.waitKey(500)
        break

cv2.destroyAllWindows()
