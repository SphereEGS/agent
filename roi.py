import cv2
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Get video source from environment variables or use default
INPUT_SOURCE = os.getenv("INPUT_SOURCE", "camera")  # Changed default to camera
CAMERA_URL = os.getenv("CAMERA_URL", "0")  # Default to local webcam
VIDEO_PATH = os.getenv("VIDEO_PATH", "input/test_video3.mov")

# Use appropriate source based on INPUT_SOURCE
if INPUT_SOURCE == "video":
    source_path = VIDEO_PATH
    print(f"Using video file: {source_path}")
    if not os.path.exists(source_path):
        print(f"Error: Video file {source_path} not found.")
        exit(1)
else:
    # Try CCTV first, fallback to local webcam if connection fails
    source_path = CAMERA_URL
    print(f"Attempting to connect to camera stream: {source_path}")
    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        print("CCTV connection failed, falling back to local webcam (index 0)")
        source_path = 0
        cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        print("Error: Unable to open any camera source")
        exit(1)

    print(f"Successfully connected to: {'CCTV' if source_path != 0 else 'Local Webcam'}")

# Remove the duplicate VideoCapture since we already have it from above
if INPUT_SOURCE == "video":
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Error: Unable to open source: {source_path}")
        exit(1)

# Read a frame for ROI selection
ret, original_frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read frame from the source.")
    exit(1)

original_width = original_frame.shape[1]
original_height = original_frame.shape[0]

target_width = 800
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
        
        with open("config.json", "w") as f:
            json.dump(original_polygon_points, f)
        print("Display polygon points:", polygon_points)
        print("Original polygon points saved to config.json:", original_polygon_points)
        cv2.waitKey(500)
        break

cv2.destroyAllWindows()
