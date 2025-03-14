import cv2
import json
import argparse
from dotenv import load_dotenv
import os

def setup_roi(camera_url, output_file):
    load_dotenv()
    
    if not camera_url:
        raise ValueError("Camera URL not provided")

    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print(f"Error: Unable to open RTSP stream: {camera_url}")
        exit(1)

    ret, original_frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame from the stream.")
        exit(1)

    original_width = original_frame.shape[1]
    original_height = original_frame.shape[0]

    target_width = 800
    scale_ratio = target_width / original_frame.shape[1]
    display_frame = cv2.resize(original_frame, (target_width, int(original_frame.shape[0] * scale_ratio)))

    polygon_points = []
    polygon_finished = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal polygon_points, polygon_finished
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(polygon_points) >= 3:
                polygon_finished = True
            else:
                print("Need at least 3 points to form a polygon.")

    window_name = f"Draw ROI for {output_file} - Left Click: add point, Right Click: finish, Esc: exit"
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
            
            with open(output_file, "w") as f:
                json.dump(original_polygon_points, f)
            print("Display polygon points:", polygon_points)
            print(f"Original polygon points saved to {output_file}:", original_polygon_points)
            cv2.waitKey(500)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up ROI for camera')
    parser.add_argument('--type', type=str, choices=['entry', 'exit'], required=True, help='Gate type (entry or exit)')
    args = parser.parse_args()
    
    load_dotenv()
    
    if args.type == 'entry':
        camera_url = os.getenv("ENTRY_CAMERA_URL") or os.getenv("CAMERA_URL")
        output_file = "config_entry.json"
    else:  # exit
        camera_url = os.getenv("EXIT_CAMERA_URL")
        output_file = "config_exit.json"

    setup_roi(camera_url, output_file)
