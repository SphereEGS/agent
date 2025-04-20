import cv2
import json
from dotenv import load_dotenv
import os
import sys
import argparse
import numpy as np
from app.config import FRAME_WIDTH, FRAME_HEIGHT, logger

def save_original_frame(original_frame, camera_id):
    """Save original frame for reference"""
    os.makedirs("debug", exist_ok=True)
    filename = f"debug/original_frame_{camera_id}.jpg"
    cv2.imwrite(filename, original_frame)
    print(f"Saved original frame to {filename}")

def save_resized_frame(resized_frame, camera_id):
    """Save resized frame for reference"""
    os.makedirs("debug", exist_ok=True)
    filename = f"debug/resized_frame_{camera_id}.jpg"
    cv2.imwrite(filename, resized_frame)
    print(f"Saved resized frame to {filename}")

def setup_roi_tool():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ROI Configuration Tool')
    parser.add_argument('--camera', type=str, default="main", help='Camera ID (default: main)')
    parser.add_argument('--roi-type', type=str, default="lpr", choices=['lpr', 'detection'], 
                        help='ROI type to configure (lpr or detection)')
    parser.add_argument('--source', type=str, help='Override camera source (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with frame saving')
    parser.add_argument('--display-width', type=int, default=1280, help='Display window width')
    parser.add_argument('--display-height', type=int, default=960, help='Display window height')
    parser.add_argument('--preserve-full-frame', action='store_true', help='Show full frame without cropping')
    args = parser.parse_args()
    
    load_dotenv()
    
    debug_mode = args.debug
    display_width = args.display_width
    display_height = args.display_height
    preserve_full_frame = args.preserve_full_frame
    
    # Get video source from environment variables or use default
    if args.source:
        CAMERA_URL = args.source
    else:
        # Look for camera-specific URL first
        camera_env_var = f"CAMERA_URL_{args.camera.upper()}" if args.camera.lower() != "main" else "CAMERA_URL"
        CAMERA_URL = os.getenv(camera_env_var, os.getenv("CAMERA_URL", "0"))  # Default to local webcam
    
    camera_id = args.camera
    roi_type = args.roi_type
    
    # Config file based on camera ID
    config_file = f"config_{camera_id}.json" if camera_id != "main" else "config.json"

    # Clean any quotes from CAMERA_URL
    if isinstance(CAMERA_URL, str):
        CAMERA_URL = CAMERA_URL.strip()
        if CAMERA_URL.startswith('"') and CAMERA_URL.endswith('"'):
            CAMERA_URL = CAMERA_URL[1:-1].strip()
        if '#' in CAMERA_URL:
            CAMERA_URL = CAMERA_URL.split('#')[0].strip()

    print(f"Camera URL: {CAMERA_URL}")
    print(f"Camera ID: {camera_id}")
    print(f"ROI Type: {roi_type}")
    print(f"Config file: {config_file}")

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

    # Save original frame if debug mode is enabled
    if debug_mode:
        save_original_frame(original_frame, camera_id)

    # Calculate the target dimensions
    if preserve_full_frame:
        # Option 1: Resize to fit display window while preserving aspect ratio
        aspect_ratio = original_width / original_height
        if aspect_ratio > (display_width / display_height):
            # Image is wider than display
            new_width = display_width
            new_height = int(display_width / aspect_ratio)
        else:
            # Image is taller than display
            new_height = display_height
            new_width = int(display_height * aspect_ratio)
            
        # Ensure dimensions are even numbers
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        
        print(f"Display dimensions (preserving full frame): {new_width}x{new_height}")
    else:
        # Option 2: Use dimensions from .env for consistency with runtime processing
        target_width = FRAME_WIDTH
        target_height = FRAME_HEIGHT

        # Calculate aspect ratio preserving dimensions as in camera.py
        aspect_ratio = original_width / original_height
        if aspect_ratio > (target_width / target_height):
            # Image is wider than target
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Image is taller than target
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # Ensure dimensions are even numbers
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        
        print(f"Target processing dimensions: {new_width}x{new_height}")

    print(f"Aspect ratio: {aspect_ratio:.3f}")

    # Calculate scale ratios for converting between original and resized coordinates
    scale_width_ratio = new_width / original_width
    scale_height_ratio = new_height / original_height
    print(f"Scale factors: width={scale_width_ratio:.3f}, height={scale_height_ratio:.3f}")

    # Resize the frame
    display_frame = cv2.resize(original_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Save resized frame if debug mode is enabled
    if debug_mode:
        save_resized_frame(display_frame, camera_id)

    # Also include the processing target dimensions for reference
    processing_width = FRAME_WIDTH
    processing_height = FRAME_HEIGHT
    print(f"Runtime processing dimensions from .env: {processing_width}x{processing_height}")

    # Load existing ROIs to display as reference
    existing_roi_points = []
    other_roi_points = []
    try:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config_data = json.load(f)
                
            # Load the existing ROI we're editing
            if roi_type == "lpr" and "lpr_roi" in config_data:
                existing_roi_points = config_data["lpr_roi"]["original_points"]
                # Convert original coordinates to display coordinates
                for i, (x, y) in enumerate(existing_roi_points):
                    display_x = int(x * scale_width_ratio)
                    display_y = int(y * scale_height_ratio)
                    existing_roi_points[i] = (display_x, display_y)
                print(f"Loaded existing LPR ROI with {len(existing_roi_points)} points")
            elif roi_type == "detection" and "detection_roi" in config_data:
                existing_roi_points = config_data["detection_roi"]["original_points"]
                # Convert original coordinates to display coordinates
                for i, (x, y) in enumerate(existing_roi_points):
                    display_x = int(x * scale_width_ratio)
                    display_y = int(y * scale_height_ratio)
                    existing_roi_points[i] = (display_x, display_y)
                print(f"Loaded existing Detection ROI with {len(existing_roi_points)} points")
                    
            # Load the other ROI type for reference
            other_type = "detection" if roi_type == "lpr" else "lpr"
            if other_type == "lpr" and "lpr_roi" in config_data:
                other_roi_points = config_data["lpr_roi"]["original_points"]
                # Convert original coordinates to display coordinates
                for i, (x, y) in enumerate(other_roi_points):
                    display_x = int(x * scale_width_ratio)
                    display_y = int(y * scale_height_ratio)
                    other_roi_points[i] = (display_x, display_y)
                print(f"Loaded other (LPR) ROI with {len(other_roi_points)} points for reference")
            elif other_type == "detection" and "detection_roi" in config_data:
                other_roi_points = config_data["detection_roi"]["original_points"]
                # Convert original coordinates to display coordinates
                for i, (x, y) in enumerate(other_roi_points):
                    display_x = int(x * scale_width_ratio)
                    display_y = int(y * scale_height_ratio)
                    other_roi_points[i] = (display_x, display_y)
                print(f"Loaded other (Detection) ROI with {len(other_roi_points)} points for reference")
                
    except Exception as e:
        print(f"Warning: Could not load existing ROI: {e}")

    polygon_points = existing_roi_points.copy() if existing_roi_points else []
    polygon_finished = bool(polygon_points)

    def mouse_callback(event, x, y, flags, param):
        nonlocal polygon_points, polygon_finished
        if event == cv2.EVENT_LBUTTONDOWN:
            if polygon_finished:
                # Clear existing points if we're starting over
                polygon_points = []
                polygon_finished = False
            polygon_points.append((x, y))
            print(f"Added point: ({x}, {y}) - display coordinates")
            # Convert to original image coordinates for reference
            orig_x = int(x / scale_width_ratio)
            orig_y = int(y / scale_height_ratio)
            print(f"  -> Original coordinates: ({orig_x}, {orig_y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(polygon_points) >= 3:
                polygon_finished = True
                print(f"Finished polygon with {len(polygon_points)} points")
            else:
                print("Need at least 3 points to form a polygon.")

    roi_color = (0, 0, 255) if roi_type == "lpr" else (255, 0, 0)
    other_roi_color = (255, 0, 0) if roi_type == "lpr" else (0, 0, 255)
    
    # Create resizable window
    window_name = f"Draw {roi_type.upper()} ROI for {camera_id} - Left Click: add point, Right Click: finish, Esc: exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set initial window size
    cv2.resizeWindow(window_name, new_width, new_height)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Instructions:")
    print("  - Left-click to add polygon points.")
    print(f"  - If modifying existing {roi_type.upper()} ROI, left-click to start over.")
    print("  - Right-click to complete the polygon (minimum 3 points).")
    print("  - Press 'Esc' to exit without saving.")
    print("  - Press 'S' to save the ROI.")
    print(f"Original frame: {original_width}x{original_height}, Display frame: {new_width}x{new_height}")

    while True:
        frame_display = display_frame.copy()
        
        # Draw the other ROI type if available (for reference)
        if other_roi_points:
            # Create transparent overlay for filling
            overlay = frame_display.copy()
            other_polygon = np.array(other_roi_points)
            other_fill_color = (192, 0, 0) if roi_type == "lpr" else (0, 0, 192)  # Opposite colors
            cv2.fillPoly(overlay, [other_polygon], other_fill_color)
            # Apply transparency
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame_display, 1 - alpha, 0, frame_display)
            
            # Draw outline
            other_polygon = [np.array(other_roi_points)]
            cv2.polylines(frame_display, other_polygon, True, other_roi_color, 1)
            # Label the ROI
            other_type_name = "LPR Zone" if roi_type == "detection" else "Detection Zone"
            cv2.putText(frame_display, other_type_name,
                       (other_roi_points[0][0], other_roi_points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, other_roi_color, 1)

        # Draw our current ROI
        if polygon_points:
            # Create transparent overlay for current ROI
            if len(polygon_points) >= 3:
                overlay = frame_display.copy()
                current_polygon = np.array(polygon_points)
                current_fill_color = (0, 0, 192) if roi_type == "lpr" else (192, 0, 0)
                cv2.fillPoly(overlay, [current_polygon], current_fill_color)
                # Apply transparency
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, frame_display, 1 - alpha, 0, frame_display)
            
            # Draw points and lines
            for pt in polygon_points:
                cv2.circle(frame_display, pt, 4, roi_color, -1)
            for i in range(1, len(polygon_points)):
                cv2.line(frame_display, polygon_points[i - 1], polygon_points[i], roi_color, 2)
            if polygon_finished:
                cv2.line(frame_display, polygon_points[-1], polygon_points[0], roi_color, 2)
                
            # Label our ROI
            roi_type_name = "LPR Zone" if roi_type == "lpr" else "Detection Zone"
            cv2.putText(frame_display, roi_type_name, 
                       (polygon_points[0][0], polygon_points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)

        # Add frame dimensions and info
        cv2.putText(frame_display, f"Camera: {camera_id} | {new_width}x{new_height} ({original_width}x{original_height})", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        # Add current settings from .env
        cv2.putText(frame_display, f"FRAME_WIDTH={FRAME_WIDTH}, FRAME_HEIGHT={FRAME_HEIGHT}", 
                   (10, new_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, frame_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            print("Exiting without saving ROI.")
            break
        elif key == ord('s') and polygon_finished:  # 'S' key
            # Convert display coordinates back to original coordinates
            original_polygon_points = []
            for x, y in polygon_points:
                original_x = int(x / scale_width_ratio)
                original_y = int(y / scale_height_ratio)
                original_polygon_points.append((original_x, original_y))
            
            # Load existing config or create new
            try:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                else:
                    config_data = {}
                
                # Update with new ROI
                if roi_type == "lpr":
                    config_data["lpr_roi"] = {
                        "original_points": original_polygon_points
                    }
                else:  # detection
                    config_data["detection_roi"] = {
                        "original_points": original_polygon_points
                    }
                
                # Save original dimensions
                config_data["original_dimensions"] = [original_width, original_height]
                config_data["target_dimensions"] = [new_width, new_height]
                config_data["scale_factors"] = {
                    "width": float(scale_width_ratio),
                    "height": float(scale_height_ratio)
                }
                
                # Save to file
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                print(f"Saved {roi_type} ROI to {config_file}")
                # Save the config file path and full parameters for reference
                print(f"Full configuration saved to: {os.path.abspath(config_file)}")
                print(f"Scale factors: width={scale_width_ratio:.3f}, height={scale_height_ratio:.3f}")
                break
                
            except Exception as e:
                print(f"Error saving ROI: {e}")
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    setup_roi_tool()