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

def create_grid_view(frames_dict, grid_width=1280, grid_height=720):
    """
    Create a grid view from multiple camera frames
    
    Args:
        frames_dict: Dictionary of {camera_id: frame}
        grid_width: Width of the grid
        grid_height: Height of the grid
    
    Returns:
        Combined grid frame
    """
    # Get the size of frames and count
    frame_count = len(frames_dict)
    if frame_count == 0:
        return None
    
    # Determine grid dimensions based on number of cameras
    if frame_count == 1:
        grid_cols, grid_rows = 1, 1
    elif frame_count <= 4:
        grid_cols, grid_rows = 2, 2
    elif frame_count <= 9:
        grid_cols, grid_rows = 3, 3
    else:
        grid_cols, grid_rows = 4, 3  # Max 12 cameras
    
    # Calculate target size for each grid cell
    cell_width = grid_width // grid_cols
    cell_height = grid_height // grid_rows
    
    # Create black canvas for the grid
    grid_view = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Sort camera IDs to maintain consistent positions
    camera_ids = sorted(frames_dict.keys())
    
    # Place frames in grid
    for i, camera_id in enumerate(camera_ids):
        if i >= grid_rows * grid_cols:
            break  # Don't exceed grid size
        
        frame = frames_dict[camera_id]
        
        # Make sure we're preserving aspect ratio properly when resizing
        current_h, current_w = frame.shape[:2]
        aspect_ratio = current_w / current_h
        
        # Calculate dimensions that preserve aspect ratio
        if aspect_ratio > (cell_width / cell_height):
            # Image is wider than target
            resize_w = cell_width
            resize_h = int(cell_width / aspect_ratio)
        else:
            # Image is taller than target
            resize_h = cell_height
            resize_w = int(cell_height * aspect_ratio)
            
        # Ensure dimensions are even numbers
        resize_w = resize_w - (resize_w % 2)
        resize_h = resize_h - (resize_h % 2)
        
        # Resize the frame preserving aspect ratio
        resized_frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        
        # Create a full-size cell with black background
        cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        
        # Calculate position to center the resized frame in the cell
        y_offset = (cell_height - resize_h) // 2
        x_offset = (cell_width - resize_w) // 2
        
        # Place the resized frame in the cell
        cell[y_offset:y_offset+resize_h, x_offset:x_offset+resize_w] = resized_frame
        
        # Add camera ID label
        cv2.putText(cell, f"Camera {camera_id}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        row = i // grid_cols
        col = i % grid_cols
        
        y_start = row * cell_height
        y_end = y_start + cell_height
        x_start = col * cell_width
        x_end = x_start + cell_width
        
        grid_view[y_start:y_end, x_start:x_end] = cell
    
    # Add grid title
    cv2.putText(grid_view, "ROI Configuration - Multi-Camera View", 
               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return grid_view

def detect_camera_urls_from_env():
    """Detect camera URLs from .env file"""
    camera_urls = {}
    
    # Look for numbered camera URLs
    for key, value in os.environ.items():
        if key.startswith("CAMERA_URL_"):
            camera_id = key[len("CAMERA_URL_"):].lower()
            camera_urls[camera_id] = value.strip('"\'')
    
    # Add default camera if available
    if "CAMERA_URL" in os.environ:
        camera_urls["main"] = os.environ["CAMERA_URL"].strip('"\'')
    
    # If no cameras found, use default webcam
    if not camera_urls:
        camera_urls["main"] = "0"
    
    return camera_urls

def setup_camera(camera_url, camera_id, preserve_full_frame=False, display_width=1280, display_height=720, debug_mode=False):
    """Setup a camera and return the frame and resize information"""
    print(f"Camera URL: {camera_url}")
    print(f"Camera ID: {camera_id}")
    
    # Clean any quotes from camera_url
    if isinstance(camera_url, str):
        camera_url = camera_url.strip()
        if camera_url.startswith('"') and camera_url.endswith('"'):
            camera_url = camera_url[1:-1].strip()
        if '#' in camera_url:
            camera_url = camera_url.split('#')[0].strip()

    # Use appropriate source based on camera_url
    if camera_url.lower().endswith(('.mp4', '.mov', '.avi')):
        source_path = camera_url
        print(f"Using video file: {source_path}")
        if not os.path.exists(source_path):
            print(f"Error: Video file {source_path} not found.")
            return None, None, None, None
    elif camera_url == "0" or camera_url.isdigit():
        # Convert string to integer if needed for webcam
        source_path = 0 if camera_url == "0" else int(camera_url)
        print(f"Using webcam with index: {source_path}")
    else:
        # Assume RTSP or other camera URL
        source_path = camera_url
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
        return None, None, None, None

    print(f"Successfully connected to: {'CCTV/File' if source_path != 0 else 'Local Webcam'}")

    # Read a frame for ROI selection
    ret, original_frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame from the source: {camera_url}")
        return None, None, None, None

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
    
    return original_frame, display_frame, scale_width_ratio, scale_height_ratio

def load_roi_config(config_file, scale_width_ratio, scale_height_ratio):
    """Load ROI configuration and return scaled display coordinates"""
    existing_roi_points = []
    other_roi_points = []
    
    try:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config_data = json.load(f)
                
            # Load the existing ROI we're editing
            if "lpr_roi" in config_data:
                existing_roi_points = config_data["lpr_roi"]["original_points"]
                # Convert original coordinates to display coordinates
                for i, (x, y) in enumerate(existing_roi_points):
                    display_x = int(x * scale_width_ratio)
                    display_y = int(y * scale_height_ratio)
                    existing_roi_points[i] = (display_x, display_y)
                print(f"Loaded existing LPR ROI with {len(existing_roi_points)} points")
            
            # Load the detection ROI for reference
            if "detection_roi" in config_data:
                other_roi_points = config_data["detection_roi"]["original_points"]
                # Convert original coordinates to display coordinates
                for i, (x, y) in enumerate(other_roi_points):
                    display_x = int(x * scale_width_ratio)
                    display_y = int(y * scale_height_ratio)
                    other_roi_points[i] = (display_x, display_y)
                print(f"Loaded detection ROI with {len(other_roi_points)} points for reference")
    except Exception as e:
        print(f"Warning: Could not load existing ROI: {e}")
    
    return existing_roi_points, other_roi_points

def setup_roi_tool():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ROI Configuration Tool')
    parser.add_argument('--camera', type=str, default="all", help='Camera ID (default: all)')
    parser.add_argument('--roi-type', type=str, default="lpr", choices=['lpr', 'detection'], 
                        help='ROI type to configure (lpr or detection)')
    parser.add_argument('--source', type=str, help='Override camera source (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with frame saving')
    parser.add_argument('--display-width', type=int, default=1280, help='Display window width')
    parser.add_argument('--display-height', type=int, default=960, help='Display window height')
    parser.add_argument('--preserve-full-frame', action='store_true', help='Show full frame without cropping')
    parser.add_argument('--grid', action='store_true', help='Display all cameras in a grid view')
    args = parser.parse_args()
    
    load_dotenv()
    
    debug_mode = args.debug
    display_width = args.display_width
    display_height = args.display_height
    preserve_full_frame = args.preserve_full_frame
    roi_type = args.roi_type
    grid_mode = args.grid
    
    # Determine which cameras to process
    if args.source:
        # Use a specific camera source
        camera_urls = {"custom": args.source}
    elif args.camera.lower() == "all" or grid_mode:
        # Use all cameras from .env
        camera_urls = detect_camera_urls_from_env()
        print(f"Found {len(camera_urls)} camera configurations from .env")
    else:
        # Use a specific camera ID
        camera_id = args.camera
        camera_env_var = f"CAMERA_URL_{camera_id.upper()}" if camera_id.lower() != "main" else "CAMERA_URL"
        camera_url = os.getenv(camera_env_var, os.getenv("CAMERA_URL", "0"))
        camera_urls = {camera_id: camera_url}
    
    if not camera_urls:
        print("Error: No cameras found.")
        sys.exit(1)
    
    # Display the cameras we're going to work with
    for camera_id, camera_url in camera_urls.items():
        print(f"Camera {camera_id}: {camera_url}")
    
    # Setup individual cameras and get frames
    camera_frames = {}
    camera_scales = {}
    
    for camera_id, camera_url in camera_urls.items():
        print(f"\nSetting up camera {camera_id}...")
        original_frame, display_frame, scale_width, scale_height = setup_camera(
            camera_url, camera_id, preserve_full_frame, 
            display_width // 2 if len(camera_urls) > 1 else display_width,
            display_height // 2 if len(camera_urls) > 1 else display_height, 
            debug_mode
        )
        
        if original_frame is not None and display_frame is not None:
            config_file = f"config_{camera_id}.json" if camera_id != "main" else "config.json"
            
            # Load existing ROIs for this camera
            lpr_points, detection_points = load_roi_config(config_file, scale_width, scale_height)
            
            # Draw ROIs on the frame
            if lpr_points and len(lpr_points) >= 3:
                # Create transparent overlay for LPR ROI
                overlay = display_frame.copy()
                lpr_polygon = np.array(lpr_points)
                lpr_fill_color = (0, 0, 192)  # Red for LPR
                cv2.fillPoly(overlay, [lpr_polygon], lpr_fill_color)
                
                # Apply transparency
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                
                # Draw outline
                cv2.polylines(display_frame, [lpr_polygon], True, (0, 0, 255), 2)
                cv2.putText(display_frame, "LPR Zone", 
                          (lpr_points[0][0], lpr_points[0][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if detection_points and len(detection_points) >= 3:
                # Create transparent overlay for Detection ROI
                overlay = display_frame.copy()
                detection_polygon = np.array(detection_points)
                detection_fill_color = (192, 0, 0)  # Blue for Detection
                cv2.fillPoly(overlay, [detection_polygon], detection_fill_color)
                
                # Apply transparency
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                
                # Draw outline
                cv2.polylines(display_frame, [detection_polygon], True, (255, 0, 0), 2)
                cv2.putText(display_frame, "Detection Zone", 
                          (detection_points[0][0], detection_points[0][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add dimensions information at the bottom
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, f"Camera: {camera_id} | {w}x{h}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save frame and scale info
            camera_frames[camera_id] = display_frame
            camera_scales[camera_id] = (scale_width, scale_height, original_frame.shape[1], original_frame.shape[0])
    
    # If grid mode or multiple cameras, show grid view
    if grid_mode or len(camera_frames) > 1:
        # Create grid view window
        window_name = f"ROI Configuration - Multiple Cameras - Press 'q' to exit, Press '1-9' to select camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        print("\nMultiple camera grid view. Press 1-9 to select a camera, or q to exit.")
        print("Available cameras:")
        for i, camera_id in enumerate(sorted(camera_frames.keys()), 1):
            print(f"  {i}: Camera {camera_id}")
        
        while True:
            # Create and show the grid
            grid_view = create_grid_view(camera_frames, display_width, display_height)
            cv2.imshow(window_name, grid_view)
            
            # Wait for key press
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                print("Exiting grid view.")
                break
            
            # Check if a number key was pressed to select a camera
            for i, camera_id in enumerate(sorted(camera_frames.keys()), 1):
                if key == ord(str(i)) and i <= 9:
                    print(f"Selected camera {camera_id} for ROI configuration")
                    cv2.destroyAllWindows()
                    
                    # Launch ROI configuration for the selected camera
                    cmd = f"python roi.py --camera {camera_id} --roi-type {roi_type} --preserve-full-frame"
                    if debug_mode:
                        cmd += " --debug"
                    print(f"Running: {cmd}")
                    os.system(cmd)
                    
                    # Return to grid view
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, display_width, display_height)
                    break
    
    # For single camera mode, configure ROI
    elif len(camera_frames) == 1:
        camera_id = next(iter(camera_frames.keys()))
        config_file = f"config_{camera_id}.json" if camera_id != "main" else "config.json"
        display_frame = camera_frames[camera_id]
        scale_width_ratio, scale_height_ratio, original_width, original_height = camera_scales[camera_id]
        
        # Set up polygon points for editing
        existing_roi_points, other_roi_points = load_roi_config(config_file, scale_width_ratio, scale_height_ratio)
        
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
        h, w = display_frame.shape[:2]
        cv2.resizeWindow(window_name, w, h)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("Instructions:")
        print("  - Left-click to add polygon points.")
        print(f"  - If modifying existing {roi_type.upper()} ROI, left-click to start over.")
        print("  - Right-click to complete the polygon (minimum 3 points).")
        print("  - Press 'Esc' to exit without saving.")
        print("  - Press 'S' to save the ROI.")
        print(f"Original frame: {original_width}x{original_height}, Display frame: {w}x{h}")
        
        while True:
            frame_display = display_frame.copy()
            
            # Draw the other ROI type if available (for reference)
            if other_roi_points and len(other_roi_points) >= 3:
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
            cv2.putText(frame_display, f"Camera: {camera_id} | {w}x{h} ({original_width}x{original_height})", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
            # Add current settings from .env
            cv2.putText(frame_display, f"FRAME_WIDTH={FRAME_WIDTH}, FRAME_HEIGHT={FRAME_HEIGHT}", 
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
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
                    config_data["target_dimensions"] = [w, h]
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
    
    else:
        print("Error: No frames to display.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    setup_roi_tool()