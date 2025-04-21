import cv2
import json
import os
import sys
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Get dimensions from .env or use defaults as fallback
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))

def get_camera_sources():
    """Get available camera sources from environment variables"""
    cameras = {}
    
    # Look for numbered camera URLs
    for key, value in os.environ.items():
        if key.startswith("CAMERA_URL_"):
            camera_id = key[len("CAMERA_URL_"):].lower()
            # Clean any quotes or whitespace
            camera_url = value.strip()
            if camera_url.startswith('"') and camera_url.endswith('"'):
                camera_url = camera_url[1:-1].strip()
            elif camera_url.startswith("'") and camera_url.endswith("'"):
                camera_url = camera_url[1:-1].strip()
            
            cameras[camera_id] = camera_url
    
    # Add default camera if available
    if "CAMERA_URL" in os.environ:
        camera_url = os.environ["CAMERA_URL"].strip()
        if camera_url.startswith('"') and camera_url.endswith('"'):
            camera_url = camera_url[1:-1].strip()
        cameras["main"] = camera_url
    
    # Add webcam as fallback
    cameras["webcam"] = "0"
    
    return cameras

def connect_to_camera(source):
    """Connect to camera source and return capture object"""
    print(f"Connecting to: {source}")
    
    # Set RTSP transport to TCP for better reliability
    if isinstance(source, str) and source.startswith('rtsp://'):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    elif source == "0" or (isinstance(source, str) and source.isdigit()):
        # Handle webcam index
        cap = cv2.VideoCapture(int(source))
    else:
        # File or other URL
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Failed to connect to {source}")
        return None
    
    return cap

def draw_roi(frame, roi_type="LPR"):
    """Interactive ROI drawing function"""
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
    
    # Create window and set callback
    window_name = f"Draw {roi_type} ROI - Left Click: add point, Right Click: finish, Esc: cancel"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Instructions
    print("\nInstructions:")
    print("  - Left-click to add polygon points")
    print("  - Right-click to complete the polygon (minimum 3 points)")
    print("  - Press 'Esc' to cancel and exit")
    
    # Main drawing loop
    while True:
        display_frame = frame.copy()
        
        # Draw the polygon points and lines
        if polygon_points:
            for pt in polygon_points:
                cv2.circle(display_frame, pt, 4, (0, 0, 255), -1)
            for i in range(1, len(polygon_points)):
                cv2.line(display_frame, polygon_points[i - 1], polygon_points[i], (0, 255, 0), 2)
            if polygon_finished:
                cv2.line(display_frame, polygon_points[-1], polygon_points[0], (0, 255, 0), 2)
                
                # Fill with semi-transparent color
                overlay = display_frame.copy()
                fill_color = (0, 255, 0) if roi_type == "LPR" else (0, 165, 255)
                pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], fill_color)
                cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
        
        # Add guide text
        padding = 20
        cv2.putText(display_frame, f"Drawing {roi_type} ROI", 
                   (padding, padding+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(display_frame, f"Drawing {roi_type} ROI", 
                   (padding, padding+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        desc = "License Plate Recognition Zone (smaller)" if roi_type == "LPR" else "Motion Trigger Zone (larger)"
        cv2.putText(display_frame, desc, 
                   (padding, padding+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(display_frame, desc, 
                   (padding, padding+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        points_text = f"Points: {len(polygon_points)}"
        cv2.putText(display_frame, points_text, 
                   (padding, padding+65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(display_frame, points_text, 
                   (padding, padding+65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Check for ESC key
        if key == 27:
            cv2.destroyWindow(window_name)
            return None
        
        # Check if polygon is finished
        if polygon_finished:
            cv2.destroyWindow(window_name)
            return polygon_points

def main():
    """Main function to handle camera selection and ROI setup"""
    # Print using dimensions from .env
    print(f"Using frame dimensions from .env: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    
    # Get camera sources
    cameras = get_camera_sources()
    
    # Exit if no cameras found
    if not cameras:
        print("No camera sources found in environment variables.")
        return
    
    # List available cameras
    print("\nAvailable Camera Sources:")
    camera_ids = list(cameras.keys())
    for i, camera_id in enumerate(camera_ids):
        print(f"{i+1}. {camera_id}: {cameras[camera_id]}")
    
    # Get user choice
    while True:
        try:
            choice = input("\nSelect camera number (or press Enter for first option): ")
            if choice.strip() == "":
                choice = 1
            else:
                choice = int(choice)
            
            if 1 <= choice <= len(camera_ids):
                selected_id = camera_ids[choice-1]
                selected_url = cameras[selected_id]
                break
            else:
                print(f"Please enter a number between 1 and {len(camera_ids)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nSelected camera: {selected_id} ({selected_url})")
    
    # Connect to selected camera
    cap = connect_to_camera(selected_url)
    if not cap:
        print("Could not connect to selected camera.")
        return
    
    # Capture a frame for ROI drawing
    ret, original_frame = cap.read()
    cap.release()
    
    if not ret or original_frame is None:
        print("Could not capture frame from camera.")
        return
    
    # Get original frame dimensions
    original_width = original_frame.shape[1]
    original_height = original_frame.shape[0]
    print(f"Original frame dimensions: {original_width}x{original_height}")
    
    # Use exact dimensions from .env
    target_width = FRAME_WIDTH
    target_height = FRAME_HEIGHT
    print(f"Target dimensions from .env: {target_width}x{target_height}")
    
    # Preserve aspect ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > (target_width / target_height):
        # Image is wider than target
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    # Ensure dimensions are even
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)
    print(f"Display dimensions (preserving aspect ratio): {new_width}x{new_height}")
    
    # Calculate scale ratios
    scale_width_ratio = new_width / original_width
    scale_height_ratio = new_height / original_height
    
    # Resize frame for display
    display_frame = cv2.resize(original_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # First define Trigger ROI (the larger zone)
    print("\n--- STEP 1: Create Trigger ROI ---")
    print("Define the larger zone that detects approaching vehicles")
    trigger_polygon = draw_roi(display_frame, roi_type="Trigger")
    
    # Define LPR ROI (the smaller zone)
    print("\n--- STEP 2: Create LPR ROI ---")
    print("Define the smaller zone where license plate recognition happens")
    lpr_polygon = draw_roi(display_frame, roi_type="LPR")
    
    # Convert display coordinates to original coordinates
    def convert_to_original(points):
        if not points:
            return None
        
        original_points = []
        for x, y in points:
            # Convert back to original frame coordinates
            original_x = int(x / scale_width_ratio)
            original_y = int(y / scale_height_ratio)
            original_points.append((original_x, original_y))
        
        return original_points
    
    original_trigger_points = convert_to_original(trigger_polygon)
    original_lpr_points = convert_to_original(lpr_polygon)
    
    # Save configuration files
    def save_config(points, display_points, filename):
        if not points:
            print(f"Skipping {filename} - no points defined")
            return False
        
        config_data = {
            "original_points": points,
            "display_points": display_points,
            "original_dimensions": [original_width, original_height],
            "display_dimensions": [new_width, new_height],
            "scale_ratios": [scale_width_ratio, scale_height_ratio]
        }
        
        # Create camera-specific filename if not main camera
        if selected_id != "main":
            name_parts = filename.split(".")
            filename = f"{selected_id}_{name_parts[0]}.{name_parts[1]}"
        
        with open(filename, "w") as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Saved {filename} successfully")
        return True
    
    # Save ROI configurations
    trigger_saved = save_config(original_trigger_points, trigger_polygon, "trigger_roi.json")
    lpr_saved = save_config(original_lpr_points, lpr_polygon, "config.json")
    
    # Display final results
    if trigger_saved and lpr_saved:
        print("\nBoth ROIs have been saved successfully!")
        
        # Create visualization with both ROIs for confirmation
        visual_frame = display_frame.copy()
        
        # Draw Trigger ROI
        if trigger_polygon:
            pts = np.array(trigger_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(visual_frame, [pts], True, (0, 165, 255), 2)
            overlay = visual_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 165, 255))
            cv2.addWeighted(overlay, 0.2, visual_frame, 0.8, 0, visual_frame)
            
            # Add label
            x, y = trigger_polygon[0]
            cv2.putText(visual_frame, "Trigger Zone", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(visual_frame, "Trigger Zone", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 1)
        
        # Draw LPR ROI
        if lpr_polygon:
            pts = np.array(lpr_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(visual_frame, [pts], True, (0, 255, 0), 3)
            overlay = visual_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, visual_frame, 0.7, 0, visual_frame)
            
            # Add label
            x, y = lpr_polygon[0]
            cv2.putText(visual_frame, "LPR Zone", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(visual_frame, "LPR Zone", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        # Add confirmation message
        cv2.putText(visual_frame, "ROIs Configured Successfully", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(visual_frame, "ROIs Configured Successfully", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        # Show camera ID on the preview
        cv2.putText(visual_frame, f"Camera: {selected_id}", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(visual_frame, f"Camera: {selected_id}", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Show the configuration and wait for key press
        cv2.imshow("ROI Configuration Complete", visual_frame)
        print("Press any key to close the preview")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nROI setup incomplete. Please try again.")

if __name__ == "__main__":
    main() 