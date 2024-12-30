import cv2
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CAMERA_URL = os.getenv("CAMERA_URL")

def capture_frames():
    # Create output directory
    os.makedirs("debug_frames", exist_ok=True)
    
    print(f"Connecting to camera: {CAMERA_URL}")
    
    while True:
        try:
            # Initialize camera
            cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                print("Error: Could not open camera stream")
                time.sleep(5)  # Wait before retrying
                continue
                
            print("Camera connected successfully")
            print("Capturing frames every second. Press Ctrl+C to stop.")
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
            
            while True:
                # Flush the buffer by reading multiple frames
                for _ in range(3):
                    cap.grab()
                
                # Capture frame
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    print("Error: Could not read frame")
                    break  # Break inner loop to reinitialize camera
                    
                # Generate timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save frame
                filename = f"debug_frames/frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                
                # Optional: Display frame
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
                
                # Wait for 1 second
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping capture...")
            break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            time.sleep(5)  # Wait before retrying
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            
    print("Camera released")

if __name__ == "__main__":
    capture_frames()