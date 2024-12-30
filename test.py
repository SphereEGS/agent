import cv2
import time
from datetime import datetime
import os
from dotenv import load_dotenv
from threading import Thread
import queue

# Load environment variables
load_dotenv()
CAMERA_URL = os.getenv("CAMERA_URL")

class CameraStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        
        # Set camera properties for better performance
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG format
        
        self.queue = queue.Queue(maxsize=2)  # Limit queue size
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        while True:
            if self.stopped:
                return
            
            # Clear queue if full
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                    
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                return
                
            self.queue.put(frame)
            
    def read(self):
        return self.queue.get()
        
    def stop(self):
        self.stopped = True
        
    def release(self):
        self.stopped = True
        self.stream.release()

def capture_frames():
    # Create output directory
    os.makedirs("debug_frames", exist_ok=True)
    
    print(f"Connecting to camera: {CAMERA_URL}")
    
    last_save_time = 0  # Track last save time
    save_interval = 1.0  # Save interval in seconds
    
    try:
        # Initialize threaded camera stream
        camera = CameraStream(CAMERA_URL)
        camera.start()
        
        print("Camera connected successfully")
        print("Press 'q' to quit")
        
        while True:
            frame = camera.read()
            
            if frame is None:
                print("Error: Could not read frame")
                break
                
            # Display frame immediately
            cv2.imshow('Frame', frame)
            
            # Save frame at specified interval
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"debug_frames/frame_{timestamp}.jpg"
                
                # Save in a separate thread to avoid blocking
                save_thread = Thread(target=cv2.imwrite, args=(filename, frame))
                save_thread.start()
                
                print(f"Saving: {filename}")
                last_save_time = current_time
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping capture...")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if 'camera' in locals():
            camera.stop()
            camera.release()
        cv2.destroyAllWindows()
        print("Camera released")

if __name__ == "__main__":
    capture_frames()