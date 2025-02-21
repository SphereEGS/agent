import cv2

def capture_rtsp_screenshot(rtsp_url, output_filename="screenshot.png"):
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Error: Unable to open the RTSP stream.")
        return

    # Read one frame from the stream
    ret, frame = cap.read()
    if ret:
        # Save the captured frame to a file
        cv2.imwrite(output_filename, frame)
        print(f"Screenshot saved as '{output_filename}'.")
    else:
        print("Error: Unable to capture a frame from the RTSP stream.")
    
    # Release the capture object
    cap.release()

if __name__ == "__main__":
    # Replace with your actual RTSP URL
    rtsp_url = "rtsp://test:123456789A@10.0.40.11:554/Streaming/channels/101"
    capture_rtsp_screenshot(rtsp_url)

