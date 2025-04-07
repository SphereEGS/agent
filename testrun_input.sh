#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====== Spherex Input Source Selection Script ======${NC}"

# Check Python environment
if [ -d "venv" ]; then
    echo -e "${GREEN}Using Python virtual environment...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${GREEN}Using Python virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found, using system Python...${NC}"
fi

# Select input source
echo -e "${YELLOW}Select camera source:${NC}"
echo -e "1) ${GREEN}CCTV camera (RTSP) [default]${NC}"
echo -e "2) ${GREEN}Webcam (local camera)${NC}"
echo -e "3) ${GREEN}Test video file${NC}"
read -p "Enter choice [1-3]: " choice

# Set the appropriate camera source based on user selection
case "$choice" in
    2)
        echo -e "${GREEN}Using webcam as camera source${NC}"
        CAMERA_URL="0"
        ;;
    3)
        echo -e "${GREEN}Using test video file as camera source${NC}"
        CAMERA_URL="input/test_video3.mov"
        ;;
    *)
        echo -e "${GREEN}Using CCTV camera (RTSP) as camera source${NC}"
        CAMERA_URL="rtsp://192.168.0.3:8554/stream"
        ;;
esac

# Default to no fallback
ALLOW_FALLBACK_VALUE="false"
echo -e "${GREEN}Webcam fallback is disabled${NC}"

# Create .env file with selected camera source
echo -e "${YELLOW}Creating .env file with camera configuration...${NC}"
cat > .env << EOL
############################################
#        SPHEREX AGENT CONFIGURATION        #
############################################

# Zone identification
ZONE=Edara

# Camera source configuration
CAMERA_URL=$CAMERA_URL

# Camera fallback behavior
ALLOW_FALLBACK=$ALLOW_FALLBACK_VALUE

# Camera settings
FRAME_WIDTH=1280
FRAME_HEIGHT=720

# Processing - higher number for better performance
PROCESS_EVERY=5

# API endpoint
API_BASE_URL=https://dev-backend.spherex.eglobalsphere.com/api

# Model paths
FONT_PATH=fonts/NotoSansArabic-Regular.ttf
LPR_MODEL_PATH=models/license_yolo8s_1024.pt
YOLO_MODEL_PATH=models/yolov8n.pt

# Gate controller
CONTROLLER_IP=127.0.0.1
CONTROLLER_PORT=4000
CONTROLLER_CA_PATH=controller/cert/ca.crt
GATE_ID=1

# Detection parameters
DETECTION_CONF=0.3
DETECTION_IOU=0.45
TARGET_WIDTH=1280
EOL

echo -e "${GREEN}Configuration updated${NC}"
echo -e "${YELLOW}Starting application...${NC}"
echo -e "${YELLOW}Press 'q' to exit the application.${NC}"

# Run the application using main.py
python main.py

echo -e "${GREEN}Test completed.${NC}"
