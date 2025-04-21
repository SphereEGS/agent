#!/bin/bash
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set default values
CAMERA="all"
CAMERA_URL=""

# Function to display help
show_help() {
    echo -e "${CYAN}Usage: ./run.sh [OPTIONS]${NC}"
    echo
    echo "Options:"
    echo "  -c, --camera CAMERA_ID    Specify camera ID to run (e.g., '1', '2', 'main')"
    echo "  -u, --url CAMERA_URL      Directly specify camera URL"
    echo "  -h, --help                Show this help message"
    echo
    echo "Examples:"
    echo "  ./run.sh -c 1 -c 2"
    echo "  ./run.sh -u rtsp://example.com/camera1"
    echo
}

# Function to display current camera configuration
show_config() {
    echo -e "${YELLOW}Current camera configuration:${NC}"
    
    # Show all camera URLs
    grep "CAMERA_URL" .env | grep -v "^#" || echo "No camera URLs configured"
    
    # Show enabled camera sources
    echo -e "\n${YELLOW}Enabled camera sources:${NC}"
    grep "ENABLE_" .env | grep -v "^#" || echo "No camera sources explicitly enabled"
    
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--camera)
      CAMERA="$2"
      shift 2
      ;;
    -u|--url)
      CAMERA_URL="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Display banner
echo -e "${CYAN}=================================="
echo -e "  SpherexAgent"
echo -e "==================================${NC}"

# Display current camera settings
show_config

# Enable virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${RED}Error: Virtual environment not found.${NC}"
    exit 1
fi

# Set the camera URL as environment variable if provided
if [ -n "$CAMERA_URL" ]; then
  export CAMERA_URL_$CAMERA="$CAMERA_URL"
  echo -e "${GREEN}Set CAMERA_URL_$CAMERA=$CAMERA_URL${NC}"
fi

# If specific camera is requested, pass it as an environment variable
if [ "$CAMERA" != "all" ]; then
  export RUN_CAMERA_ID="$CAMERA"
  echo -e "${GREEN}Running camera: $CAMERA${NC}"
else
  echo -e "${GREEN}Running all configured cameras${NC}"
fi

# Clear Python cache
echo -e "${CYAN}Clearing Python cache...${NC}"
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# Verify models exist
if [ ! -f "models/yolo11n.pt" ] || [ ! -f "models/license_yolo8s_1024.pt" ]; then
    echo -e "${RED}Error: Required model files are missing.${NC}"
    echo "Please check the models directory. Required files:"
    echo "- models/yolo11n.pt"
    echo "- models/license_yolo8s_1024.pt"
    echo
    echo -e "${YELLOW}Found these files:${NC}"
    ls -la models/
    exit 1
fi

# Check model symlinks are set up correctly
if [ ! -f "models/license_plate_recognition.pt" ]; then
    echo -e "${YELLOW}Creating symlink for license_plate_recognition.pt...${NC}"
    ln -sf license_yolo8s_1024.pt models/license_plate_recognition.pt
fi

# Run the application
echo -e "${GREEN}Starting SpherexAgent with configured cameras...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
python main.py
