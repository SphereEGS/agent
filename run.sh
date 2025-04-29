#!/bin/bash
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display banner
echo -e "${CYAN}=================================="
echo -e "  SpherexAgent"
echo -e "==================================${NC}"

# Parse command-line arguments
SETUP_MODE=false
LIST_CAMERAS=false
SETUP_ROI=false

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup)
            SETUP_MODE=true
            shift
            ;;
        --list-cameras)
            LIST_CAMERAS=true
            shift
            ;;
        --setup-roi)
            SETUP_ROI=true
            shift
            ;;
        *)
            # Skip unknown options
            shift
            ;;
    esac
done

# Activate virtual environment
if [ -d ".venv" ]; then
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${CYAN}Checking for venv...${NC}"
    if [ -d "venv" ]; then
        echo -e "${CYAN}Activating virtual environment from venv...${NC}"
        source venv/bin/activate
    else
        echo -e "${RED}Error: Virtual environment not found.${NC}"
        exit 1
    fi
fi

# Clear Python cache
echo -e "${CYAN}Clearing Python cache...${NC}"
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# Handle setup mode if requested
if [ "$SETUP_MODE" = true ]; then
    echo -e "${CYAN}Running in setup mode${NC}"
    python setup_cameras.py
    exit 0
fi

# List cameras if requested
if [ "$LIST_CAMERAS" = true ]; then
    echo -e "${CYAN}Listing configured cameras${NC}"
    python setup_cameras.py --list
    exit 0
fi

# Setup ROI if requested
if [ "$SETUP_ROI" = true ]; then
    echo -e "${CYAN}Starting ROI configuration workflow${NC}"
    python setup_cameras.py --configure
    exit 0
fi

# Verify models exist
if [ ! -f "models/yolo11n.pt" ] || [ ! -f "models/license_yolo8s_1024.pt" ]; then
    echo -e "${RED}Error: Required model files are missing.${NC}"
    echo "Please check the models directory."
    exit 1
fi

# Display current camera settings
echo -e "${YELLOW}Current camera settings:${NC}"
env | grep "CAMERA_URL" | grep -v "^#"

# Create configs directory if it doesn't exist
mkdir -p configs

# Run the application
echo -e "${GREEN}Starting SpherexAgent with multiple cameras...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
python main.py
