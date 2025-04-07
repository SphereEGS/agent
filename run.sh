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

# Display current camera settings
echo -e "${YELLOW}Current camera settings:${NC}"
cat .env | grep "CAMERA_URL" | grep -v "^#"

# Activate virtual environment
if [ -d "venv" ]; then
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${RED}Error: Virtual environment not found.${NC}"
    exit 1
fi

# Clear Python cache
echo -e "${CYAN}Clearing Python cache...${NC}"
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# Verify models exist
if [ ! -f "models/yolo11n.pt" ] || [ ! -f "models/license_yolo8s_1024.pt" ]; then
    echo -e "${RED}Error: Required model files are missing.${NC}"
    echo "Please check the models directory."
    exit 1
fi

# Run the application
echo -e "${GREEN}Starting SpherexAgent with CCTV camera...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
python main.py
