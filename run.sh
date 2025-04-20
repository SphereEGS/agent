#!/bin/bash
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display help
show_help() {
    echo -e "${CYAN}Usage: ./run.sh [OPTIONS]${NC}"
    echo
    echo "Options:"
    echo "  --help        Show this help message"
    echo "  --config      Show current camera configuration"
    echo "  --enable=CAM  Enable a specific camera (e.g., --enable=webcam)"
    echo "  --disable=CAM Disable a specific camera"
    echo
    echo "Examples:"
    echo "  ./run.sh --enable=webcam --enable=cctv"
    echo "  ./run.sh --disable=webcam"
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

# Process command line arguments
for arg in "$@"; do
    case $arg in
        --help)
            show_help
            exit 0
            ;;
        --config)
            show_config
            exit 0
            ;;
        --enable=*)
            camera="${arg#*=}"
            camera_upper=$(echo "$camera" | tr '[:lower:]' '[:upper:]')
            
            # Check if ENABLE_ entry exists already
            if grep -q "^ENABLE_$camera_upper=" .env; then
                # Update existing entry
                sed -i'' -e "s/^ENABLE_$camera_upper=.*/ENABLE_$camera_upper=true/" .env
            else
                # Add new entry
                echo "ENABLE_$camera_upper=true" >> .env
            fi
            
            echo -e "${GREEN}Enabled camera: $camera${NC}"
            ;;
        --disable=*)
            camera="${arg#*=}"
            camera_upper=$(echo "$camera" | tr '[:lower:]' '[:upper:]')
            
            # Check if ENABLE_ entry exists already
            if grep -q "^ENABLE_$camera_upper=" .env; then
                # Update existing entry
                sed -i'' -e "s/^ENABLE_$camera_upper=.*/ENABLE_$camera_upper=false/" .env
            else
                # Add new entry
                echo "ENABLE_$camera_upper=false" >> .env
            fi
            
            echo -e "${YELLOW}Disabled camera: $camera${NC}"
            ;;
    esac
done

# Display banner
echo -e "${CYAN}=================================="
echo -e "  SpherexAgent"
echo -e "==================================${NC}"

# Display current camera settings
show_config

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
