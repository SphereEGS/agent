#!/bin/bash

# Script to setup and run Spherex Agent with Docker Compose using predefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check if running as root (recommended for Docker)
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Warning: This script should be run with sudo for Docker access${NC}"
fi

# Predefined DockerHub credentials (replace with real values)
DOCKER_USERNAME="eglobalshpere" 
DOCKER_PASSWORD="dummy_pat" # This will be supplied later securely

# Predefined Spherex Agent environment variables (replace with real values)
GATE="Edara-GT-11"                # Gate ID on backend server (e.g., Edara-GT-11)
CAMERA_URL="rtsp://10.0.40.11:554"  # Camera URL (e.g., rtsp://10.0.40.11:554)
BACKEND_URL="http://backend.Spherex.com" # Backend server URL (e.g., http://backend.Spherex.com)
CONTROLLER_IP="192.168.2.49"  # Biostar Controller IP address (e.g., 192.168.2.49)
CONTROLLER_USER="admin"         # Biostar Controller Username
CONTROLLER_PASSWORD="Admin123"   # Biostar Controller Password
GATE_ID="21"                     # Gate ID on Biostar
FONT_PATH="fonts/DejaVuSans.ttf"   # default
MODEL_PATH="models/license_yolo8s_1024.pt" #

# Login to DockerHub with PAT
echo -e "${GREEN}Logging into DockerHub...${NC}"
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
if [ $? -eq 0 ]; then
    echo -e "${GREEN}DockerHub login successful${NC}"
else
    echo -e "${RED}Error: DockerHub login failed. Check credentials.${NC}"
    exit 1
fi

# Generate docker-compose.yml with predefined variables
echo -e "${GREEN}Generating docker-compose.yml...${NC}"
cat > docker-compose.yml << EOL
name: spherex-agent

services:
  spherex-agent:
    image: ${DOCKER_USERNAME}/spherex-agent:latest
    container_name: spherex-agent
    environment:
      - GATE=${GATE}
      - CAMERA_URL=${CAMERA_URL}
      - BACKEND_URL=${BACKEND_URL}
      - CONTROLLER_IP=${CONTROLLER_IP}
      - CONTROLLER_USER=${CONTROLLER_USER}
      - CONTROLLER_PASSWORD=${CONTROLLER_PASSWORD}
      - GATE_ID=${GATE_ID}
      - FONT_PATH=${FONT_PATH}
      - MODEL_PATH=${MODEL_PATH}
    volumes:
      - ./config.json:/app/config.json
    restart: unless-stopped
    network_mode: host
EOL

# Run docker-compose
echo -e "${GREEN}Starting Spherex Agent with Docker Compose...${NC}"
docker-compose up -d
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Container started successfully${NC}"
else
    echo -e "${RED}Error: Failed to start container. Check Docker Compose logs.${NC}"
    docker-compose logs
    exit 1
fi

# Exec into container to run roi.py
echo -e "${GREEN}Executing into container to run roi.py...${NC}"
docker exec -it spherex-agent bash -c "python roi.py"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}ROI setup completed. config.json should be saved.${NC}"
else
    echo -e "${RED}Error: Failed to run roi.py. Check container logs.${NC}"
    docker-compose logs
    exit 1
fi

echo -e "${GREEN}Setup complete!${NC}"
echo "The Spherex Agent is running in the background."
echo "To view logs: docker-compose logs"
echo "To stop: docker-compose down"