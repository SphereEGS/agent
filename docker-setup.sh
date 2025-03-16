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
DOCKER_USERNAME="docker_user" 
DOCKER_PASSWORD="docker_pat" # This will be supplied later securely

# Predefined Spherex Agent environment variables (replace with real values)
GATE="Edara-GT-11"                # Gate ID on backend server (e.g., Edara-GT-11)
CAMERA_URL="rtsp://10.0.40.11:554"  # Camera URL (e.g., rtsp://10.0.40.11:554)
BACKEND_URL="https://backend.spherex.com/" # Backend server URL (e.g., http://backend.Spherex.com)
CONTROLLER_IP="192.168.2.49"  # Biostar Controller IP address (e.g., 192.168.2.49)
CONTROLLER_USER="admin"         # Biostar Controller Username
CONTROLLER_PASSWORD="Admin123"   # Biostar Controller Password
GATE_ID="21"                     # Gate ID on Biostar
FONT_PATH="fonts/DejaVuSans.ttf"   # default
MODEL_PATH="models/license_yolo8s_1024.pt" #


############################################
#  DOCKER SETUP
############################################

echo -e "${GREEN}Starting Docker setup on Debian...${NC}"

# Update package index
echo "Updating package index..."
apt-get update -y || {
    echo -e "${RED}Error: Failed to update package index${NC}"
    exit 1
}

# Install required packages for HTTPS repositories
echo "Installing prerequisite packages..."
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release || {
    echo -e "${RED}Error: Failed to install prerequisite packages${NC}"
    exit 1
}

# Add Docker's official GPG key
echo "Adding Docker GPG key..."
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
if [ $? -eq 0 ]; then
    echo -e "${GREEN}GPG key added successfully${NC}"
else
    echo -e "${RED}Error: Failed to add Docker GPG key${NC}"
    exit 1
fi  # FIXED: Added closing brace here

# Set up the stable repository
echo "Adding Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Docker repository added${NC}"
else
    echo -e "${RED}Error: Failed to add Docker repository${NC}"
    exit 1
fi  # FIXED: Added closing brace here

# Update package index again with Docker repo
echo "Updating package index with Docker repository..."
apt-get update -y || {
    echo -e "${RED}Error: Failed to update package index with Docker repo${NC}"
    exit 1
}

# Install Docker Engine
echo "Installing Docker Engine..."
apt-get install -y docker-ce docker-ce-cli containerd.io || {
    echo -e "${RED}Error: Failed to install Docker Engine${NC}"
    exit 1
}

# Start and enable Docker service
echo "Starting and enabling Docker service..."
systemctl start docker
systemctl enable docker
if systemctl is-active docker > /dev/null; then
    echo -e "${GREEN}Docker service is running${NC}"
else
    echo -e "${RED}Error: Docker service failed to start${NC}"
    exit 1
fi  # FIXED: Added closing brace here

# Install Docker Compose
echo "Installing Docker Compose..."
DOCKER_COMPOSE_VERSION="2.24.6"  # Use a specific version; check latest at https://github.com/docker/compose/releases
curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
if [ $? -eq 0 ]; then
    chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose installed successfully${NC}"
else
    echo -e "${RED}Error: Failed to download Docker Compose${NC}"
    exit 1
fi  # FIXED: Added closing brace here

# Verify installations
echo "Verifying Docker installation..."
docker --version || {
    echo -e "${RED}Error: Docker not installed correctly${NC}"
    exit 1
}
docker-compose --version || {
    echo -e "${RED}Error: Docker Compose not installed correctly${NC}"
    exit 1
}

# Add current user to docker group (optional, for non-root access)
CURRENT_USER=$(logname)
if [ -n "$CURRENT_USER" ]; then
    echo "Adding user $CURRENT_USER to docker group..."
    usermod -aG docker "$CURRENT_USER"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}User $CURRENT_USER added to docker group${NC}"
        echo "You may need to log out and back in for this to take effect."
    else
        echo -e "${RED}Warning: Failed to add user to docker group${NC}"
    fi
else
    echo -e "${RED}Warning: Could not determine current user for docker group${NC}"
fi

echo -e "${GREEN}Docker setup completed successfully!${NC}"
echo "Docker version: $(docker --version)"
echo "Docker Compose version: $(docker-compose --version)"
echo "To test Docker, run: docker run hello-world"
echo "If you added a user to the docker group, log out and back in to use Docker without sudo."

##########################################
# DOCKER SETUP ENDS
#########################################



# Login to DockerHub with PAT
echo -e "${GREEN}Logging into DockerHub...${NC}"
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
if [ $? -eq 0 ]; then
    echo -e "${GREEN}DockerHub login successful${NC}"
else
    echo -e "${RED}Error: DockerHub login failed. Check credentials.${NC}"
    exit 1
fi

# Ensure log file exists on the host
LOG_FILE="./spherex-agent.log"
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${GREEN}Creating empty $LOG_FILE for container logs...${NC}"
    touch "$LOG_FILE"
    chmod 666 "$LOG_FILE"  # Ensure it’s writable by container
elif [ -d "$LOG_FILE" ]; then
    echo -e "${RED}Error: $LOG_FILE is a directory, but a file is expected${NC}"
    echo "Please remove or rename the directory and rerun the script."
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
    restart: unless-stopped
    volumes:
      - ${LOG_FILE}:/app/spherex-agent.log
    command: bash -c "python main.py >> /app/spherex-agent.log 2>&1"
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