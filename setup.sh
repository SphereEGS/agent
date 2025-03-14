#!/bin/bash

# Script to setup Spherex Agent on a Ubuntu/Debian-based microserver
# Must be run with sudo privileges

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run this script with sudo${NC}"
    exit 1
fi

echo -e "${GREEN}Starting Spherex Agent Setup...${NC}"

# Git credentials for private repository (dummy values)
GIT_USERNAME="dummy_user"
GIT_PASSWORD="dummy_password"

# Define installation directory
INSTALL_DIR="/root/spherex_agent"
mkdir -p "$INSTALL_DIR"  # Create directory if it doesn't exist

# Check available space
echo "Checking available disk space..."
df -h "$INSTALL_DIR"
AVAILABLE_SPACE=$(df -k "$INSTALL_DIR" | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 5242880 ]; then  # Less than 5GB in KB
    echo -e "${RED}Warning: Less than 5GB available. You may run out of space during installation.${NC}"
    read -p "Continue anyway? (y/N): " CONTINUE
    if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
        echo "Aborting setup."
        exit 1
    fi
fi

# Update system
echo "Updating system packages..."
apt update && apt upgrade -y || echo -e "${RED}Warning: Package update failed, continuing...${NC}"

# Clean up to free space
echo "Cleaning up APT cache to free space..."
apt clean
apt autoremove -y

# Install required software
echo "Installing Python and Git..."
apt install -y python3 python3-pip python3.12-venv git || echo -e "${RED}Warning: Software installation failed, continuing...${NC}"

# Clone the private repository
echo "Cloning Spherex Agent repository..."
cd "$INSTALL_DIR"
if [ -d "agent" ]; then
    echo "Agent directory already exists, pulling latest changes..."
    cd agent
    git pull "https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/SphereEGS/agent.git" || echo -e "${RED}Warning: Git pull failed${NC}"
else
    git clone "https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/SphereEGS/agent.git" || echo -e "${RED}Error: Git clone failed${NC}"
    cd agent || exit 1
fi

# Set up virtual environment
echo "Setting up Python virtual environment..."
rm -rf venv  # Remove any existing venv to start fresh
python3 -m venv venv || echo -e "${RED}Warning: Virtual environment creation failed${NC}"
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt || {
    echo -e "${RED}Error: Dependency installation failed. Likely due to insufficient space or incompatible packages.${NC}"
    echo "Current disk space:"
    df -h "$INSTALL_DIR"
    echo "Try freeing more space or editing requirements.txt."
    exit 1
}

# Check network connectivity function
check_connectivity() {
    local ip=$1
    echo "Testing connectivity to $ip..."
    if ping -c 4 "$ip" > /dev/null 2>&1; then
        echo -e "${GREEN}Successfully reached $ip${NC}"
        return 0
    else
        echo -e "${RED}Could not reach $ip${NC}"
        return 1
    fi
}

# Prompt user for .env variables with immediate IP testing
echo "Now, let's configure the .env file. Please provide the following values:"

read -p "Gate ID on backend server (e.g., Edara-GT-11): " GATE

while true; do
    read -p "Camera URL (e.g., rtsp://10.0.40.11:554): " CAMERA_URL
    CAMERA_IP=$(echo "$CAMERA_URL" | cut -d'/' -f3 | cut -d':' -f1)
    if check_connectivity "$CAMERA_IP"; then
        break
    else
        echo -e "${RED}Camera IP is not reachable.${NC}"
        read -p "Re-enter Camera URL or press Enter to skip: " NEW_CAMERA_URL
        if [ -z "$NEW_CAMERA_URL" ]; then
            echo "Skipping Camera URL validation."
            break
        fi
        CAMERA_URL="$NEW_CAMERA_URL"
    fi
done

while true; do
    read -p "Backend server URL (e.g., http://backend.Spherex.com): " BACKEND_URL
    BACKEND_IP=$(echo "$BACKEND_URL" | cut -d'/' -f3)
    if check_connectivity "$BACKEND_IP"; then
        break
    else
        echo -e "${RED}Backend IP is not reachable.${NC}"
        read -p "Re-enter Backend URL or press Enter to skip: " NEW_BACKEND_URL
        if [ -z "$NEW_BACKEND_URL" ]; then
            echo "Skipping Backend URL validation."
            break
        fi
        BACKEND_URL="$NEW_BACKEND_URL"
    fi
done

while true; do
    read -p "Biostar Controller IP address (e.g., 192.168.2.49): " CONTROLLER_IP
    if check_connectivity "$CONTROLLER_IP"; then
        break
    else
        echo -e "${RED}Controller IP is not reachable.${NC}"
        read -p "Re-enter Controller IP or press Enter to skip: " NEW_CONTROLLER_IP
        if [ -z "$NEW_CONTROLLER_IP" ]; then
            echo "Skipping Controller IP validation."
            break
        fi
        CONTROLLER_IP="$NEW_CONTROLLER_IP"
    fi
done

read -p "Biostar Controller Username (e.g., admin): " CONTROLLER_USER
read -s -p "Biostar Controller Password (e.g., Admin123): " CONTROLLER_PASSWORD
echo "" # Newline after password input
read -p "Gate ID on Biostar (e.g., 21): " GATE_ID
read -p "Font path (default: fonts/DejaVuSans.ttf): " FONT_PATH
FONT_PATH=${FONT_PATH:-"fonts/DejaVuSans.ttf"} # Default if empty
read -p "Model path (default: models/license_yolo8s_1024.pt): " MODEL_PATH
MODEL_PATH=${MODEL_PATH:-"models/license_yolo8s_1024.pt"} # Default if empty

# Create .env file with user-provided values
echo "Creating .env file with your inputs..."
cat > .env << EOL
# Spherex Agent Configuration
GATE="$GATE"                # Gate ID on backend server
CAMERA_URL="$CAMERA_URL"        # URL address of the camera
BACKEND_URL="$BACKEND_URL"   # Backend server URL
CONTROLLER_IP="$CONTROLLER_IP"            # Biostar Server IP address
CONTROLLER_USER="$CONTROLLER_USER"                 # Username for Biostar controller
CONTROLLER_PASSWORD="$CONTROLLER_PASSWORD"          # Password for Biostar User
GATE_ID=$GATE_ID                             # Gate ID on Biostar
FONT_PATH="$FONT_PATH"
MODEL_PATH="$MODEL_PATH"
EOL

# Make run.sh executable
if [ -f "run.sh" ]; then
    echo "Making run.sh executable..."
    chmod +x run.sh
    bash run.sh
else
    echo "Creating basic run.sh file..."
    cat > run.sh << EOL
#!/bin/bash
python main.py > /dev/null 2>&1 &
echo $! > app.pid
tail -f app.log | awk '{lines[NR%10] = $0} NR>=10 {system("clear"); for (i=NR%10+1; i<=NR%10+10; i++) print lines[i%10]}'
EOL
    chmod +x run.sh
    bash run.sh
fi

# Final network connectivity check (optional, since we already validated)
echo "Verifying network connectivity (final confirmation)..."
check_connectivity "$CAMERA_IP"
check_connectivity "$CONTROLLER_IP"
check_connectivity "$BACKEND_IP"

echo -e "${GREEN}Spherex Agent setup completed!${NC}"
echo "Next steps:"
echo "1. Verify the .env file contents:"
echo "   cat ${INSTALL_DIR}/agent/.env"
echo "2. Set the Region of Interest (ROI):"
echo "   cd ${INSTALL_DIR}/agent && source venv/bin/activate && python roi.py"
echo "3. Start the agent:"
echo "   cd ${INSTALL_DIR}/agent && ./run.sh"
echo "Note: Ensure your camera, Biostar server, and backend server are properly configured and accessible."
