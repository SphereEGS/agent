#!/bin/bash
# Setup script for Jetson Nano GPU acceleration
# Run with: sudo bash setup_jetson.sh

echo "Setting up Jetson Nano for SpherexAgent with GPU acceleration..."

# Make sure we're running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root: sudo bash setup_jetson.sh"
  exit 1
fi

# Setup working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure models directory exists
mkdir -p models

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y python3-pip libpython3-dev libopenblas-dev libopenmpi-dev libjpeg-dev zlib1g-dev

# Ensure pip is up to date
python3 -m pip install -U pip

# Install Python packages
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Install PyTorch and TorchVision optimized for Jetson
echo "Installing PyTorch for Jetson..."
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.1.0a0+41361538.nv23.6-cp38-cp38-linux_aarch64.whl

# Download YOLOv5 model (if not exists)
YOLO_MODEL_PATH="models/yolo11n.pt"
if [ ! -f "$YOLO_MODEL_PATH" ]; then
    echo "Downloading YOLOv5 model..."
    pip3 install gdown
    gdown --id 1GQxcgE2iePE1Egfnnmk7_zDIRMZY9lsL -O "$YOLO_MODEL_PATH"
fi

# Clone and install jetson-inference if not already available
if ! python3 -c "import jetson.inference" &> /dev/null; then
    echo "Installing jetson-inference..."
    cd /tmp
    git clone --recursive https://github.com/dusty-nv/jetson-inference
    cd jetson-inference
    mkdir -p build
    cd build
    cmake ../
    make -j$(nproc)
    make install
    ldconfig
    cd "$SCRIPT_DIR"
fi

# Download SSD-Mobilenet-v2 model for jetson-inference
if [ ! -d "models/ssd-mobilenet-v2" ]; then
    echo "Downloading SSD-Mobilenet-v2 model for Jetson..."
    mkdir -p models/ssd-mobilenet-v2
    cd /tmp
    wget https://nvidia.box.com/shared/static/nlqwmoug5iusqlj8bnnsahb7rq9hb8cm.gz -O ssd_mobilenet_v2.gz
    tar -xzvf ssd_mobilenet_v2.gz
    mv ssd_mobilenet_v2/* "$SCRIPT_DIR/models/ssd-mobilenet-v2/"
    cd "$SCRIPT_DIR"
fi

# Configure CUDA for optimal performance
echo "Configuring Jetson for optimal performance..."

# Set maximum clock speeds
echo "Setting maximum clock speeds..."
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
fi

# Maximize GPU and memory clock
if [ -f /sys/devices/gpu.0/devfreq/17000000.gp10b/governor ]; then
    echo "performance" > /sys/devices/gpu.0/devfreq/17000000.gp10b/governor
fi

# Set maximum power mode using nvpmodel
if command -v nvpmodel &> /dev/null; then
    echo "Setting maximum performance power mode..."
    nvpmodel -m 0
fi

# Enable Jetson maximum performance using jetson_clocks
if command -v jetson_clocks &> /dev/null; then
    echo "Enabling maximum clock speeds with jetson_clocks..."
    jetson_clocks
fi

echo ""
echo "Setup complete! Your Jetson Nano is now configured for GPU-accelerated object detection."
echo "Run the application with: python3 main.py"
echo "" 