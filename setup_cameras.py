#!/usr/bin/env python
import os
from dotenv import load_dotenv
from app.config import CAMERA_URLS, logger
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description='Camera setup utility for SpherexAgent')
    parser.add_argument('--configure', action='store_true', help='Configure ROIs for all cameras')
    parser.add_argument('--list', action='store_true', help='List all configured cameras')
    parser.add_argument('--check', action='store_true', help='Check ROI configurations for all cameras')
    
    args = parser.parse_args()
    
    if args.list:
        list_cameras()
    elif args.configure:
        configure_cameras()
    elif args.check:
        check_roi_configs()
    else:
        # Default action if no arguments provided
        print("Spherex Camera Setup Utility")
        print("============================")
        print("Available actions:")
        print("  --list       List all configured cameras")
        print("  --configure  Configure ROIs for all cameras")
        print("  --check      Check ROI configurations for all cameras")
        print("\nExample: python setup_cameras.py --list")
        
def list_cameras():
    """List all configured cameras from environment variables"""
    print(f"Found {len(CAMERA_URLS)} cameras:")
    for idx, (camera_id, url) in enumerate(CAMERA_URLS.items(), 1):
        print(f"{idx}. {camera_id:<10}: {url}")
        
    # Show ROI configuration status
    print("\nROI Configuration Status:")
    for camera_id in CAMERA_URLS.keys():
        config_path = f"configs/roi_{camera_id}.json"
        status = "✅ Configured" if os.path.exists(config_path) else "❌ Not configured"
        print(f"  {camera_id:<10}: {status}")
    
    if "main" in CAMERA_URLS and os.path.exists("config.json"):
        print("\nFound legacy config.json for main camera")

def check_roi_configs():
    """Check existing ROI configurations"""
    os.makedirs("configs", exist_ok=True)
    
    print("Checking ROI configurations...")
    configs = glob.glob("configs/roi_*.json")
    
    if not configs:
        print("No camera-specific ROI configurations found.")
    else:
        print(f"Found {len(configs)} camera configurations:")
        for config in configs:
            camera_id = os.path.basename(config).replace("roi_", "").replace(".json", "")
            print(f"  {camera_id}")
    
    # Check legacy config
    if os.path.exists("config.json"):
        print("\nFound legacy config.json file (used for 'main' camera)")

def configure_cameras():
    """Generate commands to configure ROIs for all cameras"""
    print("ROI Configuration Instructions")
    print("=============================")
    print("Run the following commands to configure each camera's ROI:")
    print()
    
    # Ensure configs directory exists
    os.makedirs("configs", exist_ok=True)
    
    for camera_id in CAMERA_URLS.keys():
        config_path = f"configs/roi_{camera_id}.json"
        status = "exists" if os.path.exists(config_path) else "missing"
        
        print(f"Camera: {camera_id} (configuration {status})")
        print(f"  python roi.py --camera {camera_id}")
        print()
    
    print("Notes:")
    print("- Configure one camera at a time")
    print("- For each camera, a window will open showing the camera feed")
    print("- Left-click to add points of the ROI polygon")
    print("- Right-click to complete the polygon (minimum 3 points)")
    print("- Press 'Esc' to cancel without saving")

if __name__ == "__main__":
    main() 