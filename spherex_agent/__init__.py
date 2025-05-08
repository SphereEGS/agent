import cv2
import argparse
from .tracking import Tracker
from .config import config
from .logging import logger
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="SphereX Agent")
    parser.add_argument("--roi", action="store_true", help="Redraw ROI")
    args = parser.parse_args()

    tracker = Tracker()

    if not config.roi or args.roi:
        new_roi = tracker.draw_roi()
        with open("config.json", "w") as f:
            import json

            json.dump(
                {
                    "camera_url": config.camera_url,
                    "roi": new_roi,
                    "lpr_model": config.lpr_model,
                    "backend_url": config.backend_url,
                    "gate": config.gate,
                },
                f,
                indent=4,
            )
        logger.info("ROI updated in config.json")

    for display_frame, _ in tracker.track_and_capture():
        cv2.imshow("Vehicle Tracking", display_frame)


if __name__ == "__main__":
    main()
