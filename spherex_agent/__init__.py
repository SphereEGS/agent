from typing import Any
import cv2
import argparse
from .tracking import Tracker
from .config import config
from .logging import logger
import logging
import threading
import json

logging.getLogger("ultralytics").setLevel(logging.ERROR)


def run_tracker(tracker: Tracker, gate_type: str):
    for display_frame, _ in tracker.track_and_capture():
        cv2.imshow(f"Vehicle Tracking ({gate_type})", display_frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="SphereX Agent")
    parser.add_argument(
        "--roi-entry", action="store_true", help="Redraw ROI for Entry gate"
    )
    parser.add_argument(
        "--roi-exit", action="store_true", help="Redraw ROI for Exit gate"
    )
    args = parser.parse_args()

    trackers: list[tuple[Tracker, str]] = []
    updated_config: dict[str, Any] = {
        "lpr_model": config.lpr_model,
        "backend_url": config.backend_url,
        "gate": config.gate,
        "entry": config.entry,
        "exit": config.exit,
    }

    if config.entry:
        entry_tracker = Tracker(
            gate_type="Entry",
            camera_url=config.entry["camera_url"],
            roi_points=config.entry["roi"],
        )
        if not config.entry["roi"] or args.roi_entry:
            new_roi = entry_tracker.draw_roi()
            updated_config["entry"]["roi"] = new_roi
            logger.info(
                f"Gate {config.gate} (Entry): ROI updated in config.json"
            )
        trackers.append((entry_tracker, "Entry"))

    if config.exit:
        exit_tracker = Tracker(
            gate_type="Exit",
            camera_url=config.exit["camera_url"],
            roi_points=config.exit["roi"],
        )
        if not config.exit["roi"] or args.roi_exit:
            new_roi = exit_tracker.draw_roi()
            updated_config["exit"]["roi"] = new_roi
            logger.info(
                f"Gate {config.gate} (Exit): ROI updated in config.json"
            )
        trackers.append((exit_tracker, "Exit"))

    with open("config.json", "w") as f:
        json.dump(updated_config, f, indent=4)

    threads: list[threading.Thread] = []
    for tracker, gate_type in trackers:
        thread = threading.Thread(
            target=run_tracker, args=(tracker, gate_type), daemon=True
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
