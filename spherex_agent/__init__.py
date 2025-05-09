import argparse
import json
import queue
import threading
from typing import Any, Dict

import cv2

from .config import config
from .logging import logger
from .tracking import Tracker, MAX_DISPLAY_HEIGHT


def run_tracker(
    tracker: Tracker, gate_type: str, frame_queue: queue.Queue[Any]
) -> None:
    try:
        for display_frame, _ in tracker.track_and_capture():
            frame_queue.put((gate_type, display_frame))
    except Exception as e:
        logger.error(f"Gate {config.gate} ({gate_type}): Tracker error: {e}")
        frame_queue.put((gate_type, None))  # Signal error


def main() -> None:
    parser = argparse.ArgumentParser(description="SphereX Agent")
    parser.add_argument(
        "--roi-entry", action="store_true", help="Redraw ROI for Entry gate"
    )
    parser.add_argument(
        "--roi-exit", action="store_true", help="Redraw ROI for Exit gate"
    )
    args = parser.parse_args()

    frame_queue: queue.Queue[Any] = queue.Queue()
    trackers = []
    updated_config: Dict[str, Any] = {
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

    threads = []
    for tracker, gate_type in trackers:
        thread = threading.Thread(
            target=run_tracker,
            args=(tracker, gate_type, frame_queue),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    while True:
        try:
            gate_type, display_frame = frame_queue.get(timeout=1.0)
            if display_frame is None:
                logger.error(
                    f"Gate {config.gate} ({gate_type}): Tracker stopped unexpectedly"
                )
                break

            # Resize frame to fit within MAX_DISPLAY_HEIGHT while preserving aspect ratio
            orig_height, orig_width = display_frame.shape[:2]
            scale_factor = min(MAX_DISPLAY_HEIGHT / orig_height, 1.0)
            display_height = int(orig_height * scale_factor)
            display_width = int(orig_width * scale_factor)
            resized_frame = cv2.resize(
                display_frame,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow(f"Vehicle Tracking ({gate_type})", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Gate {config.gate}: Error displaying frame: {e}")
            break

    for tracker, gate_type in trackers:
        cv2.destroyWindow(f"Vehicle Tracking ({gate_type})")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
