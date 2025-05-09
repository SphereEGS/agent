import argparse
import json
import cv2
from typing import Any, Dict, List, Tuple
from .config import config
from .logging import logger
from .tracking import Tracker, MAX_DISPLAY_HEIGHT
from .lpr import LPR
from .backend_sync import BackendSync
from .gate_control import GateControl


def main() -> None:
    parser = argparse.ArgumentParser(description="SphereX Agent")
    parser.add_argument(
        "--roi-entry", action="store_true", help="Redraw ROI for Entry gate"
    )
    parser.add_argument(
        "--roi-exit", action="store_true", help="Redraw ROI for Exit gate"
    )
    args = parser.parse_args()

    # Initialize shared components once
    lpr = LPR()
    backend_sync = BackendSync()
    gate_control = GateControl()

    trackers: List[Tuple[Tracker, str]] = []
    updated_config: Dict[str, Any] = {
        "lpr_model": config.lpr_model,
        "backend_url": config.backend_url,
        "gate": config.gate,
        "gpu": config.gpu,
        "entry": config.entry,
        "exit": config.exit,
    }

    if config.entry:
        entry_tracker = Tracker(
            gate_type="Entry",
            camera_url=config.entry["camera_url"],
            roi_points=config.entry["roi"],
            lpr=lpr,
            backend_sync=backend_sync,
            gate_control=gate_control,
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
            lpr=lpr,
            backend_sync=backend_sync,
            gate_control=gate_control,
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

    # Initialize tracker generators
    tracker_generators = [
        (tracker, gate_type, tracker.track_and_capture())
        for tracker, gate_type in trackers
    ]

    while True:
        try:
            for tracker, gate_type, generator in tracker_generators:
                try:
                    display_frame, _ = next(generator)
                    if display_frame is None:
                        logger.error(
                            f"Gate {config.gate} ({gate_type}): Tracker stopped unexpectedly"
                        )
                        return

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

                    cv2.imshow(
                        f"Vehicle Tracking ({gate_type})", resized_frame
                    )
                except StopIteration:
                    logger.error(
                        f"Gate {config.gate} ({gate_type}): Tracker stream ended"
                    )
                    return
                except Exception as e:
                    logger.error(
                        f"Gate {config.gate} ({gate_type}): Tracker error: {e}"
                    )
                    continue

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            logger.error(f"Gate {config.gate}: Error processing frame: {e}")
            break

    for _, gate_type in trackers:
        cv2.destroyWindow(f"Vehicle Tracking ({gate_type})")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
