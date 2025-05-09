import os
import threading
import time
from typing import Any, Dict, NoReturn

import cv2
import requests
import urllib3
from numpy.typing import NDArray

from .config import config
from .logging import logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BackendSync:
    def __init__(self) -> None:
        self.allowed_plates: set[str] = set()
        self.lock = threading.Lock()
        self.fetch_thread = threading.Thread(
            target=self._fetch_loop, daemon=True
        )
        self.fetch_thread.start()

    def _fetch_loop(self) -> NoReturn:
        while True:
            try:
                response = requests.get(
                    f"{config.backend_url}/api/method/spherex.api.license_plate.get_authorized_plates",
                    params={"gate": config.gate},
                )
                if response.status_code == 200:
                    with self.lock:
                        self.allowed_plates = set(response.json()["data"])
                    logger.info(
                        f"Gate {config.gate}: Synced {len(self.allowed_plates)} authorized plates"
                    )
                else:
                    logger.error(
                        f"Gate {config.gate}: Failed to fetch authorized plates: {response.status_code}"
                    )
            except Exception as e:
                logger.error(
                    f"Gate {config.gate}: Error syncing authorized plates: {e}"
                )
            time.sleep(60)

    def is_authorized(self, plate: str) -> bool:
        with self.lock:
            return plate in self.allowed_plates

    def log_to_backend(
        self,
        gate_type: str,
        plate: str,
        authorized: bool,
        frame: NDArray[Any],
        track_id: int,
    ) -> None:
        temp_file = ""
        try:
            temp_file = f"gate_{config.gate}_{gate_type}_{track_id}_{int(time.time())}.jpg"
            cv2.imwrite(temp_file, frame)
            log_data: Dict[str, str | int] = {
                "gate": config.gate,
                "license_plate": plate,
                "authorized": 1 if authorized else 0,
                "access_type": gate_type,
                "camera": f"{config.gate}_{gate_type.lower()}_camera",
            }
            with open(temp_file, "rb") as image_file:
                files = {"file": image_file}
                upload_response = requests.post(
                    f"{config.backend_url}/api/method/spherex.api.upload_file",
                    files=files,
                )
                upload_response.raise_for_status()
                log_data["image"] = upload_response.json()["message"][
                    "file_url"
                ]
            response = requests.post(
                f"{config.backend_url}/api/resource/Gate Entry Log",
                json=log_data,
            )
            response.raise_for_status()
            logger.info(
                f"Gate {config.gate} ({gate_type}): Logged {'authorized' if authorized else 'unauthorized'} plate {plate} for vehicle {track_id} to backend"
            )
        except Exception as e:
            logger.error(
                f"Gate {config.gate} ({gate_type}): Error logging plate {plate} for vehicle {track_id}: {e}"
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
