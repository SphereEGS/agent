from typing import NoReturn
import requests
import threading
import time
import cv2
import os

import urllib3
from .config import config
from .logging import logger
from numpy.typing import NDArray

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BackendSync:
    def __init__(self):
        self.allowed_plates = set()
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
                    verify=False,
                    timeout=5,
                )
                if response.status_code == 200:
                    with self.lock:
                        self.allowed_plates = set(response.json()["data"])
                    logger.info(
                        f"Synced {len(self.allowed_plates)} authorized plates from backend"
                    )
                else:
                    logger.error(
                        f"Failed to fetch authorized plates: {response.status_code}"
                    )
            except Exception as e:
                logger.error(f"Error syncing authorized plates: {e}")
            time.sleep(30)

    def is_authorized(self, plate: str) -> bool:
        with self.lock:
            return plate in self.allowed_plates

    def log_to_backend(
        self,
        plate: str,
        is_authorized: bool,
        frame: NDArray,
        camera_id: str = "main",
    ):
        try:
            temp_file = f"gate_entry_{camera_id}_{int(time.time())}.jpg"
            cv2.imwrite(temp_file, frame)

            log_data = {
                "gate": config.gate,
                "license_plate": plate,
                "authorized": is_authorized,
                "image": temp_file,
                "access_type": config.gate_type,
                "camera": camera_id,
            }

            with open(temp_file, "rb") as image_file:
                files = {"file": image_file}
                upload_response = requests.post(
                    f"{config.backend_url}/api/method/spherex.api.upload_file",
                    files=files,
                    verify=False,
                    timeout=5,
                )
                upload_response.raise_for_status()
                log_data["image"] = upload_response.json()["message"][
                    "file_url"
                ]

            response = requests.post(
                f"{config.backend_url}/api/resource/Gate Entry Log",
                data=log_data,
                verify=False,
                timeout=5,
            )
            response.raise_for_status()
            logger.info(
                f"Logged {'authorized' if is_authorized else 'unauthorized'} plate {plate} to backend"
            )
        except Exception as e:
            logger.error(f"Error logging plate {plate} to backend: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
