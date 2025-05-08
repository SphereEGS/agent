import requests
import threading
import time
from .config import config
from .logging import logger

class BackendSync:
    def __init__(self):
        self.allowed_plates = set()
        self.lock = threading.Lock()
        self.fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.fetch_thread.start()

    def _fetch_loop(self):
        while True:
            try:
                response = requests.get(
                    f"{config.backend_url}/api/method/spherex.api.license_plate.get_authorized_plates",
                    params={"gate": config.gate},
                    verify=False,
                    timeout=5
                )
                if response.status_code == 200:
                    with self.lock:
                        self.allowed_plates = set(response.json()["data"])
                    logger.info(f"Synced {len(self.allowed_plates)} authorized plates from backend")
                else:
                    logger.error(f"Failed to fetch authorized plates: {response.status_code}")
            except Exception as e:
                logger.error(f"Error syncing authorized plates: {e}")
            time.sleep(30)

    def is_authorized(self, plate: str) -> bool:
        with self.lock:
            return plate in self.allowed_plates

    def log_to_backend(self, plate: str, authorized: bool, track_id: int, camera_id: str = "main"):
        try:
            log_data = {
                "gate": config.gate,
                "license_plate": plate,
                "authorized": 1 if authorized else 0,
                "access_type": "Entry",
                "camera": camera_id
            }
            response = requests.post(
                f"{config.backend_url}/api/resource/Gate Entry Log",
                json=log_data,
                verify=False,
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"Logged {'authorized' if authorized else 'unauthorized'} plate {plate} for vehicle {track_id} to backend")
            else:
                logger.error(f"Failed to log plate {plate} for vehicle {track_id}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error logging plate {plate} for vehicle {track_id}: {e}")