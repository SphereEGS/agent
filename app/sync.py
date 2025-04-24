import time
from threading import Lock, Thread

import requests

from app.config import API_BASE_URL, UPDATE_INTERVAL, GATE, logger


class SyncManager:
    def __init__(self):
        self.allowed_plates = set()
        self.lock = Lock()
        self.start()

    def start(self):
        Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self):
        while True:
            self._update_allowed_plates()
            time.sleep(UPDATE_INTERVAL)

    def _update_allowed_plates(self):
        try:
            logger.info(f"Updating allowed plates for {GATE}")
            response = requests.get(
                f"{API_BASE_URL}/api/method/spherex.api.license_plate.get_authorized_plates",
                params={"gate": GATE},
                verify=False
            )
            if response.status_code == 200:
                with self.lock:
                    self.allowed_plates = set(response.json()["data"])
            else:
                logger.error(
                    f"Error updating allowed plates: {response.status_code}"
                )
            print("Allowed plates updated:", self.allowed_plates)
        except Exception as e:
            logger.error(f"Error updating allowed plates: {e}")

    def is_authorized(self, plate):
        with self.lock:
            return (
                plate in self.allowed_plates
                or plate[::-1] in self.allowed_plates
            )