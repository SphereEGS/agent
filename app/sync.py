import time
from threading import Lock, Thread

import requests

from app.config import API_BASE_URL, UPDATE_INTERVAL, ZONE


class SyncManager:
    def __init__(self):
        self.allowed_plates = set()
        self.lock = Lock()

    def start(self):
        Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self):
        while True:
            self._update_allowed_plates()
            time.sleep(UPDATE_INTERVAL)

    def _update_allowed_plates(self):
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/method/spherex.api.license_plate.get_authorized_plates",
                params={"zone": ZONE},
            )
            if response.status_code == 200:
                with self.lock:
                    self.allowed_plates = set(response.json()["message"])
        except Exception as e:
            print(f"Error updating allowed plates: {e}")

    def is_authorized(self, plate):
        with self.lock:
            return plate in self.allowed_plates or plate[::-1] in self.allowed_plates
