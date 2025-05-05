import time
from threading import Lock, Thread

import requests

from app.config import API_BASE_URL, UPDATE_INTERVAL, GATE, logger


class SyncManager:
    def __init__(self):
        self.allowed_plates = set()
        self.fuzzy_plates = set()  # Store off-by-one variants
        self.lock = Lock()
        self.start()

    def start(self):
        Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self):
        while True:
            self._update_allowed_plates()
            time.sleep(UPDATE_INTERVAL)

    def _generate_fuzzy_variants(self, plate):
        """
        Generate variants of the plate with first or last character removed.
        Only for plates of length >= 4 (to avoid too-short variants).
        """
        variants = set()
        if len(plate) > 3:
            # Remove first character
            variants.add(plate[1:])
            # Remove last character
            variants.add(plate[:-1])
        return variants

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
                    # Precompute fuzzy variants
                    fuzzy = set()
                    for plate in self.allowed_plates:
                        fuzzy.update(self._generate_fuzzy_variants(plate))
                        # Also add reversed variants
                        rev = plate[::-1]
                        fuzzy.update(self._generate_fuzzy_variants(rev))
                    self.fuzzy_plates = fuzzy
            else:
                logger.error(
                    f"Error updating allowed plates: {response.status_code}"
                )
            print("Allowed plates updated:", self.allowed_plates)
        except Exception as e:
            logger.error(f"Error updating allowed plates: {e}")

    def is_authorized(self, plate):
        with self.lock:
            # Check exact and reversed
            if plate in self.allowed_plates or plate[::-1] in self.allowed_plates:
                return True
            # Check fuzzy variants (off-by-one at start/end)
            if plate in self.fuzzy_plates or plate[::-1] in self.fuzzy_plates:
                return True
            return False