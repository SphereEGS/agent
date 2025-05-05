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
                f"http://localhost:8001/api/method/spherex.api.license_plate.get_authorized_plates",
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
            # First check exact match
            if plate in self.allowed_plates:
                return True
                
            # Then try fuzzy matching for each allowed plate
            plate = plate.strip()  # Remove any whitespace
            if not plate:  # Skip empty plates
                return False
                
            for allowed_plate in self.allowed_plates:
                allowed_plate = allowed_plate.strip()
                if not allowed_plate:
                    continue
                    
                # Skip if lengths are too different
                if abs(len(plate) - len(allowed_plate)) > 1:
                    continue
                    
                # Check if plates match except for first or last character
                if len(plate) >= 6 and len(allowed_plate) >= 6:
                    # Try matching without first character
                    if plate[1:] == allowed_plate[1:] and len(plate) == len(allowed_plate):
                        return True
                        
                    # Try matching without last character
                    if plate[:-1] == allowed_plate[:-1] and len(plate) == len(allowed_plate):
                        return True
                        
                    # If one plate is shorter, check if it's missing first or last char
                    if len(plate) == len(allowed_plate) - 1:
                        if plate == allowed_plate[1:] or plate == allowed_plate[:-1]:
                            return True
                            
                    if len(allowed_plate) == len(plate) - 1:
                        if allowed_plate == plate[1:] or allowed_plate == plate[:-1]:
                            return True
                            
            return False
