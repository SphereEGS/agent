import time
from threading import Lock, Thread

import requests

from app.config import API_BASE_URL, UPDATE_INTERVAL, GATE, logger


class SyncManager:
    def __init__(self):
        self.allowed_plates = set()
        self.fuzzy_plates = set()    # missing‐char variants of allowed plates
        self.lock = Lock()
        self.start()

    def start(self):
        Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self):
        while True:
            self._update_allowed_plates()
            time.sleep(UPDATE_INTERVAL)

    def _generate_missing_variants(self, plate: str) -> set[str]:
        """
        Precompute all ways to REMOVE one char from the allowed plate.
        E.g. "ABC123" -> {"BC123","AC123",...,"ABC12"}.
        """
        variants = set()
        if len(plate) > 3:
            for i in range(len(plate)):
                variants.add(plate[:i] + plate[i+1:])
        return variants

    def _update_allowed_plates(self):
        try:
            logger.info(f"Updating allowed plates for gate={GATE}")
            resp = requests.get(
                f"{API_BASE_URL}/api/method/spherex.api.license_plate.get_authorized_plates",
                params={"gate": GATE},
                verify=False
            )
            resp.raise_for_status()

            data = resp.json()["data"]
            with self.lock:
                self.allowed_plates = set(data)
                fuzzy = set()
                for p in self.allowed_plates:
                    # missing‐char variants of forward and reversed
                    fuzzy.update(self._generate_missing_variants(p))
                    fuzzy.update(self._generate_missing_variants(p[::-1]))
                self.fuzzy_plates = fuzzy

            logger.info(f"Loaded {len(self.allowed_plates)} plates and "
                        f"{len(self.fuzzy_plates)} missing‐char variants")
        except Exception as e:
            logger.error(f"Failed to update plates: {e}")

    def is_authorized(self, plate: str) -> bool:
        logger.info(f'Received plate "{plate}"')
        rev = plate[::-1]

        with self.lock:
            # 1) Exact or reversed exact?
            if plate in self.allowed_plates:
                logger.info(f'Exact match "{plate}" → AUTHORIZED')
                return True
            if rev in self.allowed_plates:
                logger.info(f'Reversed exact match "{plate}" → AUTHORIZED')
                return True

            # 2) Missing‐char (OCR dropped something) → check precomputed fuzzy_plates
            if plate in self.fuzzy_plates:
                logger.info(f'Missing‐char match "{plate}" ∈ precomputed variants → AUTHORIZED')
                return True
            if rev in self.fuzzy_plates:
                logger.info(f'Missing‐char on reversed "{plate}" ∈ variants → AUTHORIZED')
                return True

            # 3) Extra‐char (OCR injected something) → try dropping each char of the input
            logger.info('No exact/missing‐char match; trying extra‐char drop‐one')
            for i in range(len(plate)):
                variant = plate[:i] + plate[i+1:]
                if variant in self.allowed_plates:
                    logger.info(f'Extra‐char drop at pos {i} → "{variant}" found → AUTHORIZED')
                    return True
                if variant[::-1] in self.allowed_plates:
                    logger.info(f'Extra‐char drop pos {i}, reversed variant "{variant[::-1]}" → AUTHORIZED')
                    return True

            # 4) Still nothing? DENY
            logger.info(f'No match found for "{plate}" → DENIED')
            return False
