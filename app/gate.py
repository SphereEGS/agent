import requests
import urllib3
import time
import threading

from app.config import CONTROLLER_IP, DOOR_ID, logger
from app.config import CONTROLLER_USER, CONTROLLER_PASSWORD

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GateControl:
    def __init__(self):
        self.base_url = f"https://{CONTROLLER_IP}/api"
        self.door_id = DOOR_ID
        self.session_id = None
        self.lock = threading.Lock()
        self.login()

    def login(self):
        threading.Thread(target=self._login, daemon=True).start()

    def _login(self):
        while True:
            logger.info("🔑 Rotating session id")
            try:
                payload = {
                    "User": {
                        "login_id": CONTROLLER_USER,
                        "password": CONTROLLER_PASSWORD,
                    }
                }
                response = requests.post(
                    f"{self.base_url}/login",
                    json=payload,
                    verify=False,
                    timeout=5,
                )
                response.raise_for_status()
                session_id = response.headers.get("bs-session-id")
                if session_id:
                    self.session_id = session_id
                    logger.info(f"✅ Session id updated: {self.session_id}")
                else:
                    logger.error("❌ Failed to update session id.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Login failed: {e}")
            except Exception as e:
                logger.error(f"❌ Error rotating session id: {e}")
            time.sleep(600)

    def _call_api(self, endpoint, action):
        url = f"{self.base_url}/doors/{endpoint}"
        payload = {
            "DoorCollection": {
                "total": 1,
                "rows": [{"id": self.door_id}],
            }
        }

        headers = {
            "accept": "application/json",
            "bs-session-id": self.session_id,
            "Content-Type": "application/json",
        }
        logger.info(f"{action} door")
        logger.info(f"Headers: {headers}")
        response = requests.post(
            url, json=payload, headers=headers, verify=False
        )
        if response.ok:
            logger.info(f"{action} completed")
        else:
            logger.error(f"Error during {action}: {response.text}")

    def open(self):
        self._call_api("open", "🚪 Open")
