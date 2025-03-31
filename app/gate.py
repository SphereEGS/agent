import requests
import urllib3
import time
import threading

from app.config import CONTROLLERS, logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GateControl:
    def __init__(self, gate_type="entry"):
        self.gate_type = gate_type
        if gate_type not in CONTROLLERS or not CONTROLLERS[gate_type]["ip"]:
            logger.warning(f"No controller configuration for {gate_type}. Gate control disabled.")
            self.enabled = False
            return
            
        self.enabled = True
        controller = CONTROLLERS[gate_type]
        self.base_url = f"https://{controller['ip']}/api"
        self.CONTROLLER_GATE = controller["CONTROLLER_GATE"]
        self.username = controller["user"]
        self.password = controller["password"]
        self.session_id = None
        self.lock = threading.Lock()
        self.login()

    def login(self):
        if not self.enabled:
            return
        threading.Thread(target=self._login, daemon=True).start()

    def _login(self):
        while True:
            if not self.enabled:
                return
                
            logger.info(f"🔑 Rotating session id for {self.gate_type} gate")
            try:
                payload = {
                    "User": {
                        "login_id": self.username,
                        "password": self.password,
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
                    logger.info(f"✅ Session id updated for {self.gate_type}: {self.session_id}")
                else:
                    logger.error(f"❌ Failed to update session id for {self.gate_type}.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Login failed for {self.gate_type}: {e}")
            except Exception as e:
                logger.error(f"❌ Error rotating session id for {self.gate_type}: {e}")
            time.sleep(600)

    def _call_api(self, endpoint, action):
        if not self.enabled:
            logger.info(f"{action} skipped - {self.gate_type} gate control disabled")
            return
            
        url = f"{self.base_url}/doors/{endpoint}"
        payload = {
            "DoorCollection": {
                "total": 1,
                "rows": [{"id": self.CONTROLLER_GATE}],
            }
        }

        headers = {
            "accept": "application/json",
            "bs-session-id": self.session_id,
            "Content-Type": "application/json",
        }
        logger.info(f"{action} {self.gate_type} door")
        logger.info(f"Headers: {headers}")
        response = requests.post(
            url, json=payload, headers=headers, verify=False
        )
        if response.ok:
            logger.info(f"{action} completed for {self.gate_type}")
        else:
            logger.error(f"Error during {action} for {self.gate_type}: {response.text}")

    def open(self):
        self._call_api("open", "🚪 Open")
