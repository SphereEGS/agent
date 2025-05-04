import requests
import urllib3
import threading
import socketio
import time

from app.config import (
    logger,
    SOCKETIO_SERVER,
    SOCKETIO_NAMESPACE,
    CONTROLLER_IP,
    CONTROLLER_USER,
    CONTROLLER_PASSWORD,
    GATE_IDS,
    DOOR_ID,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GateControl:
    def __init__(self):
        self.base_url = f"https://{CONTROLLER_IP}/api"
        self.door_id = str(DOOR_ID)
        self.username = CONTROLLER_USER
        self.password = CONTROLLER_PASSWORD
        self.session_id = None
        self.lock = threading.Lock()
        self.sio = None
        # self._connect_socketio()
        self.login()

    def _connect_socketio(self):
        """Connect to Socket.IO server and emit 'connect_agent' event."""
        try:
            self.sio = socketio.Client(logger=True, engineio_logger=True)
            self.sio.connect(SOCKETIO_SERVER, namespaces=[SOCKETIO_NAMESPACE])
            payload = {
                "agent": "AGENT-123",
                "gates": [self.door_id],
            }
            self.sio.emit("connect_agent", payload, namespace=SOCKETIO_NAMESPACE)
            logger.info(f"Emitted 'connect_agent' with payload {payload} to {SOCKETIO_SERVER}{SOCKETIO_NAMESPACE}")
        except Exception as e:
            logger.error(f"Socket.IO connection failed: {e}")
            self.sio = None

    def login(self):
        """Start a thread to handle login and session renewal."""
        threading.Thread(target=self._login, daemon=True).start()

    def _login(self):
        """Login to the gate system and renew session every 10 minutes."""
        while True:
            logger.info("🔑 Rotating session id")
            try:
                response = requests.post(
                    f"{self.base_url}/login",
                    headers={
                        "accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json={
                        "User": {
                            "login_id": self.username,
                            "password": self.password,
                        }
                    },
                    verify=False,
                    timeout=5,
                )
                response.raise_for_status()
                session_id = response.headers.get("bs-session-id")
                if session_id:
                    with self.lock:
                        self.session_id = session_id
                    logger.info(f"✅ Session id updated: {self.session_id}")
                else:
                    logger.error("❌ Failed to update session id")
            except requests.exceptions.RequestException as e:
                logger.error(f"Login failed: {e}")
            except Exception as e:
                logger.error(f"❌ Error rotating session id: {e}")
            time.sleep(600)  # Renew every 10 minutes

    def open(self, cam_id):
        """Open the door associated with the given cam_id."""
        door_id = GATE_IDS.get(cam_id)
        if not door_id:
            logger.error(f"No door_id found for cam_id: {cam_id}")
            return

        if not self.session_id:
            logger.error("No valid session id")
            return

        url = f"{self.base_url}/doors/open"
        payload = {
            "DoorCollection": {
                "total": 1,
                "rows": [{"id": str(door_id)}],
            }
        }

        headers = {
            "accept": "application/json",
            "bs-session-id": self.session_id,
            "Content-Type": "application/json",
        }
        logger.info(f"🚪 Open door {door_id}")
        try:
            response = requests.post(url, json=payload, headers=headers, verify=False, timeout=5)
            if response.ok:
                logger.info(f"🚪 Open completed for door {door_id}")
            else:
                logger.error(f"Error during 🚪 Open for door {door_id}: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to open door {door_id}: {e}")
