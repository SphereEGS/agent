import requests
import urllib3
import threading
import socketio  # python-socketio client

# Import constants directly instead of the CONTROLLERS dictionary
from app.config import (
    logger,
    SOCKETIO_SERVER,
    SOCKETIO_NAMESPACE,
    CONTROLLER_IP,
    CONTROLLER_USER,
    CONTROLLER_PASSWORD,
    DOOR_ID,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GateControl:
    # Removed gate_type from __init__
    def __init__(self):
        # Use imported constants directly
        self.enabled: bool = True
        self.base_url: str = f"https://{CONTROLLER_IP}/api"
        self.door_id: str = str(DOOR_ID)
        self.username: str = CONTROLLER_USER
        self.password: str = CONTROLLER_PASSWORD
        self.session_id: str | None = None
        self.lock = threading.Lock()
        self.sio: socketio.Client | None = None
        # self._connect_socketio()
        self.login()

    def _connect_socketio(self):
        """Connect to local Socket.IO server and emit 'connect_agent' event."""
        try:
            self.sio = socketio.Client(logger=True, engineio_logger=True)
            self.sio.connect(SOCKETIO_SERVER, namespaces=[SOCKETIO_NAMESPACE])
            # Updated payload: Removed gate_type reference
            # Assuming the server might still expect a 'gates' list,
            # perhaps with the door ID or a generic identifier.
            # If the server doesn't need 'gates', this key can be removed.
            payload = {
                "agent": "AGENT-123", # Consider making agent ID configurable too
                "gates": [self.door_id] # Example: Using door_id as the gate identifier
            }
            self.sio.emit('connect_agent', payload, namespace='/spherex')
            logger.info(f"Emitted 'connect_agent' event with payload {payload} to Socket.IO server at {SOCKETIO_SERVER}{SOCKETIO_NAMESPACE} and staying connected")
        except Exception as e:
            logger.error(f"Socket.IO connection failed: {e}")
            self.sio = None

    def login(self):
        if not self.enabled:
            return
        threading.Thread(target=self._login, daemon=True).start()

    def _login(self):
        """Login to the gate system and get a session ID."""
        try:
            response = requests.post(
                f"{self.base_url}/login",
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json"
                },
                json={
                    "User": {
                        "login_id": self.username,
                        "password": self.password
                    }
                },
                verify=False
            )
            if response.status_code != 200:
                logger.error(f"Login failed with status {response.status_code}: {response.text}")
                return
            self.session_id = response.json().get("session_id")
            if not self.session_id:
                logger.error(f"Login response missing session_id: {response.text}")
                return
            logger.info(f"[OK] Session id updated: {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to login to controller at {CONTROLLER_IP}: {str(e)}")

    def _call_api(self, endpoint, action):
        if not self.enabled:
             # Updated logging: Removed gate_type reference
            logger.info(f"{action} skipped - gate control disabled")
            return

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
         # Updated logging: Removed gate_type reference
        logger.info(f"{action} door {self.door_id}")
        logger.info(f"Headers: {headers}")
        response = requests.post(
            url, json=payload, headers=headers, verify=False
        )
        if response.ok:
             # Updated logging: Removed gate_type reference
            logger.info(f"{action} completed for door {self.door_id}")
        else:
             # Updated logging: Removed gate_type reference
            logger.error(f"Error during {action} for door {self.door_id}: {response.text}")

    def open(self):
        self._call_api("open", "🚪 Open")