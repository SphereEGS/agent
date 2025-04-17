import requests
import urllib3
import threading
import socketio  # python-socketio client

from app.config import CONTROLLERS, logger, SOCKETIO_SERVER, SOCKETIO_NAMESPACE

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GateControl:
    def __init__(self, gate_type: str = "entry"):
        self.gate_type: str = gate_type
        if gate_type not in CONTROLLERS or not CONTROLLERS[gate_type]["ip"]:
            logger.warning(f"No controller configuration for {gate_type}. Gate control disabled.")
            self.enabled: bool = False
            return

        self.enabled: bool = True
        controller: dict = CONTROLLERS[gate_type]
        self.base_url: str = f"https://{controller['ip']}/api"
        self.door_id: str = controller["door_id"]
        self.username: str = controller["user"]
        self.password: str = controller["password"]
        self.session_id: str | None = None
        self.lock = threading.Lock()
        self.sio: socketio.Client | None = None
        self._connect_socketio()
        self.login()

    def _connect_socketio(self):
        """Connect to local Socket.IO server and emit 'connect_agent' event with hardcoded agent and gate."""
        try:
            self.sio = socketio.Client(logger=True, engineio_logger=True)
            self.sio.connect(SOCKETIO_SERVER, namespaces=[SOCKETIO_NAMESPACE])
            payload = {
                "agent": "AGENT-123",
                "gates": [self.gate_type]
            }
            self.sio.emit('connect_agent', payload, namespace='/spherex')
            logger.info(f"Emitted 'connect_agent' event with payload {payload} to Socket.IO server at localhost:9001/spherex and staying connected")
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
                json={"username": self.username, "password": self.password},
                verify=False
            )
            response.raise_for_status()
            self.session_id = response.json()["session_id"]
            logger.info(f"[OK] Session id updated for {self.gate_type}: {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to login to {self.gate_type} gate: {str(e)}")
            raise

    def _call_api(self, endpoint, action):
        if not self.enabled:
            logger.info(f"{action} skipped - {self.gate_type} gate control disabled")
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