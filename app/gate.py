import requests
import urllib3
import threading
import time
import socketio
import os

from app.config import (
    logger,
    SOCKETIO_SERVER,
    SOCKETIO_NAMESPACE,
    CONTROLLER_IP,
    CONTROLLER_USER,
    CONTROLLER_PASSWORD,
    GATE_IDS,
)

# Import the new embedded API
from embedded.Sphere_X_Extension.Programming.api.sphereX_ext import Gate as EmbeddedGate

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GateControl:
    def __init__(self, port=None):
        # Allow port override via env/config, else auto-detect
        if port is None:
            port = os.getenv("GATE_SERIAL_PORT", None)
        try:
            self.gate = EmbeddedGate(port=port)
            logger.info(f"Connected to embedded gate controller on port {self.gate.port}")
        except Exception as e:
            logger.error(f"Failed to initialize embedded gate controller: {e}")
            self.gate = None
        self.lock = threading.Lock()

    def open_entry(self):
        with self.lock:
            if not self.gate:
                logger.error("Embedded gate controller not initialized")
                return
            try:
                logger.info("Opening entry barrier")
                self.gate.entry_barrier.open()
                logger.info("✅ Entry barrier opened")
            except Exception as e:
                logger.error(f"❌ Error opening entry barrier: {e}")

    def close_entry(self):
        with self.lock:
            if not self.gate:
                logger.error("Embedded gate controller not initialized")
                return
            try:
                logger.info("Closing entry barrier")
                self.gate.entry_barrier.close()
                logger.info("✅ Entry barrier closed")
            except Exception as e:
                logger.error(f"❌ Error closing entry barrier: {e}")

    def open_exit(self):
        with self.lock:
            if not self.gate:
                logger.error("Embedded gate controller not initialized")
                return
            try:
                logger.info("Opening exit barrier")
                self.gate.exit_barrier.open()
                logger.info("✅ Exit barrier opened")
            except Exception as e:
                logger.error(f"❌ Error opening exit barrier: {e}")

    def close_exit(self):
        with self.lock:
            if not self.gate:
                logger.error("Embedded gate controller not initialized")
                return
            try:
                logger.info("Closing exit barrier")
                self.gate.exit_barrier.close()
                logger.info("✅ Exit barrier closed")
            except Exception as e:
                logger.error(f"❌ Error closing exit barrier: {e}")

    def _connect_socketio(self):
        """Connect to Socket.IO server and emit 'connect_agent' event."""
        try:
            self.sio = socketio.Client(logger=True, engineio_logger=True)
            self.sio.connect(SOCKETIO_SERVER, namespaces=[SOCKETIO_NAMESPACE])
            payload = {
                "agent": "AGENT-123",
                "gates": list(GATE_IDS.values()),  # Send all gate IDs
            }
            self.sio.emit("connect_agent", payload, namespace=SOCKETIO_NAMESPACE)
            logger.info(f"Emitted 'connect_agent' with payload {payload} to {SOCKETIO_SERVER}{SOCKETIO_NAMESPACE}")
        except Exception as e:
            logger.error(f"Socket.IO connection failed: {e}")
            self.sio = None

    def login(self):
        """Start a thread to handle login and session renewal every 10 minutes."""
        threading.Thread(target=self._login_loop, daemon=True).start()

    def _login_loop(self):
        """Login and renew session every 10 minutes."""
        while True:
            logger.info("🔑 Updating session ID")
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
                    logger.info(f"✅ Session ID updated: {self.session_id}")
                else:
                    logger.error("❌ No session ID received")
            except Exception as e:
                logger.error(f"❌ Login failed: {e}")
            time.sleep(600)  # Wait 10 minutes before renewing

    def open(self, cam_id):
        """Open the gate associated with the given cam_id."""
        gate_id = GATE_IDS.get(cam_id)
        if not gate_id:
            logger.error(f"No gate ID found for cam_id: {cam_id}")
            return

        with self.lock:
            if not self.session_id:
                logger.error("No valid session ID available")
                return
            headers = {
                "accept": "application/json",
                "bs-session-id": self.session_id,
                "Content-Type": "application/json",
            }

        url = f"{self.base_url}/doors/open"
        payload = {
            "DoorCollection": {
                "total": 1,
                "rows": [{"id": str(gate_id)}],
            }
        }

        logger.info(f"🚪 Opening gate {gate_id}")
        try:
            response = requests.post(
                url, json=payload, headers=headers, verify=False, timeout=5
            )
            if response.ok:
                logger.info(f"✅ Gate {gate_id} opened successfully")
            else:
                logger.error(f"❌ Failed to open gate {gate_id}: {response.text}")
        except Exception as e:
            logger.error(f"❌ Error opening gate {gate_id}: {e}")
