import requests
from app.config import CONTROLLER_IP, CONTROLLER_KEY, DOOR_ID


class GateControl:
    def __init__(self):
        self.base_url = f"https://{CONTROLLER_IP}/api"
        self.headers = {
            "Authorization": CONTROLLER_KEY,
            "Content-Type": "application/json",
        }
        self.door_id = DOOR_ID

    def _call_api(self, endpoint, action):
        url = f"{self.base_url}/doors/{endpoint}"
        payload = {"DoorCollection": {"rows": [{"id": self.door_id}]}}
        response = requests.post(
            url, json=payload, headers=self.headers, verify=False
        )
        if response.ok:
            print(f"{action} completed")
        else:
            print(f"Error during {action}: {response.text}")

    def lock(self):
        self._call_api("lock", "🔒 Lock")

    def unlock(self):
        self._call_api("unlock", "🔓 Unlock")
