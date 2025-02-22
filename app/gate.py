import requests
import urllib3
from app.config import CONTROLLER_IP, CONTROLLER_KEY, DOOR_ID

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GateControl:
    def __init__(self):
        self.base_url = f"https://{CONTROLLER_IP}/api"
        self.headers = {
            "accept": "application/json",
            "bs-session-id": CONTROLLER_KEY,
            "Content-Type": "application/json",
        }
        self.door_id = DOOR_ID

    def _call_api(self, endpoint, action):
        url = f"{self.base_url}/doors/{endpoint}"
        payload = {
            "DoorCollection": {
                "total": 1,
                "rows": [{"id": self.door_id}],
            }
        }
        response = requests.post(
            url, json=payload, headers=self.headers, verify=False
        )
        if response.ok:
            print(f"{action} completed")
        else:
            print(f"Error during {action}: {response.text}")

    def open(self):
        self._call_api("open", "🚪 Open")
