import pickle
from pathlib import Path
import requests
from typing import Optional

class AuthManager:
    def __init__(self):
        self.auth_file = Path.home() / ".spherex" / "auth.pickle"
        self.auth_file.parent.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self._load_auth()
        
    def _load_auth(self):
        if self.auth_file.exists():
            with open(self.auth_file, "rb") as f:
                self.session.cookies.update(pickle.load(f))
                
    def _save_auth(self):
        with open(self.auth_file, "wb") as f:
            pickle.dump(self.session.cookies, f)
            
    async def login(self, email: str, password: str) -> bool:
        try:
            response = self.session.post(
                "https://api.spherex.com/v1/auth/login",
                json={"email": email, "password": password},
                timeout=5
            )
            if response.status_code == 200:
                self._save_auth()
                return True
            return False
        except Exception:
            return False
            
    def is_authenticated(self) -> bool:
        try:
            response = self.session.get(
                "https://api.spherex.com/v1/auth/verify",
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
