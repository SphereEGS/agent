from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class Config:
    def __init__(self, config_path: str = "config.json") -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        with open(config_path, "r") as f:
            config = json.load(f)

        self.lpr_model: str = config.get("lpr_model", "lpr_nano.pt")
        self.backend_url: str = config.get("backend_url", "")
        self.gate: str = config.get("gate", "")

        # Initialize entry and exit configurations
        self.entry: Optional[Dict[str, Any]] = config.get("entry")
        self.exit: Optional[Dict[str, Any]] = config.get("exit")

        # Validate required fields
        if not (self.entry or self.exit):
            raise ValueError("Config must contain at least 'entry' or 'exit'")
        if not all([self.lpr_model, self.backend_url, self.gate]):
            raise ValueError(
                "Config must contain lpr_model, backend_url, and gate"
            )

        # Validate entry configuration
        if self.entry:
            if not (self.entry.get("camera_url") and self.entry.get("roi")):
                raise ValueError(
                    "Entry config must contain camera_url and roi"
                )

        # Validate exit configuration
        if self.exit:
            if not (self.exit.get("camera_url") and self.exit.get("roi")):
                raise ValueError("Exit config must contain camera_url and roi")


config = Config()
