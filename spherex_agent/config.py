from __future__ import annotations
import json
import os
from typing import List


class Config:
    def __init__(self, config_path: str = "config.json") -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        with open(config_path, "r") as f:
            config = json.load(f)
        self.camera_url: str = config.get("camera_url", "")
        self.roi: List[List[int]] = config.get("roi", [])
        self.lpr_model: str = config.get("lpr_model", "lpr_nano.pt")
        self.backend_url: str = config.get("backend_url", "")
        self.gate: str = config.get("gate", "")
        self.gate_type: str = config.get("gate_type", "Entry")
        if not all(
            [
                self.camera_url,
                self.roi,
                self.lpr_model,
                self.backend_url,
                self.gate,
            ]
        ):
            raise ValueError(
                "Config must contain camera_url, roi, lpr_model, backend_url, and gate"
            )


config = Config()
