from __future__ import annotations
import threading
import os
from .logging import logger
from .config import config
from .spherex_suprema_ext import Gate as EmbeddedGate


class GateControl:
    def __init__(self, port: str | None = None):
        if port is None:
            port = os.getenv("GATE_SERIAL_PORT", None)
        try:
            self.gate = EmbeddedGate(port=port)
            logger.info(f"Gate {config.gate}: Connected to gate controller")
        except Exception as e:
            logger.error(
                f"Gate {config.gate}: Failed to initialize gate controller: {e}"
            )
            self.gate = None
        self.lock = threading.Lock()

    def open(self):
        with self.lock:
            if not self.gate:
                logger.error(
                    f"Gate {config.gate}: Gate controller not initialized"
                )
                return
            try:
                barrier = (
                    self.gate.exit_barrier
                    if config.gate_type == "Exit"
                    else self.gate.entry_barrier
                )
                logger.info(
                    f"Gate {config.gate}: Opening {config.gate_type.lower()} gate"
                )
                barrier.open()
                logger.info(
                    f"Gate {config.gate}: {config.gate_type} gate opened successfully"
                )
            except Exception as e:
                logger.error(
                    f"Gate {config.gate}: Error opening {config.gate_type.lower()} gate: {e}"
                )

    def close(self):
        with self.lock:
            if not self.gate:
                logger.error(
                    f"Gate {config.gate}: Gate controller not initialized"
                )
                return
            try:
                barrier = (
                    self.gate.exit_barrier
                    if config.gate_type == "Exit"
                    else self.gate.entry_barrier
                )
                logger.info(
                    f"Gate {config.gate}: Closing {config.gate_type.lower()} gate"
                )
                barrier.close()
                logger.info(
                    f"Gate {config.gate}: {config.gate_type} gate closed successfully"
                )
            except Exception as e:
                logger.error(
                    f"Gate {config.gate}: Error closing {config.gate_type.lower()} gate: {e}"
                )
