import threading
import os
from .logging import logger
from .config import Config
from .spherex_suprema_ext import Gate as EmbeddedGate


class GateControl:
    def __init__(self, config: Config, port: str | None = None):
        self.config = config
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
                    f"Gate {self.config.gate}: Gate controller not initialized"
                )
                return
            try:
                barrier = (
                    self.gate.exit_barrier
                    if self.config.gate_type == "Exit"
                    else self.gate.entry_barrier
                )
                logger.info(
                    f"Gate {self.config.gate}: Opening {self.config.gate_type.lower()} gate"
                )
                barrier.open()
                logger.info(
                    f"Gate {self.config.gate}: {self.config.gate_type} gate opened successfully"
                )
            except Exception as e:
                logger.error(
                    f"Gate {self.config.gate}: Error opening {self.config.gate_type.lower()} gate: {e}"
                )

    def close(self):
        with self.lock:
            if not self.gate:
                logger.error(
                    f"Gate {self.config.gate}: Gate controller not initialized"
                )
                return
            try:
                barrier = (
                    self.gate.exit_barrier
                    if self.config.gate_type == "Exit"
                    else self.gate.entry_barrier
                )
                logger.info(
                    f"Gate {self.config.gate}: Closing {self.config.gate_type.lower()} gate"
                )
                barrier.close()
                logger.info(
                    f"Gate {self.config.gate}: {self.config.gate_type} gate closed successfully"
                )
            except Exception as e:
                logger.error(
                    f"Gate {self.config.gate}: Error closing {self.config.gate_type.lower()} gate: {e}"
                )
