"""Auto-start and manage a local llama-server process."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
SERVER_BIN_NAME = "llama-server.exe" if IS_WINDOWS else "llama-server"

# Default paths relative to project root (ocr-correction-pipeline/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SERVER_DIR = PROJECT_ROOT / "llm"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "llm" / "models"


def find_server_bin(server_dir: Path | None = None) -> Path | None:
    """Find llama-server binary in project dir or PATH."""
    if server_dir is None:
        server_dir = DEFAULT_SERVER_DIR

    candidate = server_dir / SERVER_BIN_NAME
    if candidate.exists():
        return candidate

    found = shutil.which("llama-server")
    if found:
        return Path(found)

    return None


def find_model(model_dir: Path | None = None) -> Path | None:
    """Find first .gguf file in model directory."""
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    if not model_dir.exists():
        return None

    for f in sorted(model_dir.glob("*.gguf")):
        return f
    return None


def is_server_running(api_base: str = "http://localhost:8080/v1") -> bool:
    """Check if an LLM server is already running."""
    try:
        req = Request(f"{api_base.rstrip('/')}/models", method="GET")
        with urlopen(req, timeout=2):
            return True
    except (URLError, OSError):
        return False


class LlmServerProcess:
    """Manages a llama-server subprocess."""

    def __init__(
        self,
        server_bin: Path,
        model_path: Path,
        port: int = 8080,
        n_gpu_layers: int = 0,
        ctx_size: int = 512,
    ):
        self.server_bin = server_bin
        self.model_path = model_path
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self._process: subprocess.Popen | None = None

    def start(self, timeout: float = 120.0) -> str:
        """Start llama-server and wait until ready. Returns API base URL."""
        api_base = f"http://localhost:{self.port}/v1"

        if is_server_running(api_base):
            logger.info("LLM server already running on port %d", self.port)
            return api_base

        cmd = [
            str(self.server_bin),
            "-m", str(self.model_path),
            "--port", str(self.port),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--ctx-size", str(self.ctx_size),
            "--reasoning-budget", "0",  # disable thinking for A/B judgment
        ]
        logger.info("Starting llama-server: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self._process.returncode}. "
                    f"Check model: {self.model_path}"
                )
            if is_server_running(api_base):
                logger.info("llama-server ready on port %d", self.port)
                return api_base
            time.sleep(0.5)

        self.stop()
        raise TimeoutError(f"llama-server did not start within {timeout}s")

    def stop(self):
        """Stop the llama-server process if we started it."""
        if self._process is not None:
            logger.info("Stopping llama-server (PID %d)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
