"""NDLOCR-Lite OCR frontend."""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ndlocr-lite is cloned into the project root by the installer
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NDLOCR_DIR = PROJECT_ROOT / "ndlocr-lite"
NDLOCR_SRC = NDLOCR_DIR / "src"
NDLOCR_SCRIPT = NDLOCR_SRC / "ocr.py"


def is_ndlocr_available() -> bool:
    """Check if ndlocr-lite is installed (cloned + dependencies)."""
    return NDLOCR_SCRIPT.exists()


def ocr_image(image_path: str | Path, output_dir: str | Path | None = None) -> str:
    """Run NDLOCR-Lite on an image and return the recognized text.

    Args:
        image_path: Path to the input image.
        output_dir: Directory for OCR output. Uses a temp dir if None.

    Returns:
        The recognized text as a string.
    """
    if not is_ndlocr_available():
        raise RuntimeError(
            f"ndlocr-lite not found at {NDLOCR_DIR}. "
            f"Run the installer or: git clone https://github.com/ndl-lab/ndlocr-lite.git"
        )

    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ocr_corrector_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run: python ndlocr-lite/src/ocr.py --sourceimg <image> --output <dir>
    cmd = [
        sys.executable,
        str(NDLOCR_SCRIPT),
        "--sourceimg", str(image_path),
        "--output", str(output_dir),
    ]
    logger.info("Running NDLOCR-Lite: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(NDLOCR_SRC),
        )
        if result.returncode != 0:
            raise RuntimeError(f"NDLOCR-Lite failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("NDLOCR-Lite timed out (120s)")

    # Read output text files
    txt_files = list(output_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No text output found in {output_dir}")

    texts = []
    for tf in sorted(txt_files):
        texts.append(tf.read_text(encoding="utf-8"))

    return "\n".join(texts)
