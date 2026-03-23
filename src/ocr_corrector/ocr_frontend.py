"""Optional NDLOCR-Lite OCR frontend."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def is_ndlocr_available() -> bool:
    """Check if ndlocr-lite is importable."""
    try:
        import ndlocr_lite  # noqa: F401
        return True
    except ImportError:
        return False


def ocr_image(image_path: str | Path, output_dir: str | Path | None = None) -> str:
    """Run NDLOCR-Lite on an image and return the recognized text.

    Args:
        image_path: Path to the input image.
        output_dir: Directory for OCR output. Uses a temp dir if None.

    Returns:
        The recognized text as a string.

    Raises:
        ImportError: If ndlocr-lite is not installed.
        RuntimeError: If OCR fails.
    """
    if not is_ndlocr_available():
        raise ImportError(
            "ndlocr-lite is not installed. Install with: pip install ndlocr-lite"
        )

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ocr_corrector_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run ndlocr-lite CLI
    try:
        result = subprocess.run(
            [
                "python", "-m", "ndlocr_lite.ocr",
                "--sourceimg", str(image_path),
                "--output", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"NDLOCR-Lite failed: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("Could not run ndlocr-lite. Is it installed and in PATH?")

    # Read output text
    txt_files = list(output_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No text output found in {output_dir}")

    # Concatenate all text files
    texts = []
    for tf in sorted(txt_files):
        texts.append(tf.read_text(encoding="utf-8"))

    return "\n".join(texts)
