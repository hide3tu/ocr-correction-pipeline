"""NDLOCR-Lite OCR frontend."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Structured OCR output (text + bounding boxes for PDF generation)
# ---------------------------------------------------------------------------

@dataclass
class OcrLine:
    """A single text line with its bounding box in image pixel coordinates."""

    text: str
    bbox: tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class OcrPage:
    """Structured OCR result for one page/image."""

    text: str
    lines: list[OcrLine] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    image_path: str = ""


def _parse_ndlocr_json(json_path: Path, image_path: str) -> OcrPage:
    """Parse NDLOCR-Lite JSON output into an OcrPage."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    imginfo = data.get("imginfo", {})
    img_w = imginfo.get("img_width", 0)
    img_h = imginfo.get("img_height", 0)

    lines: list[OcrLine] = []
    all_texts: list[str] = []
    for block in data.get("contents", []):
        for item in block:
            text = item.get("text", "")
            if not text:
                continue
            all_texts.append(text)

            # boundingBox: [[x1,y1], [x1,y2], [x2,y1], [x2,y2]]
            bb = item.get("boundingBox", [])
            if len(bb) == 4:
                x = bb[0][0]
                y = bb[0][1]
                w = bb[2][0] - bb[0][0]
                h = bb[1][1] - bb[0][1]
                lines.append(OcrLine(text=text, bbox=(x, y, w, h)))

    return OcrPage(
        text="\n".join(all_texts),
        lines=lines,
        image_width=img_w,
        image_height=img_h,
        image_path=image_path,
    )


def ocr_image_with_layout(image_path: str | Path) -> OcrPage:
    """Run NDLOCR-Lite on an image and return structured result with bounding boxes.

    Falls back to text-only OcrPage if JSON output is not available.
    """
    if not is_ndlocr_available():
        raise RuntimeError(
            f"ndlocr-lite not found at {NDLOCR_DIR}. "
            f"Run the installer or: git clone https://github.com/ndl-lab/ndlocr-lite.git"
        )

    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(tempfile.mkdtemp(prefix="ocr_corrector_"))

    cmd = [
        sys.executable,
        str(NDLOCR_SCRIPT),
        "--sourceimg", str(image_path),
        "--output", str(output_dir),
    ]
    logger.info("Running NDLOCR-Lite: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(NDLOCR_SRC),
        )
        if result.returncode != 0:
            raise RuntimeError(f"NDLOCR-Lite failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("NDLOCR-Lite timed out (120s)")

    # Prefer JSON (has bounding boxes)
    json_files = sorted(output_dir.glob("*.json"))
    if json_files:
        return _parse_ndlocr_json(json_files[0], str(image_path))

    # Fallback: text only
    txt_files = sorted(output_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No OCR output found in {output_dir}")

    text = "\n".join(tf.read_text(encoding="utf-8") for tf in txt_files)
    return OcrPage(text=text, image_path=str(image_path))
