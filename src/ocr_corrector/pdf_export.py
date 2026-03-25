"""Generate searchable PDF with transparent text overlay on images."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Font search paths (project-bundled first, then system fallbacks)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FONT_CANDIDATES = [
    PROJECT_ROOT / "fonts" / "ipaexg.ttf",
    # Windows
    Path("C:/Windows/Fonts/msgothic.ttc"),
    # macOS
    Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
    # Linux
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/noto-cjk/NotoSansCJKjp-Regular.otf"),
]


def find_cjk_font() -> Path | None:
    """Find a usable CJK font file."""
    for p in _FONT_CANDIDATES:
        if p.exists():
            return p
    return None


def generate_searchable_pdf(
    pages,
    corrected_texts: list[str],
    font_path: Path | None = None,
) -> str | None:
    """Generate a searchable PDF with images and invisible text overlay.

    Args:
        pages: List of OcrPage objects (from ocr_frontend).
        corrected_texts: Corrected text for each page (same order as pages).
        font_path: Path to a CJK TrueType font. Auto-detected if None.

    Returns:
        Path to the generated PDF file, or None if generation failed.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed — skipping PDF generation")
        return None

    if font_path is None:
        font_path = find_cjk_font()
    if font_path is None:
        logger.warning("No CJK font found — skipping PDF generation")
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)

    # Register font (fpdf2 auto-subsets to only used glyphs)
    pdf.add_font("cjk", fname=str(font_path))

    for page, corrected in zip(pages, corrected_texts):
        img_path = Path(page.image_path)
        if not img_path.exists():
            logger.warning("Image not found, skipping page: %s", img_path)
            continue

        img_w = page.image_width
        img_h = page.image_height
        if img_w == 0 or img_h == 0:
            continue

        # Determine PDF page size to match image aspect ratio
        # Use A4 width (210mm) as reference, scale height proportionally
        pdf_w = 210.0
        pdf_h = pdf_w * (img_h / img_w)
        pdf.add_page(format=(pdf_w, pdf_h))

        # Place image as full-page background
        pdf.image(str(img_path), x=0, y=0, w=pdf_w, h=pdf_h)

        # Overlay invisible text at bounding box positions
        from fpdf.enums import TextMode

        pdf.set_font("cjk", size=10)
        pdf.text_mode = TextMode.INVISIBLE

        # Split corrected text into lines matching OCR output
        corrected_lines = corrected.split("\n")

        for i, ocr_line in enumerate(page.lines):
            # Use corrected text if available, otherwise original
            text = corrected_lines[i] if i < len(corrected_lines) else ocr_line.text

            bx, by, bw, bh = ocr_line.bbox

            # Convert image pixel coords to PDF mm coords
            x_mm = (bx / img_w) * pdf_w
            y_mm = (by / img_h) * pdf_h
            w_mm = (bw / img_w) * pdf_w
            h_mm = (bh / img_h) * pdf_h

            # Scale font size to fit the bounding box height
            font_size = h_mm * 0.72  # mm to pt approximation
            if font_size < 4:
                font_size = 4
            if font_size > 72:
                font_size = 72
            pdf.set_font_size(font_size)

            pdf.set_xy(x_mm, y_mm)
            pdf.cell(w=w_mm, h=h_mm, text=text)

    pdf.text_mode = TextMode.FILL

    out_path = Path(tempfile.mkdtemp(prefix="ocr_pdf_")) / "searchable.pdf"
    pdf.output(str(out_path))
    return str(out_path)
