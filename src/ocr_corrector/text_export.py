"""Export correction results as downloadable text and CSV files."""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from typing import Callable

from .escalation import CorrectionResult


def _replace_in_line(line: str, original: str, replacement: str) -> str:
    """Replace the first occurrence of a token in a line."""
    if original in line:
        return line.replace(original, replacement, 1)
    return line


def _apply_corrections_to_lines(
    resplit_lines: list[str],
    corrections: list[CorrectionResult],
    filter_fn: Callable[[CorrectionResult], bool],
) -> list[str]:
    """Apply selected corrections to the re-split lines."""
    corrected = list(resplit_lines)
    for c in corrections:
        if not filter_fn(c):
            continue
        idx = c.suspect.line_index
        if idx < len(corrected):
            corrected[idx] = _replace_in_line(
                corrected[idx], c.suspect.original, c.suggested_fix
            )
    return corrected


def _map_to_original_breaks(
    original_text: str, corrected_resplit_lines: list[str]
) -> str:
    """Map corrected re-split text back to original line break positions.

    The pipeline re-splits text by Japanese punctuation, changing line
    boundaries.  This function restores the original newline positions after
    corrections have been applied to the re-split lines.
    """
    original_flat = original_text.replace("\r", "").replace("\n", "")
    corrected_flat = "".join(corrected_resplit_lines)

    if len(original_flat) == len(corrected_flat):
        # Same length — direct character mapping preserving original newlines
        result: list[str] = []
        flat_pos = 0
        for ch in original_text:
            if ch in ("\r", "\n"):
                result.append(ch)
            else:
                if flat_pos < len(corrected_flat):
                    result.append(corrected_flat[flat_pos])
                flat_pos += 1
        return "".join(result)

    # Length changed — fall back to re-split line structure
    return "\n".join(corrected_resplit_lines)


def apply_corrections(
    original_text: str,
    resplit_lines: list[str],
    corrections: list[CorrectionResult],
    filter_fn: Callable[[CorrectionResult], bool],
) -> str:
    """Apply selected corrections, preserving original line breaks."""
    corrected = _apply_corrections_to_lines(resplit_lines, corrections, filter_fn)
    return _map_to_original_breaks(original_text, corrected)


def build_csv(corrections: list[CorrectionResult], lines: list[str]) -> str:
    """Build CSV content from correction results."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["行", "元", "修正候補", "BERT確率", "LLM判定", "最終判定", "行テキスト"])
    for c in corrections:
        line_text = ""
        if c.suspect.line_index < len(lines):
            line_text = lines[c.suspect.line_index].strip()
        writer.writerow([
            c.suspect.line_index,
            c.suspect.original,
            c.suggested_fix,
            f"{c.suggested_prob:.0%}",
            c.qwen_verdict or "-",
            c.verdict.value,
            line_text,
        ])
    return buf.getvalue()


def generate_downloads(
    original_text: str,
    ocr_text: str | None,
    resplit_lines: list[str],
    corrections: list[CorrectionResult],
    llm_enabled: bool,
    autofix_threshold: float = 0.70,
) -> list[str]:
    """Generate download files and return list of file paths.

    Files generated:
      - ocr_raw.txt          : Raw OCR output (only if image input)
      - corrections.csv      : Correction results table
      - corrected_bert.txt   : BERT auto-fix applied (prob >= autofix_threshold)
      - corrected_llm.txt    : LLM-approved corrections only (if LLM enabled)
      - corrected_all.txt    : BERT OR LLM corrections (if LLM enabled)
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="ocr_correction_"))
    files: list[str] = []

    # 1. OCR raw text (only for image input)
    if ocr_text:
        p = tmpdir / "ocr_raw.txt"
        p.write_text(ocr_text, encoding="utf-8")
        files.append(str(p))

    # 2. Correction results CSV (BOM for Excel compatibility)
    csv_content = build_csv(corrections, resplit_lines)
    p = tmpdir / "corrections.csv"
    p.write_text("\ufeff" + csv_content, encoding="utf-8")
    files.append(str(p))

    # 3. BERT auto-fix: apply where BERT confidence is high on its own
    bert_text = apply_corrections(
        original_text, resplit_lines, corrections,
        filter_fn=lambda c: c.suggested_prob >= autofix_threshold,
    )
    p = tmpdir / "corrected_bert.txt"
    p.write_text(bert_text, encoding="utf-8")
    files.append(str(p))

    if llm_enabled:
        # 4. LLM-approved corrections only
        llm_text = apply_corrections(
            original_text, resplit_lines, corrections,
            filter_fn=lambda c: c.qwen_verdict == "FIX",
        )
        p = tmpdir / "corrected_llm.txt"
        p.write_text(llm_text, encoding="utf-8")
        files.append(str(p))

        # 5. All corrections: BERT auto-fix OR LLM FIX
        all_text = apply_corrections(
            original_text, resplit_lines, corrections,
            filter_fn=lambda c: (
                c.suggested_prob >= autofix_threshold or c.qwen_verdict == "FIX"
            ),
        )
        p = tmpdir / "corrected_all.txt"
        p.write_text(all_text, encoding="utf-8")
        files.append(str(p))

    return files
