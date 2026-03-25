"""Export correction results as downloadable text and CSV files."""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from typing import Callable

from .escalation import CorrectionResult


def apply_corrections(
    original_text: str,
    resplit_lines: list[str],
    corrections: list[CorrectionResult],
    filter_fn: Callable[[CorrectionResult], bool],
) -> str:
    """Apply selected corrections directly to the original text.

    Newlines in ``original_text`` are always preserved because replacements
    are applied via a flat-text position mapping that skips newline characters.
    """
    # Compute the start offset of each re-split line in the flat text
    line_starts: list[int] = []
    offset = 0
    for line in resplit_lines:
        line_starts.append(offset)
        offset += len(line)

    # Build mapping: flat-text position -> index in original_text (skips \r\n)
    flat_to_orig: list[int] = []
    for i, ch in enumerate(original_text):
        if ch not in ("\r", "\n"):
            flat_to_orig.append(i)

    # Collect (flat_pos, old_token, new_token) for each selected correction
    replacements: list[tuple[int, str, str]] = []
    for c in corrections:
        if not filter_fn(c):
            continue
        idx = c.suspect.line_index
        if idx >= len(resplit_lines):
            continue
        token_pos = resplit_lines[idx].find(c.suspect.original)
        if token_pos < 0:
            continue
        flat_pos = line_starts[idx] + token_pos
        replacements.append((flat_pos, c.suspect.original, c.suggested_fix))

    if not replacements:
        return original_text

    # Apply from end to start so earlier positions stay valid
    replacements.sort(key=lambda r: r[0], reverse=True)
    chars = list(original_text)
    for flat_pos, old_token, new_token in replacements:
        end_flat = flat_pos + len(old_token) - 1
        if end_flat >= len(flat_to_orig):
            continue
        orig_start = flat_to_orig[flat_pos]
        orig_end = flat_to_orig[end_flat] + 1
        chars[orig_start:orig_end] = list(new_token)

    return "".join(chars)


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
