"""Main pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .bert_scanner import BertScanner, SuspectToken
from .config import PipelineConfig
from .escalation import (
    CorrectionResult,
    classify_with_qwen,
    classify_without_qwen,
)
from .gpu_detect import resolve_device
from .qwen_judge import QwenJudge

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result of running the pipeline on a text."""

    corrections: list[CorrectionResult]
    raw_suspects: int  # before filtering
    filtered_suspects: int  # after filtering
    timing: dict[str, float] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


def _filter_suspects(
    suspects: list[SuspectToken],
    min_prob: float = 0.30,
    skip_subword: bool = True,
) -> list[SuspectToken]:
    """Filter out false positives from BERT detection."""
    filtered = []
    for s in suspects:
        if not s.candidates:
            continue

        top_text, top_prob = s.candidates[0]

        # Skip subword tokens (##prefix)
        if skip_subword and (top_text.startswith("##") or s.original.startswith("##")):
            continue

        # Skip if BERT's top candidate has low probability
        if top_prob < min_prob:
            continue

        filtered.append(s)

    return filtered


def _build_fixed_line(line: str, suspect: SuspectToken, fix: str) -> str:
    """Replace the suspect token in the line with the fix.

    This is an approximation: we search for the original token text in the line
    and replace the first occurrence. Not perfect for ambiguous positions, but
    sufficient for A/B comparison.
    """
    if suspect.original in line:
        return line.replace(suspect.original, fix, 1)
    return line


def _get_context(lines: list[str], line_idx: int, window: int = 2) -> str:
    """Get surrounding lines as context for Qwen."""
    start = max(0, line_idx - window)
    end = min(len(lines), line_idx + window + 1)
    return "\n".join(lines[start:end])


class Pipeline:
    """OCR correction pipeline: BERT scan -> filter -> Qwen judge -> escalate."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._scanner: BertScanner | None = None
        self._judge: QwenJudge | None = None
        self._bert_device: str = "cpu"
        self._ollama_num_gpu: str = "0"

    def setup(self):
        """Initialize models based on config."""
        self._bert_device, self._ollama_num_gpu = resolve_device(self.config.gpu_mode)
        logger.info(
            "Device config: BERT=%s, ollama GPU layers=%s",
            self._bert_device, self._ollama_num_gpu,
        )

        # Load BERT
        self._scanner = BertScanner(
            model_name=self.config.bert_model,
            device=self._bert_device,
            threshold=self.config.bert_threshold,
        )

        # Load Qwen (if enabled)
        if self.config.qwen_enabled:
            self._judge = QwenJudge(
                model=self.config.qwen_model,
                ollama_num_gpu=self._ollama_num_gpu,
            )

    def run(self, text: str) -> PipelineResult:
        """Run the full pipeline on input text."""
        if self._scanner is None:
            self.setup()

        lines = text.splitlines()
        timing: dict[str, float] = {}

        # Step 1: BERT scan
        t0 = time.perf_counter()
        raw_suspects = self._scanner.scan(text)
        timing["bert_scan"] = time.perf_counter() - t0
        logger.info("BERT scan: %d suspects found in %.2fs", len(raw_suspects), timing["bert_scan"])

        # Step 2: Filter
        t0 = time.perf_counter()
        filtered = _filter_suspects(
            raw_suspects,
            min_prob=self.config.min_candidate_prob,
            skip_subword=self.config.skip_subword,
        )
        timing["filter"] = time.perf_counter() - t0
        logger.info("Filtered: %d -> %d suspects", len(raw_suspects), len(filtered))

        # Step 3: Classify each suspect
        corrections: list[CorrectionResult] = []
        t0 = time.perf_counter()

        for suspect in filtered:
            if self.config.qwen_enabled and self._judge is not None:
                # Build A/B lines for Qwen
                line = lines[suspect.line_index] if suspect.line_index < len(lines) else ""
                top_fix = suspect.candidates[0][0] if suspect.candidates else ""
                fixed_line = _build_fixed_line(line, suspect, top_fix)
                context = _get_context(lines, suspect.line_index)

                qwen_verdict = self._judge.judge(line, fixed_line, context)
                result = classify_with_qwen(
                    suspect, qwen_verdict,
                    escalation_threshold=self.config.escalation_threshold,
                )
            else:
                result = classify_without_qwen(
                    suspect,
                    autofix_threshold=self.config.autofix_threshold,
                    escalation_threshold=self.config.min_candidate_prob,
                )
            corrections.append(result)

        timing["qwen_judge"] = time.perf_counter() - t0
        if self.config.qwen_enabled:
            logger.info("Qwen judged %d suspects in %.2fs", len(filtered), timing["qwen_judge"])

        return PipelineResult(
            corrections=corrections,
            raw_suspects=len(raw_suspects),
            filtered_suspects=len(filtered),
            timing=timing,
            lines=lines,
        )

    def cleanup(self):
        """Release resources."""
        if self._scanner:
            self._scanner.unload()
            self._scanner = None
        if self._judge:
            self._judge.cleanup()
            self._judge = None
