"""Escalation logic: combine BERT detection with Qwen judgment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .bert_scanner import SuspectToken


class Verdict(str, Enum):
    AUTO_FIX = "AUTO-FIX"
    ESCALATE = "ESCALATE"
    AUTO_KEEP = "AUTO-KEEP"


@dataclass
class CorrectionResult:
    """Result of the correction pipeline for a single suspect."""

    suspect: SuspectToken
    verdict: Verdict
    suggested_fix: str  # top1 candidate from BERT
    suggested_prob: float
    qwen_verdict: str | None  # "FIX", "KEEP", or None if Qwen disabled


def classify_with_qwen(
    suspect: SuspectToken,
    qwen_verdict: str,
    escalation_threshold: float = 0.50,
) -> CorrectionResult:
    """Classify a suspect token when Qwen is enabled."""
    top_candidate, top_prob = suspect.candidates[0]

    if qwen_verdict == "FIX":
        verdict = Verdict.AUTO_FIX
    elif top_prob >= escalation_threshold:
        verdict = Verdict.ESCALATE
    else:
        verdict = Verdict.AUTO_KEEP

    return CorrectionResult(
        suspect=suspect,
        verdict=verdict,
        suggested_fix=top_candidate,
        suggested_prob=top_prob,
        qwen_verdict=qwen_verdict,
    )


def classify_without_qwen(
    suspect: SuspectToken,
    autofix_threshold: float = 0.70,
    escalation_threshold: float = 0.30,
) -> CorrectionResult:
    """Classify a suspect token when Qwen is disabled (BERT-only mode)."""
    top_candidate, top_prob = suspect.candidates[0]

    if top_prob >= autofix_threshold:
        verdict = Verdict.AUTO_FIX
    elif top_prob >= escalation_threshold:
        verdict = Verdict.ESCALATE
    else:
        verdict = Verdict.AUTO_KEEP

    return CorrectionResult(
        suspect=suspect,
        verdict=verdict,
        suggested_fix=top_candidate,
        suggested_prob=top_prob,
        qwen_verdict=None,
    )
