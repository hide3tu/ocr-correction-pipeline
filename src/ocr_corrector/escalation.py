"""Escalation logic: combine BERT detection with Qwen judgment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .bert_scanner import SuspectToken
from .qwen_judge import JudgeResult

SAFE_CATEGORIES = {"ocr_typo", "grammar"}
PROTECTED_CATEGORIES = {"punctuation", "semantic", "structure", "proper_noun", "dialect", "paraphrase", "unclear"}
KEEP_ESCALATE_CATEGORIES = {
    "general": {"ocr_typo", "grammar", "semantic", "structure", "proper_noun", "dialect", "unclear"},
    "fiction": {"ocr_typo", "grammar", "semantic", "structure"},
}


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
    category: str | None = None
    reason: str | None = None


def classify_guarded_candidate(
    suspect: SuspectToken,
    category: str,
    reason: str,
) -> CorrectionResult:
    """Return an AUTO-KEEP result for candidates blocked by local guardrails."""
    top_candidate, top_prob = suspect.candidates[0]

    return CorrectionResult(
        suspect=suspect,
        verdict=Verdict.AUTO_KEEP,
        suggested_fix=top_candidate,
        suggested_prob=top_prob,
        qwen_verdict="KEEP",
        category=category,
        reason=reason,
    )


def classify_manual_candidate(
    suspect: SuspectToken,
    verdict: Verdict,
    category: str,
    reason: str,
    qwen_verdict: str | None = None,
) -> CorrectionResult:
    """Return a pre-classified correction result for non-BERT rules."""
    top_candidate, top_prob = suspect.candidates[0]

    return CorrectionResult(
        suspect=suspect,
        verdict=verdict,
        suggested_fix=top_candidate,
        suggested_prob=top_prob,
        qwen_verdict=qwen_verdict,
        category=category,
        reason=reason,
    )


def _coerce_judge_result(value: JudgeResult | str) -> JudgeResult:
    if isinstance(value, JudgeResult):
        return value
    verdict = "FIX" if str(value).upper() == "FIX" else "KEEP"
    return JudgeResult(verdict=verdict)


def classify_with_qwen(
    suspect: SuspectToken,
    qwen_verdict: JudgeResult | str,
    escalation_threshold: float = 0.50,
    mode: str = "general",
) -> CorrectionResult:
    """Classify a suspect token when Qwen is enabled."""
    top_candidate, top_prob = suspect.candidates[0]
    judge = _coerce_judge_result(qwen_verdict)
    category = judge.category or "unclear"
    reason = judge.reason or None
    keep_escalate = KEEP_ESCALATE_CATEGORIES.get(mode, KEEP_ESCALATE_CATEGORIES["general"])

    if judge.verdict == "FIX" and category in PROTECTED_CATEGORIES:
        verdict = Verdict.ESCALATE
    elif judge.verdict == "FIX":
        verdict = Verdict.AUTO_FIX
    elif top_prob >= escalation_threshold and category in keep_escalate:
        verdict = Verdict.ESCALATE
    else:
        verdict = Verdict.AUTO_KEEP

    return CorrectionResult(
        suspect=suspect,
        verdict=verdict,
        suggested_fix=top_candidate,
        suggested_prob=top_prob,
        qwen_verdict=judge.verdict,
        category=category,
        reason=reason,
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
        category=None,
        reason=None,
    )
