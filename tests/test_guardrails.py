from __future__ import annotations

from pathlib import Path

from ocr_corrector.bert_scanner import SuspectToken
from ocr_corrector.escalation import (
    CorrectionResult,
    Verdict,
    classify_guarded_candidate,
    classify_with_qwen,
)
from ocr_corrector.pipeline import _guard_candidate
from ocr_corrector.qwen_judge import JudgeResult, _parse_judge_response
from ocr_corrector.text_export import generate_downloads


def _suspect(original: str, fixed: str, prob: float = 0.9) -> SuspectToken:
    return SuspectToken(
        position=1,
        original=original,
        probability=0.001,
        candidates=[(fixed, prob)],
        line_index=0,
    )


def test_parse_qwen_json_response():
    result = _parse_judge_response(
        '{"verdict":"KEEP","category":"proper_noun","reason":"固有名詞の可能性"}'
    )
    assert result.verdict == "KEEP"
    assert result.category == "proper_noun"
    assert result.reason == "固有名詞の可能性"


def test_guard_candidate_blocks_punctuation():
    suspect = _suspect("、", "。")
    assert _guard_candidate(suspect) == ("punctuation", "句読点候補は自動修正しない")


def test_guard_candidate_blocks_paraphrase():
    suspect = _suspect("通販", "ネット通販")
    assert _guard_candidate(suspect) == ("paraphrase", "語彙の追加・言い換え候補")


def test_classify_with_qwen_keeps_protected_category_out_of_escalation():
    suspect = _suspect("阿賀野", "阿賀の", prob=0.98)
    result = classify_with_qwen(
        suspect,
        JudgeResult("KEEP", "proper_noun", "固有名詞の可能性"),
        escalation_threshold=0.5,
        mode="fiction",
    )
    assert result.verdict == Verdict.AUTO_KEEP


def test_classify_with_qwen_general_mode_escalates_proper_noun_keep():
    suspect = _suspect("阿賀野", "阿賀の", prob=0.98)
    result = classify_with_qwen(
        suspect,
        JudgeResult("KEEP", "proper_noun", "固有名詞の可能性"),
        escalation_threshold=0.5,
        mode="general",
    )
    assert result.verdict == Verdict.ESCALATE


def test_classify_with_qwen_escalates_sensitive_fix():
    suspect = _suspect("阿賀の", "阿賀野", prob=0.98)
    result = classify_with_qwen(
        suspect,
        JudgeResult("FIX", "proper_noun", "固有名詞の誤認識"),
        escalation_threshold=0.5,
        mode="general",
    )
    assert result.verdict == Verdict.ESCALATE


def test_classify_guarded_candidate_marks_auto_keep():
    suspect = _suspect("通販", "ネット通販")
    result = classify_guarded_candidate(suspect, "paraphrase", "語彙の追加・言い換え候補")
    assert result.verdict == Verdict.AUTO_KEEP
    assert result.qwen_verdict == "KEEP"


def test_generate_downloads_skips_escalated_sensitive_fix_in_llm_outputs():
    correction = CorrectionResult(
        suspect=_suspect("阿賀の", "阿賀野", prob=0.98),
        verdict=Verdict.ESCALATE,
        suggested_fix="阿賀野",
        suggested_prob=0.98,
        qwen_verdict="FIX",
        category="proper_noun",
        reason="固有名詞の誤認識",
    )
    files = generate_downloads(
        original_text="阿賀の",
        ocr_text=None,
        resplit_lines=["阿賀の"],
        corrections=[correction],
        llm_enabled=True,
        autofix_threshold=0.7,
        pages=None,
    )

    llm_text = Path(next(p for p in files if p.endswith("corrected_llm.txt"))).read_text(encoding="utf-8")
    all_text = Path(next(p for p in files if p.endswith("corrected_all.txt"))).read_text(encoding="utf-8")
    bert_text = Path(next(p for p in files if p.endswith("corrected_bert.txt"))).read_text(encoding="utf-8")

    assert llm_text == "阿賀の"
    assert all_text == "阿賀の"
    assert bert_text == "阿賀野"
