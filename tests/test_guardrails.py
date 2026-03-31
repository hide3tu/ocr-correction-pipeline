from __future__ import annotations

from pathlib import Path

from ocr_corrector.bert_scanner import SuspectToken
from ocr_corrector.escalation import (
    CorrectionResult,
    Verdict,
    classify_guarded_candidate,
    classify_with_qwen,
)
from ocr_corrector.pipeline import (
    _detect_structure_issues,
    _find_protected_term_candidates,
    _guard_candidate,
    _resplit_by_punctuation,
)
from ocr_corrector.qwen_judge import (
    JudgeResult,
    _parse_judge_response,
    _parse_semantic_check_response,
)
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


def test_parse_semantic_check_response():
    result = _parse_semantic_check_response(
        '{"verdict":"DIFF","reason":"場所が変わっている"}'
    )
    assert result.verdict == "DIFF"
    assert result.reason == "場所が変わっている"


def test_guard_candidate_blocks_punctuation():
    suspect = _suspect("、", "。")
    assert _guard_candidate(suspect) == ("punctuation", "句読点候補は自動修正しない")


def test_guard_candidate_blocks_paraphrase():
    suspect = _suspect("通販", "ネット通販")
    assert _guard_candidate(suspect) == ("paraphrase", "語彙の追加・言い換え候補")


def test_guard_candidate_blocks_single_kanji_semantic_replacement():
    suspect = _suspect("裏", "中", prob=0.31)
    assert _guard_candidate(suspect) == ("semantic", "単漢字の内容語置換候補")


def test_guard_candidate_blocks_large_semantic_replacement():
    suspect = _suspect("僕", "相手", prob=0.59)
    assert _guard_candidate(suspect) == ("semantic", "意味差が大きい内容語置換候補")


def test_guard_candidate_blocks_dialogue_dialect_in_fiction_mode():
    suspect = _suspect("とる", "てる", prob=0.51)
    assert _guard_candidate(
        suspect,
        mode="fiction",
        line="「ちゃんとやっとる?」",
    ) == ("dialect", "会話文・口語・方言候補")


def test_guard_candidate_blocks_dialect_without_quotes_in_fiction_mode():
    suspect = _suspect("とる", "てる", prob=0.51)
    assert _guard_candidate(
        suspect,
        mode="fiction",
        line="ちゃんとやっとる?",
    ) == ("dialect", "会話文・口語・方言候補")


def test_guard_candidate_keeps_general_mode_open_for_same_candidate():
    suspect = _suspect("とる", "てる", prob=0.51)
    assert _guard_candidate(
        suspect,
        mode="general",
        line="「ちゃんとやっとる?」",
    ) is None


def test_guard_candidate_blocks_registered_protected_term():
    suspect = _suspect("魔", "「", prob=0.73)
    assert _guard_candidate(
        suspect,
        line="魔美は帳場までやってきて、",
        protected_terms=("魔美",),
    ) == ("proper_noun", "保護語句に含まれるため自動修正しない")


def test_find_protected_term_candidates_surfaces_near_match():
    candidates = _find_protected_term_candidates(
        ["阿賀のは店先にいた。"],
        ("阿賀野",),
    )
    assert len(candidates) == 1
    assert candidates[0].original == "阿賀の"
    assert candidates[0].candidates[0][0] == "阿賀野"


def test_find_protected_term_candidates_skips_exact_match():
    candidates = _find_protected_term_candidates(
        ["阿賀野は店先にいた。"],
        ("阿賀野",),
    )
    assert candidates == []


def test_resplit_by_punctuation_splits_dialogue_at_question_mark():
    text = "「ちゃんとやっとる?\n店の裏から和装に着替えた鷹美が出てきて、もう昼過ぎだと気づいた。"
    assert _resplit_by_punctuation(text).splitlines()[0] == "「ちゃんとやっとる?"


def test_detect_structure_issue_for_missing_closing_quote():
    issues = _detect_structure_issues(["「ちゃんとやっとる?"])
    assert issues[0] == "会話の開き括弧に対する閉じ括弧欠落の可能性があるため語彙修正を停止"


def test_detect_structure_issue_allows_quote_to_close_on_next_line():
    issues = _detect_structure_issues([
        "「珍しく通販の注文が来てたから、",
        "処理したよ。あとは発送するだけ」",
    ])
    assert issues == [None, None]


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


def test_generate_downloads_splits_actionable_and_debug_csv():
    keep = CorrectionResult(
        suspect=_suspect("とる", "てる", prob=0.51),
        verdict=Verdict.AUTO_KEEP,
        suggested_fix="てる",
        suggested_prob=0.51,
        qwen_verdict="KEEP",
        category="dialect",
        reason="会話文・口語・方言候補",
    )
    escalate = CorrectionResult(
        suspect=_suspect("誤", "正", prob=0.91),
        verdict=Verdict.ESCALATE,
        suggested_fix="正",
        suggested_prob=0.91,
        qwen_verdict="KEEP",
        category="ocr_typo",
        reason="確認が必要",
    )
    files = generate_downloads(
        original_text="誤とる",
        ocr_text=None,
        resplit_lines=["誤とる"],
        corrections=[keep, escalate],
        llm_enabled=True,
        autofix_threshold=0.7,
        pages=None,
    )

    csv_text = Path(next(p for p in files if p.endswith("corrections.csv"))).read_text(encoding="utf-8")
    debug_text = Path(next(p for p in files if p.endswith("corrections_debug.csv"))).read_text(encoding="utf-8")

    assert "AUTO-KEEP" not in csv_text
    assert "会話文・口語・方言候補" not in csv_text
    assert "ESCALATE" in csv_text
    assert "AUTO-KEEP" in debug_text
    assert "ESCALATE" in debug_text
