"""Main pipeline orchestrator."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .bert_scanner import BertScanner, SuspectToken
from .config import PipelineConfig
from .escalation import (
    CorrectionResult,
    classify_guarded_candidate,
    classify_with_qwen,
    classify_without_qwen,
)
from .gpu_detect import resolve_device
from .llm_server import LlmServerProcess, find_model, find_server_bin, is_server_running
from .qwen_judge import LlmJudge

logger = logging.getLogger(__name__)
PUNCTUATION_CHARS = set("、。,.，．！？!?：:；;・…‥ー―-「」『』（）()［］【】〔〕〈〉《》")
SENTENCE_ENDERS = set("。！？!?")
SOFT_ENDERS = set("、")
CLOSING_QUOTES = set("」』）)]】〉》")
OPEN_TO_CLOSE = {
    "「": "」",
    "『": "』",
    "(": ")",
    "（": "）",
    "[": "]",
    "［": "］",
    "【": "】",
    "〔": "〕",
    "〈": "〉",
    "《": "》",
}
CLOSE_TO_OPEN = {close: open_ for open_, close in OPEN_TO_CLOSE.items()}
DIALECT_MARKERS = (
    "やっとる",
    "しとる",
    "しとく",
    "やん",
    "やね",
    "やろ",
    "やで",
    "やし",
    "かて",
    "へん",
    "ねん",
    "もんや",
)
DIALECT_ENDINGS = ("とる", "とく", "やん", "やね", "やろ", "やで", "かて", "へん", "ねん")


@dataclass
class PipelineResult:
    """Complete result of running the pipeline on a text."""

    corrections: list[CorrectionResult]
    raw_suspects: int  # before filtering
    filtered_suspects: int  # after filtering
    timing: dict[str, float] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


def _resplit_by_punctuation(text: str) -> str:
    """Join all lines and re-split by Japanese punctuation.

    OCR output splits text at physical line boundaries, which breaks words
    across lines (e.g. "空\\n気" for "空気"). Re-splitting by punctuation
    (。、！？!?) eliminates these mid-word breaks while keeping sentence
    boundaries around dialogue.
    """
    joined = text.replace("\n", "")
    sentences: list[str] = []
    buf: list[str] = []
    i = 0

    while i < len(joined):
        ch = joined[i]
        buf.append(ch)
        i += 1

        if ch in SENTENCE_ENDERS:
            while i < len(joined) and joined[i] in CLOSING_QUOTES:
                buf.append(joined[i])
                i += 1
            sentence = "".join(buf).strip()
            if sentence:
                sentences.append(sentence)
            buf = []
        elif ch in SOFT_ENDERS:
            sentence = "".join(buf).strip()
            if sentence:
                sentences.append(sentence)
            buf = []

    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)

    return "\n".join(sentences)


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


def _detect_structure_issues(lines: list[str]) -> list[str | None]:
    """Detect bracket/quote mismatches across the whole text.

    Quotes often open on one line and close on the next, so line-local detection
    creates false positives. This scanner keeps a global stack and only flags
    lines that truly introduce unmatched closing brackets, plus lines where an
    opening bracket is still unmatched at end of text.
    """
    issues: list[str | None] = [None] * len(lines)
    stack: list[tuple[str, int]] = []

    for line_idx, line in enumerate(lines):
        for ch in line:
            if ch in OPEN_TO_CLOSE:
                stack.append((ch, line_idx))
            elif ch in CLOSE_TO_OPEN:
                expected = CLOSE_TO_OPEN[ch]
                if stack and stack[-1][0] == expected:
                    stack.pop()
                else:
                    issues[line_idx] = "閉じ括弧の対応が崩れているため語彙修正を停止"

    for open_ch, line_idx in stack:
        if issues[line_idx] is not None:
            continue
        if open_ch in {"「", "『"}:
            issues[line_idx] = "会話の開き括弧に対する閉じ括弧欠落の可能性があるため語彙修正を停止"
        else:
            issues[line_idx] = "括弧の対応が崩れているため語彙修正を停止"

    return issues


def _is_punctuation_token(token: str) -> bool:
    stripped = token.strip()
    return bool(stripped) and all(ch in PUNCTUATION_CHARS for ch in stripped)


def _is_hiragana_word(token: str) -> bool:
    return bool(token) and bool(re.fullmatch(r"[ぁ-んー]+", token))


def _is_kanji_word(token: str) -> bool:
    return bool(token) and bool(re.fullmatch(r"[一-龠々ヶ]+", token))


def _looks_like_dialogue(line: str) -> bool:
    return any(marker in line for marker in ("「", "」", "『", "』", "?", "？", "!", "！"))


def _looks_like_dialect_context(line: str, original: str, fixed: str) -> bool:
    compact_line = "".join(line.split())
    if any(marker in compact_line for marker in DIALECT_MARKERS):
        return True
    return any(
        token.endswith(ending)
        for token in (original, fixed)
        for ending in DIALECT_ENDINGS
    )


def _normalize_protected_terms(terms: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not terms:
        return ()
    return tuple(dict.fromkeys(term.strip() for term in terms if term and term.strip()))


def _find_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not text or not needle:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            return spans
        spans.append((idx, idx + len(needle)))
        start = idx + 1


def _overlaps(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


def _touches_protected_term(
    line: str,
    original: str,
    protected_terms: tuple[str, ...],
) -> bool:
    if not line or not original or not protected_terms:
        return False
    if original in protected_terms:
        return True

    protected_spans: list[tuple[int, int]] = []
    for term in protected_terms:
        protected_spans.extend(_find_occurrences(line, term))
    if not protected_spans:
        return False

    original_spans = _find_occurrences(line, original)
    if not original_spans:
        return False

    overlapping = [
        span for span in original_spans
        if any(_overlaps(span, protected_span) for protected_span in protected_spans)
    ]
    return bool(overlapping) and len(overlapping) == len(original_spans)


def _guard_candidate(
    suspect: SuspectToken,
    mode: str = "general",
    line: str = "",
    protected_terms: tuple[str, ...] = (),
) -> tuple[str, str] | None:
    """Block candidate types that frequently cause OCR over-correction."""
    if not suspect.candidates:
        return None

    top_fix, top_prob = suspect.candidates[0]
    original = suspect.original.strip()
    fixed = top_fix.strip()
    if not original or not fixed or original == fixed:
        return None

    if _touches_protected_term(line, original, protected_terms):
        return ("proper_noun", "保護語句に含まれるため自動修正しない")

    if _is_punctuation_token(original) or _is_punctuation_token(fixed):
        return ("punctuation", "句読点候補は自動修正しない")

    compact_original = "".join(original.split())
    compact_fixed = "".join(fixed.split())
    if (
        len(compact_fixed) >= len(compact_original) + 2
        and (
            compact_original in compact_fixed
            or compact_fixed.startswith(compact_original)
            or compact_fixed.endswith(compact_original)
        )
    ):
        return ("paraphrase", "語彙の追加・言い換え候補")

    shared_chars = set(compact_original) & set(compact_fixed)
    if (
        mode == "fiction"
        and (
            _looks_like_dialogue(line)
            or _looks_like_dialect_context(line, compact_original, compact_fixed)
        )
        and _is_hiragana_word(compact_original)
        and _is_hiragana_word(compact_fixed)
    ):
        return ("dialect", "会話文・口語・方言候補")

    if (
        _is_kanji_word(compact_original)
        and _is_kanji_word(compact_fixed)
        and len(compact_original) == 1
        and len(compact_fixed) == 1
        and not shared_chars
        and top_prob < 0.90
    ):
        return ("semantic", "単漢字の内容語置換候補")

    if (
        not shared_chars
        and (
            len(compact_original) >= 2
            or len(compact_fixed) >= 2
        )
    ):
        return ("semantic", "意味差が大きい内容語置換候補")

    if (
        mode == "fiction"
        and _is_hiragana_word(compact_original)
        and len(compact_original) >= 2
        and not shared_chars
    ):
        return ("dialect", "口語・かな語の置換候補")

    if (
        len(compact_original) >= 2
        and len(compact_fixed) >= 2
        and not shared_chars
        and abs(len(compact_fixed) - len(compact_original)) >= 1
    ):
        return ("unclear", "語形差が大きすぎる候補")

    return None


class Pipeline:
    """OCR correction pipeline: BERT scan -> filter -> Qwen judge -> escalate."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._scanner: BertScanner | None = None
        self._judge: LlmJudge | None = None
        self._server: LlmServerProcess | None = None
        self._bert_device: str = "cpu"

    def setup(self):
        """Initialize models based on config."""
        self._bert_device, self._llm_ngl = resolve_device(self.config.gpu_mode)
        logger.info("Device config: BERT=%s, LLM n_gpu_layers=%s", self._bert_device, self._llm_ngl)

        # Load BERT
        self._scanner = BertScanner(
            model_name=self.config.bert_model,
            device=self._bert_device,
            threshold=self.config.bert_threshold,
        )

        # Connect to LLM (if enabled)
        if self.config.llm_enabled:
            api_base = self.config.llm_api_base

            # Auto-start llama-server if no server is running
            if not is_server_running(api_base):
                server_bin = find_server_bin()
                model_path = find_model()
                if server_bin and model_path:
                    logger.info("No LLM server detected. Auto-starting llama-server...")
                    self._server = LlmServerProcess(
                        server_bin=server_bin,
                        model_path=model_path,
                        port=8080,
                        n_gpu_layers=int(self._llm_ngl),
                    )
                    api_base = self._server.start()
                else:
                    missing = []
                    if not server_bin:
                        missing.append("llama-server binary (run installer)")
                    if not model_path:
                        missing.append("GGUF model in llm/models/")
                    raise RuntimeError(
                        f"LLM server not running and cannot auto-start. "
                        f"Missing: {', '.join(missing)}"
                    )

            self._judge = LlmJudge(
                model=self.config.llm_model,
                api_base=api_base,
                mode=self.config.correction_mode,
                protected_terms=self.config.protected_terms,
            )

    def run(self, text: str) -> PipelineResult:
        """Run the full pipeline on input text."""
        if self._scanner is None:
            self.setup()

        # Pre-process: re-split by punctuation to eliminate line-break artifacts
        text = _resplit_by_punctuation(text)
        lines = text.splitlines()
        structure_issues = _detect_structure_issues(lines)
        protected_terms = _normalize_protected_terms(self.config.protected_terms)
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
            line = lines[suspect.line_index] if suspect.line_index < len(lines) else ""
            structure_issue = (
                structure_issues[suspect.line_index]
                if suspect.line_index < len(structure_issues)
                else None
            )
            if structure_issue is not None:
                result = classify_guarded_candidate(suspect, "structure", structure_issue)
            else:
                guard = _guard_candidate(
                    suspect,
                    mode=self.config.correction_mode,
                    line=line,
                    protected_terms=protected_terms,
                )
                if guard is not None:
                    category, reason = guard
                    result = classify_guarded_candidate(suspect, category, reason)
                elif self.config.llm_enabled and self._judge is not None:
                    # Build A/B lines for Qwen
                    top_fix = suspect.candidates[0][0] if suspect.candidates else ""
                    fixed_line = _build_fixed_line(line, suspect, top_fix)
                    context = _get_context(lines, suspect.line_index)

                    qwen_verdict = self._judge.judge_with_details(
                        original_line=line,
                        fixed_line=fixed_line,
                        original_token=suspect.original,
                        fixed_token=top_fix,
                        context=context,
                    )
                    if qwen_verdict.verdict == "FIX":
                        semantic_check = self._judge.semantic_check(
                            original_line=line,
                            fixed_line=fixed_line,
                            original_token=suspect.original,
                            fixed_token=top_fix,
                            context=context,
                        )
                        if semantic_check.verdict == "DIFF":
                            result = classify_guarded_candidate(
                                suspect,
                                "semantic",
                                semantic_check.reason or "意味保持チェックで差分あり",
                            )
                        else:
                            result = classify_with_qwen(
                                suspect, qwen_verdict,
                                escalation_threshold=self.config.escalation_threshold,
                                mode=self.config.correction_mode,
                            )
                    else:
                        result = classify_with_qwen(
                            suspect, qwen_verdict,
                            escalation_threshold=self.config.escalation_threshold,
                            mode=self.config.correction_mode,
                        )
                else:
                    result = classify_without_qwen(
                        suspect,
                        autofix_threshold=self.config.autofix_threshold,
                        escalation_threshold=self.config.min_candidate_prob,
                    )
            corrections.append(result)

        timing["qwen_judge"] = time.perf_counter() - t0
        if self.config.llm_enabled:
            logger.info("Qwen judged %d suspects in %.2fs", len(filtered), timing["qwen_judge"])

        return PipelineResult(
            corrections=corrections,
            raw_suspects=len(raw_suspects),
            filtered_suspects=len(filtered),
            timing=timing,
            lines=lines,
        )

    def run_steps(self, text: str, ocr_text: str | None = None):
        """Run pipeline yielding intermediate results at each stage.

        Yields (stage, data) tuples:
          ("ocr", ocr_text)
          ("bert", {"raw": N, "filtered": N, "suspects": [...], "lines": [...], "time": F})
          ("llm", {"index": i, "total": N, "result": CorrectionResult})
          ("done", PipelineResult)
        """
        if self._scanner is None:
            self.setup()

        # If ocr_text was passed (from image input), use it
        if ocr_text:
            text = ocr_text

        text = _resplit_by_punctuation(text)
        lines = text.splitlines()
        structure_issues = _detect_structure_issues(lines)
        protected_terms = _normalize_protected_terms(self.config.protected_terms)
        timing: dict[str, float] = {}

        # Step 1: BERT scan
        t0 = time.perf_counter()
        raw_suspects = self._scanner.scan(text)
        timing["bert_scan"] = time.perf_counter() - t0

        filtered = _filter_suspects(
            raw_suspects,
            min_prob=self.config.min_candidate_prob,
            skip_subword=self.config.skip_subword,
        )

        yield "bert", {
            "raw": len(raw_suspects),
            "filtered": len(filtered),
            "suspects": filtered,
            "lines": lines,
            "time": timing["bert_scan"],
        }

        # Step 2: LLM judgment (one by one)
        corrections: list[CorrectionResult] = []
        t0 = time.perf_counter()

        for i, suspect in enumerate(filtered):
            line = lines[suspect.line_index] if suspect.line_index < len(lines) else ""
            structure_issue = (
                structure_issues[suspect.line_index]
                if suspect.line_index < len(structure_issues)
                else None
            )
            if structure_issue is not None:
                result = classify_guarded_candidate(suspect, "structure", structure_issue)
            else:
                guard = _guard_candidate(
                    suspect,
                    mode=self.config.correction_mode,
                    line=line,
                    protected_terms=protected_terms,
                )
                if guard is not None:
                    category, reason = guard
                    result = classify_guarded_candidate(suspect, category, reason)
                elif self.config.llm_enabled and self._judge is not None:
                    top_fix = suspect.candidates[0][0] if suspect.candidates else ""
                    fixed_line = _build_fixed_line(line, suspect, top_fix)
                    context = _get_context(lines, suspect.line_index)

                    qwen_verdict = self._judge.judge_with_details(
                        original_line=line,
                        fixed_line=fixed_line,
                        original_token=suspect.original,
                        fixed_token=top_fix,
                        context=context,
                    )
                    if qwen_verdict.verdict == "FIX":
                        semantic_check = self._judge.semantic_check(
                            original_line=line,
                            fixed_line=fixed_line,
                            original_token=suspect.original,
                            fixed_token=top_fix,
                            context=context,
                        )
                        if semantic_check.verdict == "DIFF":
                            result = classify_guarded_candidate(
                                suspect,
                                "semantic",
                                semantic_check.reason or "意味保持チェックで差分あり",
                            )
                        else:
                            result = classify_with_qwen(
                                suspect, qwen_verdict,
                                escalation_threshold=self.config.escalation_threshold,
                                mode=self.config.correction_mode,
                            )
                    else:
                        result = classify_with_qwen(
                            suspect, qwen_verdict,
                            escalation_threshold=self.config.escalation_threshold,
                            mode=self.config.correction_mode,
                        )
                else:
                    result = classify_without_qwen(
                        suspect,
                        autofix_threshold=self.config.autofix_threshold,
                        escalation_threshold=self.config.min_candidate_prob,
                    )
            corrections.append(result)
            yield "llm", {"index": i, "total": len(filtered), "result": result}

        timing["llm_judge"] = time.perf_counter() - t0

        yield "done", PipelineResult(
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
        if self._server:
            self._server.stop()
            self._server = None
