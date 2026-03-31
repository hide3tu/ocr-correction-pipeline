"""LLM judgment for OCR correction candidates.

Supports any OpenAI-compatible API endpoint:
- ollama (default, http://localhost:11434/v1)
- llama-server (http://localhost:8080/v1)
- LM Studio (http://localhost:1234/v1)
- Any other OpenAI-compatible server
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {
    "ocr_typo",
    "grammar",
    "punctuation",
    "proper_noun",
    "dialect",
    "paraphrase",
    "unclear",
}


@dataclass(frozen=True)
class JudgeResult:
    """Structured LLM judgment for a candidate correction."""

    verdict: str
    category: str = "unclear"
    reason: str = ""


JUDGE_PROMPT = """あなたはOCR校正の判定器です。
目的はOCR誤認識だけを直すことです。文章改善や言い換えは禁止です。

FIXにしてよい条件:
- BがAの明確なOCR誤認識、誤字脱字、文字化け、脱落を直している
- 文脈上もBでないと不自然または破綻する

次の場合は必ずKEEP:
- 人名、地名、作品固有語、専門用語の可能性がある
- 方言、口語、癖、文体差、表記ゆれ
- 句読点や記号だけの変更
- 意味の追加、言い換え、一般化、具体化
- どちらも成立し、誤りだと断定できない

categoryは次のいずれかを使う:
- ocr_typo
- grammar
- punctuation
- proper_noun
- dialect
- paraphrase
- unclear

JSONのみで答えてください。
{{"verdict":"FIX|KEEP","category":"ocr_typo|grammar|punctuation|proper_noun|dialect|paraphrase|unclear","reason":"短く"}}

文脈:
{context}

怪しい元トークン: {original_token}
修正候補: {fixed_token}

A: {line_a}
B: {line_b}"""

# Well-known endpoints
KNOWN_ENDPOINTS = {
    "ollama": "http://localhost:11434/v1",
    "llama-server": "http://localhost:8080/v1",
    "lm-studio": "http://localhost:1234/v1",
}


class LlmJudge:
    """Use any OpenAI-compatible LLM to judge FIX/KEEP for each suspect."""

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        api_base: str = "http://localhost:11434/v1",
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self._check_connection()

    def _check_connection(self):
        """Verify the API endpoint is reachable."""
        url = f"{self.api_base}/models"
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m.get("id", "") for m in data.get("data", [])]
                logger.info(
                    "LLM API connected: %s (%d models available)",
                    self.api_base, len(models),
                )
                if self.model not in models and models:
                    logger.warning(
                        "Model '%s' not found in available models: %s",
                        self.model, models[:5],
                    )
        except (URLError, OSError) as e:
            logger.warning("LLM API not reachable at %s: %s", self.api_base, e)
            raise ConnectionError(
                f"LLM API not reachable at {self.api_base}. "
                f"Start your LLM server (ollama, llama-server, LM Studio, etc.) "
                f"or use --no-qwen for BERT-only mode."
            ) from e

    def judge(
        self,
        original_line: str,
        fixed_line: str,
        context: str = "",
    ) -> str:
        """Backward-compatible verdict-only API."""
        return self.judge_with_details(
            original_line=original_line,
            fixed_line=fixed_line,
            context=context,
        ).verdict

    def judge_with_details(
        self,
        original_line: str,
        fixed_line: str,
        original_token: str = "",
        fixed_token: str = "",
        context: str = "",
    ) -> JudgeResult:
        """Judge whether the fix should be applied.

        Returns:
            Structured decision containing verdict, category, and reason.
        """
        prompt = JUDGE_PROMPT.format(
            context=context or "(なし)",
            original_token=original_token or "(不明)",
            fixed_token=fixed_token or "(不明)",
            line_a=original_line,
            line_b=fixed_line,
        )

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 96,
            "enable_thinking": False,
        }).encode("utf-8")

        url = f"{self.api_base}/chat/completions"
        req = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                answer = data["choices"][0]["message"]["content"].strip()
                logger.debug("LLM response: %s", answer)
                return _parse_judge_response(answer)
        except Exception:
            logger.exception("LLM judgment failed, defaulting to KEEP")
            return JudgeResult("KEEP", "unclear", "LLM判定失敗")

    def cleanup(self):
        """No-op for HTTP-based client."""
        pass


def _parse_judge_response(answer: str) -> JudgeResult:
    """Parse a model response into a normalized JudgeResult."""
    payload = answer.strip()
    if not payload:
        return JudgeResult("KEEP", "unclear", "空応答")

    candidates = [payload]
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        verdict = "FIX" if str(data.get("verdict", "")).upper() == "FIX" else "KEEP"
        category = str(data.get("category", "unclear")).strip().lower()
        if category not in ALLOWED_CATEGORIES:
            category = "unclear"
        reason = str(data.get("reason", "")).strip()
        return JudgeResult(verdict=verdict, category=category, reason=reason)

    if "FIX" in payload.upper():
        return JudgeResult("FIX", "unclear", "非JSON応答")
    return JudgeResult("KEEP", "unclear", "非JSON応答")
