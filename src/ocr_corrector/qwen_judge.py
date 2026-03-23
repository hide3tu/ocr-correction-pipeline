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
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """以下のOCR読み取りテキストで、AとBのどちらが正しい日本語か判定してください。
AかBのみで答えてください。

文脈:
{context}

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
        """Judge whether the fix should be applied.

        Returns:
            "FIX" if the fixed version is better, "KEEP" if original is fine.
        """
        prompt = JUDGE_PROMPT.format(
            context=context or "(なし)",
            line_a=original_line,
            line_b=fixed_line,
        )

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 16,
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
                answer = data["choices"][0]["message"]["content"].strip().upper()
                logger.debug("LLM response: %s", answer)

                if "B" in answer:
                    return "FIX"
                return "KEEP"
        except Exception:
            logger.exception("LLM judgment failed, defaulting to KEEP")
            return "KEEP"

    def cleanup(self):
        """No-op for HTTP-based client."""
        pass
