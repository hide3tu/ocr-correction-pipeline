"""Qwen LLM judgment for OCR correction candidates."""

from __future__ import annotations

import logging
import os

import ollama

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """以下のOCR読み取りテキストで、AとBのどちらが正しい日本語か判定してください。
AかBのみで答えてください。

文脈:
{context}

A: {line_a}
B: {line_b}"""


class QwenJudge:
    """Use Qwen via ollama to judge FIX/KEEP for each suspect."""

    def __init__(self, model: str = "qwen3.5:4b", ollama_num_gpu: str = "99"):
        self.model = model
        self._orig_env = os.environ.get("OLLAMA_NUM_GPU")
        os.environ["OLLAMA_NUM_GPU"] = ollama_num_gpu
        self._ensure_model()

    def _ensure_model(self):
        """Pull the model if not already downloaded."""
        try:
            ollama.show(self.model)
            logger.info("Qwen model ready: %s", self.model)
        except ollama.ResponseError:
            logger.info("Downloading model: %s ...", self.model)
            ollama.pull(self.model)
            logger.info("Model download complete: %s", self.model)

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

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 16},
            )
            answer = response["message"]["content"].strip().upper()
            logger.debug("Qwen response: %s", answer)

            if "B" in answer:
                return "FIX"
            return "KEEP"
        except Exception:
            logger.exception("Qwen judgment failed, defaulting to KEEP")
            return "KEEP"

    def cleanup(self):
        """Restore environment."""
        if self._orig_env is not None:
            os.environ["OLLAMA_NUM_GPU"] = self._orig_env
        elif "OLLAMA_NUM_GPU" in os.environ:
            del os.environ["OLLAMA_NUM_GPU"]
