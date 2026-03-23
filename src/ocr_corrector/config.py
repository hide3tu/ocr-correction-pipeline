"""Configuration for the OCR correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # BERT settings
    bert_model: str = "cl-tohoku/bert-base-japanese-v3"
    bert_threshold: float = 0.01  # tokens below this probability are flagged

    # LLM settings (any OpenAI-compatible endpoint)
    llm_model: str = "qwen3.5:4b"
    llm_enabled: bool = True
    llm_api_base: str = "http://localhost:11434/v1"  # ollama default

    # Escalation settings
    escalation_threshold: float = 0.50  # BERT confidence above this -> ESCALATE when Qwen says KEEP
    autofix_threshold: float = 0.70  # Qwen-OFF mode: BERT confidence above this -> AUTO-FIX

    # Filter settings
    min_candidate_prob: float = 0.30  # minimum BERT top1 probability to consider
    skip_subword: bool = True  # skip ##-prefixed subword tokens

    # GPU settings
    gpu_mode: str = "auto"  # auto, both-gpu, bert-only, qwen-only, cpu-only

    # Available BERT models
    BERT_MODELS: dict[str, str] = field(default_factory=lambda: {
        "tohoku-bert-v3": "cl-tohoku/bert-base-japanese-v3",
        "luke-japanese-base-lite": "studio-ousia/luke-japanese-base-lite",
    })
