"""BERT perplexity-based OCR error detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SuspectToken:
    """A token flagged as potentially incorrect by BERT."""

    position: int  # token index in the input
    original: str  # the original token text
    probability: float  # probability of the original token
    candidates: list[tuple[str, float]]  # top-k (token_text, probability) pairs
    line_index: int = 0  # which line this belongs to
    char_offset: int = 0  # approximate character offset in original text


class BertScanner:
    """Scan text for low-probability tokens using BERT masked language model."""

    def __init__(self, model_name: str, device: str, threshold: float = 0.01):
        logger.info("Loading BERT model: %s -> %s", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
        self.mask_token_id = self.tokenizer.mask_token_id

    def scan_line(self, text: str, line_index: int = 0) -> list[SuspectToken]:
        """Scan a single line and return suspect tokens."""
        if not text.strip():
            return []

        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoding["input_ids"][0]
        num_tokens = len(input_ids)
        suspects = []

        # Skip [CLS] (index 0) and [SEP] (last index)
        for i in range(1, num_tokens - 1):
            original_id = input_ids[i].item()

            # Skip special tokens
            if original_id in self.tokenizer.all_special_ids:
                continue

            masked = input_ids.clone().unsqueeze(0).to(self.device)
            masked[0, i] = self.mask_token_id

            with torch.no_grad():
                logits = self.model(input_ids=masked).logits

            probs = torch.softmax(logits[0, i], dim=-1)
            original_prob = probs[original_id].item()

            if original_prob < self.threshold:
                top_k = probs.topk(5)
                candidates = [
                    (self.tokenizer.decode([idx.item()]).strip(), p.item())
                    for idx, p in zip(top_k.indices, top_k.values)
                ]

                original_text = self.tokenizer.decode([original_id]).strip()
                suspects.append(SuspectToken(
                    position=i,
                    original=original_text,
                    probability=original_prob,
                    candidates=candidates,
                    line_index=line_index,
                ))

        return suspects

    def scan(self, text: str) -> list[SuspectToken]:
        """Scan multi-line text and return all suspect tokens."""
        all_suspects = []
        for idx, line in enumerate(text.splitlines()):
            suspects = self.scan_line(line, line_index=idx)
            all_suspects.extend(suspects)
        return all_suspects

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("BERT model unloaded")
