"""GPU detection and model placement."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_gpu() -> tuple[str, str | None, int | None]:
    """Detect available GPU and return (backend, device_name, vram_bytes).

    Returns:
        ("cuda", device_name, vram_bytes) for NVIDIA GPUs
        ("mps", "Apple Silicon", None) for Apple Silicon
        ("cpu", None, None) for CPU-only
    """
    try:
        import torch
    except ImportError:
        return "cpu", None, None

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory
        logger.info("CUDA detected: %s (VRAM: %.1f GB)", name, vram / 1e9)
        return "cuda", name, vram

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS detected: Apple Silicon")
        return "mps", "Apple Silicon", None

    logger.info("No GPU detected, using CPU")
    return "cpu", None, None


def resolve_device(gpu_mode: str) -> tuple[str, str]:
    """Resolve BERT device and Qwen GPU setting based on gpu_mode.

    Args:
        gpu_mode: "auto", "both-gpu", "bert-only", "qwen-only", "cpu-only"

    Returns:
        (bert_device, ollama_num_gpu) where ollama_num_gpu is "99" or "0"
    """
    backend, _, vram = detect_gpu()

    if gpu_mode == "cpu-only":
        return "cpu", "0"

    if gpu_mode == "auto":
        if backend == "cpu":
            return "cpu", "0"
        if backend == "mps":
            # Apple Silicon: both on GPU (unified memory)
            return "mps", "99"
        # CUDA: check VRAM
        if vram and vram >= 6 * 1e9:
            gpu_mode = "both-gpu"
        elif vram and vram >= 3 * 1e9:
            gpu_mode = "bert-only"
        else:
            return "cpu", "0"

    if gpu_mode == "both-gpu":
        return backend, "99"
    elif gpu_mode == "bert-only":
        return backend, "0"
    elif gpu_mode == "qwen-only":
        return "cpu", "99"

    return "cpu", "0"
