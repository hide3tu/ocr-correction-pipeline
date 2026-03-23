# OCR Correction Pipeline

BERT perplexity scan + Qwen LLM judgment pipeline for OCR error correction.

## Quick Start

```bash
# Install
pip install -e .

# Run on a text file
python -m ocr_corrector input.txt

# Run with image (requires ndlocr-lite)
python -m ocr_corrector --image scan.jpg

# BERT-only mode (no ollama needed)
python -m ocr_corrector --no-qwen input.txt

# Launch WebUI
python -m ocr_corrector --webui
```

## Requirements

- Python 3.10+
- PyTorch (CPU, CUDA, or MPS)
- [ollama](https://ollama.com) (for Qwen judgment, optional)

## GPU Modes

| Mode | BERT | Qwen | VRAM |
|------|------|------|------|
| `both-gpu` | GPU | GPU | ~5GB |
| `bert-only` | GPU | CPU | ~2.5GB |
| `qwen-only` | CPU | GPU | ~2.5GB |
| `cpu-only` | CPU | CPU | 0 |

```bash
python -m ocr_corrector --gpu-mode both-gpu input.txt
```
