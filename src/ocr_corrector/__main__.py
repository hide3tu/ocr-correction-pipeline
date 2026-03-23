"""CLI entry point for the OCR correction pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import PipelineConfig
from .escalation import Verdict
from .pipeline import Pipeline
from .qwen_judge import KNOWN_ENDPOINTS


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_results(result, use_color: bool = True):
    """Print pipeline results to stdout."""
    colors = {
        Verdict.AUTO_FIX: "\033[32m",   # green
        Verdict.ESCALATE: "\033[33m",    # yellow
        Verdict.AUTO_KEEP: "\033[90m",   # gray
    }
    reset = "\033[0m"

    print(f"\n--- Results ---")
    print(f"Raw suspects: {result.raw_suspects}")
    print(f"After filter: {result.filtered_suspects}")
    print()

    for c in result.corrections:
        color = colors.get(c.verdict, "") if use_color else ""
        end = reset if use_color else ""

        line_text = ""
        if c.suspect.line_index < len(result.lines):
            line_text = result.lines[c.suspect.line_index].strip()

        print(
            f"{color}[{c.verdict.value}] "
            f"Line {c.suspect.line_index}: "
            f"'{c.suspect.original}' -> '{c.suggested_fix}' "
            f"(BERT: {c.suggested_prob:.0%}"
            f"{', LLM: ' + c.qwen_verdict if c.qwen_verdict else ''}"
            f"){end}"
        )
        if line_text:
            print(f"  {line_text}")
        print()

    # Timing
    if result.timing:
        print("--- Timing ---")
        for step, elapsed in result.timing.items():
            print(f"  {step}: {elapsed:.2f}s")
        total = sum(result.timing.values())
        print(f"  total: {total:.2f}s")


def _resolve_api_base(value: str) -> str:
    """Resolve shorthand names to full URLs."""
    return KNOWN_ENDPOINTS.get(value, value)


def main():
    parser = argparse.ArgumentParser(
        prog="ocr-corrector",
        description="BERT + LLM OCR correction pipeline",
    )
    parser.add_argument("input", nargs="?", help="Input text file (or - for stdin)")
    parser.add_argument("--image", help="Input image file (requires ndlocr-lite)")
    parser.add_argument(
        "--bert-model",
        default="cl-tohoku/bert-base-japanese-v3",
        help="BERT model name (default: cl-tohoku/bert-base-japanese-v3)",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen3.5:4b",
        help="LLM model name (default: qwen3.5:4b)",
    )
    parser.add_argument(
        "--llm-api",
        default="http://localhost:11434/v1",
        help=(
            "OpenAI-compatible API base URL. "
            "Shortcuts: ollama, llama-server, lm-studio. "
            "(default: http://localhost:11434/v1)"
        ),
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM judgment (BERT-only mode)",
    )
    parser.add_argument(
        "--gpu-mode",
        choices=["auto", "both-gpu", "bert-only", "qwen-only", "cpu-only"],
        default="auto",
        help="GPU placement mode (default: auto)",
    )
    parser.add_argument(
        "--bert-threshold",
        type=float,
        default=0.01,
        help="BERT perplexity threshold (default: 0.01)",
    )
    parser.add_argument(
        "--escalation-threshold",
        type=float,
        default=0.50,
        help="Escalation threshold for BERT confidence (default: 0.50)",
    )
    parser.add_argument("--webui", action="store_true", help="Launch Gradio WebUI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # WebUI mode
    if args.webui:
        try:
            from .webui import launch
            launch()
        except ImportError:
            print("Gradio is required for WebUI. Install with: pip install gradio", file=sys.stderr)
            sys.exit(1)
        return

    # Build config
    config = PipelineConfig(
        bert_model=args.bert_model,
        llm_model=args.llm_model,
        llm_enabled=not args.no_llm,
        llm_api_base=_resolve_api_base(args.llm_api),
        gpu_mode=args.gpu_mode,
        bert_threshold=args.bert_threshold,
        escalation_threshold=args.escalation_threshold,
    )

    # Get input text
    text = None
    if args.image:
        from .ocr_frontend import ocr_image
        print(f"Running OCR on {args.image}...")
        text = ocr_image(args.image)
        print(f"OCR complete: {len(text)} chars")
    elif args.input and args.input != "-":
        with open(args.input, encoding="utf-8") as f:
            text = f.read()
    elif args.input == "-" or not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    if not text or not text.strip():
        print("No input text.", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    pipeline = Pipeline(config)
    try:
        result = pipeline.run(text)
        _print_results(result, use_color=not args.no_color)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
