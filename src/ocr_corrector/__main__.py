"""CLI entry point for the OCR correction pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .escalation import Verdict
from .pipeline import Pipeline
from .qwen_judge import KNOWN_ENDPOINTS

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp", "gif", "webp"}
TEXT_EXTENSIONS = {"txt", "text", "csv"}


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


def _is_image_file(path: str) -> bool:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return ext in IMAGE_EXTENSIONS


def _is_text_file(path: str) -> bool:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return ext in TEXT_EXTENSIONS


def _ocr_files(paths: list[str]) -> tuple[str, str, list]:
    """OCR one or more image files. Returns (combined_text, ocr_text, pages)."""
    from .ocr_frontend import ocr_image_with_layout

    pages = []
    for i, path in enumerate(paths, 1):
        label = f"({i}/{len(paths)}) " if len(paths) > 1 else ""
        print(f"Running OCR {label}on {path}...", file=sys.stderr)
        pages.append(ocr_image_with_layout(path))

    combined = "\n".join(pg.text.rstrip("\n") for pg in pages)
    n = len(paths)
    chars = len(combined)
    print(f"OCR complete: {n} image{'s' if n > 1 else ''}, {chars} chars", file=sys.stderr)
    return combined, combined, pages


def main():
    parser = argparse.ArgumentParser(
        prog="ocr-corrector",
        description="BERT + LLM OCR correction pipeline",
    )
    parser.add_argument(
        "input", nargs="*",
        help="Input image(s) or text file (multiple images are OCR'd and concatenated)",
    )
    parser.add_argument(
        "--text", action="store_true",
        help="Treat input as text file instead of image",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Save correction results (CSV, corrected texts) to this directory",
    )
    parser.add_argument(
        "--bert-model",
        default="cl-tohoku/bert-base-japanese-v3",
        help="BERT model name (default: cl-tohoku/bert-base-japanese-v3)",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen3.5-4B-Q4_K_M.gguf",
        help="LLM model name (default: Qwen3.5-4B-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--llm-api",
        default="http://localhost:8080/v1",
        help=(
            "OpenAI-compatible API base URL. "
            "Shortcuts: ollama, llama-server, lm-studio. "
            "(default: http://localhost:8080/v1 = llama-server)"
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
    parser.add_argument("--port", type=int, default=7860, help="WebUI server port (default: 7860)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # WebUI mode
    if args.webui:
        try:
            from .webui import launch
            launch(server_port=args.port)
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
    ocr_text = None  # non-None when input came from OCR
    ocr_pages = None  # list[OcrPage] for PDF generation

    if not args.input and not sys.stdin.isatty():
        text = sys.stdin.read()
    elif args.input == ["-"]:
        text = sys.stdin.read()
    elif args.input and args.text:
        # Explicit text mode (first file only)
        with open(args.input[0], encoding="utf-8") as f:
            text = f.read()
    elif args.input:
        # Auto-detect: all images → OCR, text file → read
        image_files = [p for p in args.input if _is_image_file(p)]
        text_files = [p for p in args.input if _is_text_file(p)]

        if image_files and text_files:
            print("Error: cannot mix image and text files.", file=sys.stderr)
            sys.exit(1)

        if image_files:
            text, ocr_text, ocr_pages = _ocr_files(image_files)
        elif text_files:
            with open(text_files[0], encoding="utf-8") as f:
                text = f.read()
        else:
            # Unknown extension — try as images
            text, ocr_text, ocr_pages = _ocr_files(args.input)
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

        # Save result files if --output-dir specified
        if args.output_dir:
            from .text_export import generate_downloads

            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            files = generate_downloads(
                original_text=text,
                ocr_text=ocr_text,
                resplit_lines=result.lines,
                corrections=result.corrections,
                llm_enabled=config.llm_enabled,
                autofix_threshold=config.autofix_threshold,
                pages=ocr_pages,
            )
            # Move from temp dir to output dir
            for src in files:
                dst = out / Path(src).name
                Path(src).rename(dst)
                print(f"  -> {dst}", file=sys.stderr)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
