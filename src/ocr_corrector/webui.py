"""Gradio WebUI for the OCR correction pipeline."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def launch():
    """Launch the Gradio WebUI. Placeholder for Phase 5."""
    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        raise ImportError("Gradio is required: pip install gradio")

    logger.info("WebUI will be implemented in Phase 5")
    print("WebUI is not yet implemented. Coming in Phase 5.")
    raise SystemExit(0)
