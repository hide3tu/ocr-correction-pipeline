"""Gradio WebUI for the OCR correction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator

from .config import PipelineConfig
from .escalation import Verdict
from .llm_server import DEFAULT_MODEL_DIR
from .pipeline import Pipeline
from .qwen_judge import KNOWN_ENDPOINTS
from .text_export import generate_downloads

logger = logging.getLogger(__name__)


def _find_gguf_models() -> list[str]:
    """Scan llm/models/ for available GGUF files."""
    if not DEFAULT_MODEL_DIR.exists():
        return []
    return [f.name for f in sorted(DEFAULT_MODEL_DIR.glob("*.gguf"))]


def _resolve_api_base(value: str) -> str:
    return KNOWN_ENDPOINTS.get(value, value)


def _format_row(c, lines: list[str]) -> list[Any]:
    line_text = ""
    if c.suspect.line_index < len(lines):
        line_text = lines[c.suspect.line_index].strip()
        if len(line_text) > 60:
            line_text = line_text[:60] + "..."
    return [
        c.suspect.line_index,
        c.suspect.original,
        c.suggested_fix,
        f"{c.suggested_prob:.0%}",
        c.qwen_verdict or "-",
        c.verdict.value,
        line_text,
    ]


def _run_pipeline_streaming(
    text: str,
    image,
    bert_model: str,
    llm_model: str,
    llm_enabled: bool,
    llm_api: str,
    gpu_mode: str,
    bert_threshold: float,
    escalation_threshold: float,
):
    """Run pipeline, yielding (table, timing, status, ocr_output, btn, files) at each stage."""
    import gradio as gr

    model_map = {
        "tohoku-bert-v3": "cl-tohoku/bert-base-japanese-v3",
        "luke-japanese-base-lite": "studio-ousia/luke-japanese-base-lite",
    }
    bert_model_name = model_map.get(bert_model, bert_model)

    # Button states
    btn_busy = gr.update(value="処理中...", interactive=False)
    btn_ready = gr.update(value="校正実行", interactive=True)

    # Initially hide OCR output
    ocr_hide = gr.update(value="", visible=False)
    ocr_display = ocr_hide

    # Stage 0: OCR (if image)
    ocr_text = None
    ocr_pages = None  # list[OcrPage] for PDF generation
    if image is not None:
        yield [], "", "OCR処理中...", ocr_display, btn_busy, None
        try:
            from .ocr_frontend import ocr_image_with_layout
            page = ocr_image_with_layout(image)
            ocr_text = page.text
            ocr_pages = [page]
            text = ocr_text
        except Exception as e:
            yield [], "", f"OCRエラー: {e}", ocr_display, btn_ready, None
            return

    if not text or not text.strip():
        yield [], "", "テキストが入力されていません", ocr_display, btn_ready, None
        return

    # Save original text for download (before pipeline re-splits by punctuation)
    original_text = text

    # Show OCR text immediately when available
    if ocr_text:
        ocr_display = gr.update(value=ocr_text, visible=True)
        yield [], "", f"OCR完了: {len(ocr_text)}文字。BERTスキャン開始...", ocr_display, btn_busy, None
    else:
        yield [], "", "BERTスキャン開始...", ocr_display, btn_busy, None

    config = PipelineConfig(
        bert_model=bert_model_name,
        llm_model=llm_model,
        llm_enabled=llm_enabled,
        llm_api_base=_resolve_api_base(llm_api),
        gpu_mode=gpu_mode,
        bert_threshold=bert_threshold,
        escalation_threshold=escalation_threshold,
    )

    pipeline = Pipeline(config)
    rows: list[list[Any]] = []
    lines: list[str] = []
    timing_str = ""

    try:
        for stage, data in pipeline.run_steps(text, ocr_text=ocr_text):
            if stage == "bert":
                lines = data["lines"]
                timing_str = f"BERTスキャン: {data['time']:.1f}秒"
                status = (
                    f"BERT完了: {data['raw']}箇所検出 → フィルタ後{data['filtered']}箇所 "
                    f"({data['time']:.1f}秒)"
                )
                if llm_enabled and data["filtered"] > 0:
                    status += f"。LLM判定開始 (0/{data['filtered']})..."
                yield [], timing_str, status, ocr_display, btn_busy, None

            elif stage == "llm":
                i = data["index"]
                total = data["total"]
                result = data["result"]
                rows.append(_format_row(result, lines))
                yield list(rows), timing_str, f"LLM判定中: {i+1}/{total}...", ocr_display, btn_busy, None

            elif stage == "done":
                final = data
                table_data = [_format_row(c, final.lines) for c in final.corrections]
                parts = []
                for k, v in final.timing.items():
                    label = {"bert_scan": "BERTスキャン", "llm_judge": "LLM判定"}.get(k, k)
                    parts.append(f"{label}: {v:.1f}秒")
                parts.append(f"合計: {sum(final.timing.values()):.1f}秒")
                timing_str = " / ".join(parts)

                n_fix = sum(1 for c in final.corrections if c.verdict == Verdict.AUTO_FIX)
                n_esc = sum(1 for c in final.corrections if c.verdict == Verdict.ESCALATE)
                n_keep = sum(1 for c in final.corrections if c.verdict == Verdict.AUTO_KEEP)
                status = (
                    f"完了 | 検出: {final.raw_suspects}箇所 → "
                    f"フィルタ後: {final.filtered_suspects}箇所 | "
                    f"AUTO-FIX: {n_fix} | ESCALATE: {n_esc} | AUTO-KEEP: {n_keep}"
                )

                # Generate download files
                dl_files = generate_downloads(
                    original_text=original_text,
                    ocr_text=ocr_text,
                    resplit_lines=final.lines,
                    corrections=final.corrections,
                    llm_enabled=llm_enabled,
                    autofix_threshold=config.autofix_threshold,
                    pages=ocr_pages,
                )
                yield table_data, timing_str, status, ocr_display, btn_ready, dl_files

    except Exception as e:
        logger.exception("Pipeline failed")
        yield [], "", f"パイプラインエラー: {e}", ocr_display, btn_ready, None
    finally:
        pipeline.cleanup()


def _run_multi_image_streaming(
    images,
    bert_model: str,
    llm_model: str,
    llm_enabled: bool,
    llm_api: str,
    gpu_mode: str,
    bert_threshold: float,
    escalation_threshold: float,
):
    """Run pipeline on multiple images, yielding (table, timing, status, ocr_output, btn, files)."""
    import gradio as gr

    model_map = {
        "tohoku-bert-v3": "cl-tohoku/bert-base-japanese-v3",
        "luke-japanese-base-lite": "studio-ousia/luke-japanese-base-lite",
    }
    bert_model_name = model_map.get(bert_model, bert_model)

    btn_busy = gr.update(value="処理中...", interactive=False)
    btn_ready = gr.update(value="校正実行", interactive=True)
    ocr_hide = gr.update(value="", visible=False)
    ocr_display = ocr_hide

    # Validate input
    if not images:
        yield [], "", "画像がアップロードされていません", ocr_hide, btn_ready, None
        return

    total_images = len(images)
    ocr_pages = []  # list[OcrPage] for PDF generation

    # Stage 0: OCR each image sequentially
    for i, img_path in enumerate(images):
        yield [], "", f"OCR処理中: 画像 {i + 1}/{total_images}...", ocr_display, btn_busy, None
        try:
            from .ocr_frontend import ocr_image_with_layout
            page = ocr_image_with_layout(img_path)
            ocr_pages.append(page)
        except Exception as e:
            yield [], "", f"OCRエラー (画像 {i + 1}): {e}", ocr_display, btn_ready, None
            return

    # Combine OCR results
    combined_ocr = "\n".join(pg.text.rstrip("\n") for pg in ocr_pages)
    original_text = combined_ocr
    ocr_display = gr.update(value=combined_ocr, visible=True)
    yield [], "", (
        f"OCR完了: {total_images}画像, {len(combined_ocr)}文字。BERTスキャン開始..."
    ), ocr_display, btn_busy, None

    # Run pipeline on combined text
    config = PipelineConfig(
        bert_model=bert_model_name,
        llm_model=llm_model,
        llm_enabled=llm_enabled,
        llm_api_base=_resolve_api_base(llm_api),
        gpu_mode=gpu_mode,
        bert_threshold=bert_threshold,
        escalation_threshold=escalation_threshold,
    )

    pipeline = Pipeline(config)
    rows: list[list[Any]] = []
    lines: list[str] = []
    timing_str = ""

    try:
        for stage, data in pipeline.run_steps(combined_ocr, ocr_text=combined_ocr):
            if stage == "bert":
                lines = data["lines"]
                timing_str = f"BERTスキャン: {data['time']:.1f}秒"
                status = (
                    f"BERT完了: {data['raw']}箇所検出 → フィルタ後{data['filtered']}箇所 "
                    f"({data['time']:.1f}秒)"
                )
                if llm_enabled and data["filtered"] > 0:
                    status += f"。LLM判定開始 (0/{data['filtered']})..."
                yield [], timing_str, status, ocr_display, btn_busy, None

            elif stage == "llm":
                i = data["index"]
                total = data["total"]
                result = data["result"]
                rows.append(_format_row(result, lines))
                yield list(rows), timing_str, f"LLM判定中: {i+1}/{total}...", ocr_display, btn_busy, None

            elif stage == "done":
                final = data
                table_data = [_format_row(c, final.lines) for c in final.corrections]
                parts = []
                for k, v in final.timing.items():
                    label = {"bert_scan": "BERTスキャン", "llm_judge": "LLM判定"}.get(k, k)
                    parts.append(f"{label}: {v:.1f}秒")
                parts.append(f"合計: {sum(final.timing.values()):.1f}秒")
                timing_str = " / ".join(parts)

                n_fix = sum(1 for c in final.corrections if c.verdict == Verdict.AUTO_FIX)
                n_esc = sum(1 for c in final.corrections if c.verdict == Verdict.ESCALATE)
                n_keep = sum(1 for c in final.corrections if c.verdict == Verdict.AUTO_KEEP)
                status = (
                    f"完了 ({total_images}画像) | 検出: {final.raw_suspects}箇所 → "
                    f"フィルタ後: {final.filtered_suspects}箇所 | "
                    f"AUTO-FIX: {n_fix} | ESCALATE: {n_esc} | AUTO-KEEP: {n_keep}"
                )

                dl_files = generate_downloads(
                    original_text=original_text,
                    ocr_text=combined_ocr,
                    resplit_lines=final.lines,
                    corrections=final.corrections,
                    llm_enabled=llm_enabled,
                    autofix_threshold=config.autofix_threshold,
                    pages=ocr_pages,
                )
                yield table_data, timing_str, status, ocr_display, btn_ready, dl_files

    except Exception as e:
        logger.exception("Multi-image pipeline failed")
        yield [], "", f"パイプラインエラー: {e}", ocr_display, btn_ready, None
    finally:
        pipeline.cleanup()


REPO_URL = "https://github.com/hide3tu/ocr-correction-pipeline"


def _check_update() -> str:
    """Check if a newer version is available on GitHub. Returns banner markdown or empty."""
    import json
    from urllib.error import URLError
    from urllib.request import Request, urlopen
    from . import __version__

    try:
        req = Request(
            "https://api.github.com/repos/hide3tu/ocr-correction-pipeline/releases/latest",
            method="GET",
        )
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            remote_ver = data.get("tag_name", "").lstrip("v")

        if remote_ver and remote_ver != __version__:
            return (
                f"**新しいバージョン v{remote_ver} があります**（現在 v{__version__}）— "
                f"[GitHub]({REPO_URL}/releases) からダウンロードしてください。"
            )
    except (URLError, OSError, KeyError):
        pass
    return ""


def create_app():
    """Create the Gradio app."""
    import gradio as gr

    gguf_models = _find_gguf_models()
    default_model = gguf_models[0] if gguf_models else "Qwen3.5-4B-Q4_K_M.gguf"
    update_banner = _check_update()

    with gr.Blocks(title="OCR校正パイプライン") as app:
        gr.Markdown("# OCR校正パイプライン\nBERT perplexityスキャン + LLM判定（llama-server / ollama）")
        if update_banner:
            gr.Markdown(f"> {update_banner}")

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 設定")

                bert_model = gr.Dropdown(
                    choices=["tohoku-bert-v3", "luke-japanese-base-lite"],
                    value="tohoku-bert-v3",
                    label="BERTモデル",
                )
                llm_model = gr.Dropdown(
                    choices=gguf_models or [default_model],
                    value=default_model,
                    label="LLMモデル (llm/models/ 内のGGUF)",
                    allow_custom_value=True,
                )
                llm_enabled = gr.Checkbox(value=True, label="LLM判定を使用")
                llm_api = gr.Textbox(
                    value="http://localhost:8080/v1",
                    label="LLM API URL",
                    info="llama-server, ollama, lm-studio も可",
                )
                gpu_mode = gr.Dropdown(
                    choices=["auto", "both-gpu", "bert-only", "cpu-only"],
                    value="auto",
                    label="GPU配置",
                )
                bert_threshold = gr.Slider(
                    minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                    label="BERT閾値",
                    info="この確率未満のトークンを検出",
                )
                escalation_threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.50, step=0.05,
                    label="エスカレーション閾値",
                    info="LLMがKEEPでもBERT確信度がこれ以上なら人間に差し戻し",
                )

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("テキスト入力"):
                        text_input = gr.Textbox(
                            lines=10,
                            label="OCRテキスト",
                            placeholder="OCRで読み取ったテキストを貼り付け...",
                        )
                        run_btn_text = gr.Button("校正実行", variant="primary", size="lg")
                    with gr.TabItem("画像入力"):
                        image_input = gr.Image(
                            type="filepath",
                            label="画像ファイル",
                        )
                        gr.Markdown("*画像からOCRテキストを抽出して校正します*")
                        run_btn_image = gr.Button("校正実行", variant="primary", size="lg")
                    with gr.TabItem("複数画像入力"):
                        multi_image_input = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="画像ファイル（複数可）",
                        )
                        gr.Markdown("*複数の画像からOCRテキストを抽出して一括校正します*")
                        run_btn_multi = gr.Button("校正実行", variant="primary", size="lg", elem_id="run-btn-multi")
                status_text = gr.Textbox(label="ステータス", interactive=False)

                ocr_output = gr.Textbox(
                    label="OCR読取結果",
                    lines=8,
                    interactive=False,
                    visible=False,
                )

                results_table = gr.Dataframe(
                    headers=["行", "元", "修正候補", "BERT確率", "LLM", "判定", "行テキスト"],
                    datatype=["number", "str", "str", "str", "str", "str", "str"],
                    label="校正結果",
                    wrap=True,
                )

                timing_text = gr.Textbox(label="処理時間", interactive=False)

                gr.Markdown("### ダウンロード")
                gr.Markdown(
                    "*OCR原文 / 校正結果CSV / BERT校正テキスト"
                    "（確信度≥70%を適用） / LLM校正テキスト"
                    "（LLM承認のみ適用） / 全校正テキスト"
                    "（BERT∪LLM）*"
                )
                download_files = gr.File(
                    label="校正結果ファイル",
                    file_count="multiple",
                    interactive=False,
                )

        # Text tab: image=None so pipeline runs on text only
        run_btn_text.click(
            fn=_run_pipeline_streaming,
            inputs=[
                text_input, image_input,
                bert_model, llm_model, llm_enabled, llm_api,
                gpu_mode, bert_threshold, escalation_threshold,
            ],
            outputs=[results_table, timing_text, status_text, ocr_output, run_btn_text, download_files],
            concurrency_limit=1,
            trigger_mode="once",
        )
        # Single image tab: same handler, dispatches on image presence
        run_btn_image.click(
            fn=_run_pipeline_streaming,
            inputs=[
                text_input, image_input,
                bert_model, llm_model, llm_enabled, llm_api,
                gpu_mode, bert_threshold, escalation_threshold,
            ],
            outputs=[results_table, timing_text, status_text, ocr_output, run_btn_image, download_files],
            concurrency_limit=1,
            trigger_mode="once",
        )

        run_btn_multi.click(
            fn=_run_multi_image_streaming,
            inputs=[
                multi_image_input,
                bert_model, llm_model, llm_enabled, llm_api,
                gpu_mode, bert_threshold, escalation_threshold,
            ],
            outputs=[results_table, timing_text, status_text, ocr_output, run_btn_multi, download_files],
            concurrency_limit=1,
            trigger_mode="once",
        )

    return app


def launch(share: bool = False, server_port: int = 7860):
    """Create and launch the Gradio app."""
    import gradio as gr
    app = create_app()
    app.launch(
        share=share,
        server_port=server_port,
        theme=gr.themes.Soft(),
        footer_links=[],
    )
