"""Gradio WebUI for the OCR correction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .escalation import Verdict
from .llm_server import DEFAULT_MODEL_DIR
from .pipeline import Pipeline
from .qwen_judge import KNOWN_ENDPOINTS

logger = logging.getLogger(__name__)


def _find_gguf_models() -> list[str]:
    """Scan llm/models/ for available GGUF files."""
    if not DEFAULT_MODEL_DIR.exists():
        return []
    return [f.name for f in sorted(DEFAULT_MODEL_DIR.glob("*.gguf"))]


def _resolve_api_base(value: str) -> str:
    return KNOWN_ENDPOINTS.get(value, value)


def _run_pipeline(
    text: str,
    image,
    bert_model: str,
    llm_model: str,
    llm_enabled: bool,
    llm_api: str,
    gpu_mode: str,
    bert_threshold: float,
    escalation_threshold: float,
) -> tuple[list[list[Any]], dict[str, Any], str]:
    """Run the pipeline and return (table_data, timing_dict, status_message)."""

    model_map = {
        "tohoku-bert-v3": "cl-tohoku/bert-base-japanese-v3",
        "luke-japanese-base-lite": "studio-ousia/luke-japanese-base-lite",
    }
    bert_model_name = model_map.get(bert_model, bert_model)

    if image is not None:
        try:
            from .ocr_frontend import ocr_image
            text = ocr_image(image)
        except ImportError:
            return [], {}, "ndlocr-lite がインストールされていません"
        except Exception as e:
            return [], {}, f"OCRエラー: {e}"

    if not text or not text.strip():
        return [], {}, "テキストが入力されていません"

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
    try:
        result = pipeline.run(text)
    except Exception as e:
        logger.exception("Pipeline failed")
        return [], {}, f"パイプラインエラー: {e}"
    finally:
        pipeline.cleanup()

    table_data = []
    for c in result.corrections:
        line_text = ""
        if c.suspect.line_index < len(result.lines):
            line_text = result.lines[c.suspect.line_index].strip()
            if len(line_text) > 60:
                line_text = line_text[:60] + "..."

        table_data.append([
            c.suspect.line_index,
            c.suspect.original,
            c.suggested_fix,
            f"{c.suggested_prob:.0%}",
            c.qwen_verdict or "-",
            c.verdict.value,
            line_text,
        ])

    timing = {k: f"{v:.2f}s" for k, v in result.timing.items()}
    timing["total"] = f"{sum(result.timing.values()):.2f}s"

    status = (
        f"検出: {result.raw_suspects}箇所 → フィルタ後: {result.filtered_suspects}箇所 | "
        f"AUTO-FIX: {sum(1 for c in result.corrections if c.verdict == Verdict.AUTO_FIX)} | "
        f"ESCALATE: {sum(1 for c in result.corrections if c.verdict == Verdict.ESCALATE)} | "
        f"AUTO-KEEP: {sum(1 for c in result.corrections if c.verdict == Verdict.AUTO_KEEP)}"
    )

    return table_data, timing, status


def create_app():
    """Create the Gradio app."""
    import gradio as gr

    gguf_models = _find_gguf_models()
    default_model = gguf_models[0] if gguf_models else "Qwen3.5-4B-Q4_K_M.gguf"

    with gr.Blocks(title="OCR校正パイプライン") as app:
        gr.Markdown("# OCR校正パイプライン\nBERT perplexityスキャン + LLM判定（llama-server / ollama）")

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
                    with gr.TabItem("画像入力"):
                        image_input = gr.Image(
                            type="filepath",
                            label="画像ファイル",
                        )
                        gr.Markdown("*画像入力にはndlocr-liteのインストールが必要です*")

                run_btn = gr.Button("校正実行", variant="primary", size="lg")
                status_text = gr.Textbox(label="ステータス", interactive=False)

                results_table = gr.Dataframe(
                    headers=["行", "元", "修正候補", "BERT確率", "LLM", "判定", "行テキスト"],
                    datatype=["number", "str", "str", "str", "str", "str", "str"],
                    label="校正結果",
                    wrap=True,
                )

                timing_json = gr.JSON(label="処理時間")

        run_btn.click(
            fn=_run_pipeline,
            inputs=[
                text_input, image_input,
                bert_model, llm_model, llm_enabled, llm_api,
                gpu_mode, bert_threshold, escalation_threshold,
            ],
            outputs=[results_table, timing_json, status_text],
        )

    return app


def launch(share: bool = False, server_port: int = 7860):
    """Create and launch the Gradio app."""
    app = create_app()
    app.launch(share=share, server_port=server_port, theme=gr.themes.Soft())
