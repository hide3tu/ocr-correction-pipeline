# OCR Correction Pipeline

BERT perplexityスキャン + LLM判定で、OCRの誤認識を自動検出・修正するツール。

画像を入れたらNDLOCR-LiteでOCRして、校正結果をテーブルで返す。テキスト入力にも対応。

## 最低動作環境

| 項目 | 必須 | 推奨 |
|------|------|------|
| OS | Windows 10/11, macOS (Intel/Apple Silicon), Linux | - |
| Python | 3.10以上 | 3.12 |
| メモリ | 8GB | 16GB以上 |
| ストレージ | 10GB（PyTorch + BERT + LLMモデル） | 15GB |
| GPU | なくても動く（CPU推論） | NVIDIA VRAM 4GB以上 |
| ネットワーク | 初回インストール時のみ必要 | - |

### GPU別の構成

| GPU VRAM | 構成 | BERT | LLM | 速度目安 |
|----------|------|------|-----|---------|
| なし | cpu-only | CPU | CPU | 遅い |
| 3-4GB | bert-only | GPU | CPU | BERT高速、LLM遅い |
| 6GB以上 | both-gpu | GPU | GPU | 両方高速 |

Apple Silicon (M1/M2/M3/M4) は統合メモリなのでVRAM制約なし。

## インストール

### Windows

PowerShell（Windows標準搭載）で実行。PowerShell 7は不要。

PythonもGitも入っていなければインストーラーがwinget経由で自動インストールする。

```powershell
# まだリポジトリを取得していない場合（Gitがあるなら）
git clone https://github.com/hide3tu/ocr-correction-pipeline.git
cd ocr-correction-pipeline

# GitがなければGitHubからZIPダウンロード→展開でもOK

# インストール
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Mac / Linux

```bash
git clone https://github.com/hide3tu/ocr-correction-pipeline.git
cd ocr-correction-pipeline
bash install.sh
```

インストーラーが自動で行うこと:

1. **Python** — 未インストールならwinget経由で自動インストール（Windows）
2. **Git** — 未インストールならwinget経由で自動インストール（Windows）
3. venv作成 + GPU検出に応じたPyTorchインストール
4. BERT関連パッケージ（transformers, fugashi, unidic-lite等）
5. NDLOCR-Lite（OCRエンジン、git clone + 依存インストール）
6. llama-serverバイナリ（llama.cpp最新リリースから自動ダウンロード）
7. LLMモデル（Qwen3.5-4B Q4_K_M、HuggingFaceから自動ダウンロード）

Windowsの場合、PythonもGitも入っていない状態からインストーラーだけで全部揃う。

## 使い方

### WebUI（推奨）

`start.bat`（Windows）または `start.sh`（Mac/Linux）をダブルクリック。ブラウザで http://localhost:7860 が開く。

### コマンドライン

```bash
# venv有効化
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # Mac/Linux

# 画像から校正（NDLOCR-Lite OCR → BERT → LLM）
python -m ocr_corrector scan.jpg

# テキストファイルから校正
python -m ocr_corrector --text input.txt

# BERT検出のみ（LLMなし、高速）
python -m ocr_corrector --no-llm scan.jpg
```

llama-serverは自動で起動・停止する。手動管理は不要。

## WebUI

`python -m ocr_corrector --webui` で http://localhost:7860 にアクセス。

- 画像またはテキストを入力して校正実行
- 処理はOCR → BERT → LLMの順で段階的に進捗表示
- 結果テーブルにAUTO-FIX（自動修正）/ ESCALATE（要確認）/ AUTO-KEEP（問題なし）を色分け表示

WebUI使用時は `pip install "gradio>=4.0"` が別途必要。

## LLMバックエンド

デフォルトはllama-server（llama.cpp）。OpenAI互換APIなら何でも使える。

```bash
# 別のLLMバックエンドを使う場合
python -m ocr_corrector --llm-api ollama scan.jpg
python -m ocr_corrector --llm-api lm-studio scan.jpg
python -m ocr_corrector --llm-api http://192.168.1.100:8080/v1 scan.jpg
```

モデル差し替えは `llm/models/` にGGUFファイルを置くだけ。

## GPU配置

```bash
python -m ocr_corrector --gpu-mode auto scan.jpg       # 自動検出（デフォルト）
python -m ocr_corrector --gpu-mode both-gpu scan.jpg    # BERT+LLM両方GPU
python -m ocr_corrector --gpu-mode bert-only scan.jpg   # BERTだけGPU
python -m ocr_corrector --gpu-mode cpu-only scan.jpg    # 全部CPU
```

## ディレクトリ構成

```
ocr-correction-pipeline/
├── install.sh / install.ps1       # インストーラー
├── llm/                           # 自動配置（git管理外）
│   ├── llama-server(.exe)
│   └── models/*.gguf
├── ndlocr-lite/                   # 自動clone（git管理外）
├── src/ocr_corrector/
│   ├── __main__.py                # CLI
│   ├── pipeline.py                # パイプライン本体
│   ├── bert_scanner.py            # BERTスキャン
│   ├── qwen_judge.py              # LLM判定（OpenAI互換API）
│   ├── escalation.py              # AUTO-FIX / ESCALATE / AUTO-KEEP
│   ├── llm_server.py              # llama-server自動起動
│   ├── gpu_detect.py              # GPU検出
│   ├── ocr_frontend.py            # NDLOCR-Lite OCR
│   ├── config.py                  # 設定
│   └── webui.py                   # Gradio WebUI
└── test_input.txt                 # テスト用OCRテキスト
```

## 関連記事

- [BERT+Qwen OCR校正パイプラインをPythonツールにした](https://lilting.ch/articles/bert-qwen-ocr-correction-tool)
- [エンコーダーモデル+ローカルLLMでOCR誤字を自動検出・修正する](https://lilting.ch/articles/encoder-local-llm-ocr-correction)
