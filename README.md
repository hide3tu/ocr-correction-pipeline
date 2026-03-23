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

事前に [Python 3.12](https://www.python.org/downloads/) を入れておくのを推奨（インストール時に「Add python.exe to PATH」にチェック）。未インストールでもインストーラーがwinget経由で自動インストールを試みる。Gitはあると速いがなくても動く。

**方法A: リリースページからZIPダウンロード（Git不要）**

1. [Releases](https://github.com/hide3tu/ocr-correction-pipeline/releases) から最新版の「Source code (zip)」をダウンロード
2. ZIPを展開してフォルダに入る
3. `install.ps1` を右クリック →「PowerShellで実行」、または以下を実行:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

**方法B: git clone（Gitがある場合）**

```powershell
git clone https://github.com/hide3tu/ocr-correction-pipeline.git
cd ocr-correction-pipeline
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Mac / Linux

**方法A: リリースページからダウンロード（Git不要）**

1. [Releases](https://github.com/hide3tu/ocr-correction-pipeline/releases) から最新版の「Source code (tar.gz)」をダウンロード
2. 展開してフォルダに入り `bash install.sh`

**方法B: git clone**

```bash
git clone https://github.com/hide3tu/ocr-correction-pipeline.git
cd ocr-correction-pipeline
bash install.sh
```

インストーラーが自動で行うこと:

1. **Python** — 未インストールならwinget経由で自動インストール（Windows）
2. venv作成 + GPU検出に応じたPyTorchインストール
3. BERT関連パッケージ（transformers, fugashi, unidic-lite等）
4. NDLOCR-Lite — Gitがあればclone、なければGitHubソースアーカイブを自動DL
5. llama-serverバイナリ（llama.cpp最新リリースから自動ダウンロード）
6. LLMモデル（Qwen3.5-4B Q4_K_M、HuggingFaceから自動ダウンロード）
7. Gradio（WebUI用）

Gitは入れておくとインストールが速い（cloneの方がソースアーカイブDLより効率的）。なくても動く。

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

`start.bat` / `start.sh` で http://localhost:7860 にWebUIが開く。

- 画像またはテキストを入力して校正実行
- 処理はOCR → BERT → LLMの順で段階的に進捗表示
- 結果テーブルにAUTO-FIX（自動修正）/ ESCALATE（要確認）/ AUTO-KEEP（問題なし）を色分け表示
- LLM API URLの欄でollama、LM Studio等のバックエンドにも切り替え可能

## LLMバックエンド

デフォルトはllama-server（llama.cpp）。OpenAI互換APIなら何でも使える。WebUIのLLM API URL欄か、CLIの `--llm-api` オプションで切り替え。

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
├── start.bat / start.sh           # WebUI起動（ダブルクリック）
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
