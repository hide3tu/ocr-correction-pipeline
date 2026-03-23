#!/bin/bash
set -e

echo "=== OCR Correction Pipeline Installer ==="
echo ""

# Python 3.10+ check
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 が見つかりません。Python 3.10以上をインストールしてください。"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10以上が必要です（現在: $PY_VERSION）"
    exit 1
fi
echo "Python $PY_VERSION 検出"

# Create venv
if [ -d ".venv" ]; then
    echo "既存の .venv を検出。再利用します。"
else
    echo "仮想環境を作成中..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "仮想環境を有効化: $(which python)"

# Detect platform and GPU
OS_NAME="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS_NAME" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
    echo ""
    echo "Apple Silicon (MPS) 検出"
    echo "PyTorch をインストール中..."
    pip install --upgrade pip
    pip install torch torchvision
elif command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "NVIDIA GPU 検出:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo "PyTorch (CUDA 12.4) をインストール中..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo ""
    echo "GPU 未検出。CPU版 PyTorch をインストールします。"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install the package
echo ""
echo "ocr-corrector をインストール中..."
pip install -e .

# Verify installation
echo ""
echo "=== インストール確認 ==="
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'  MPS: {torch.backends.mps.is_available()}')
import transformers
print(f'  transformers: {transformers.__version__}')
"

# LLM backend info
echo ""
echo "=== LLM Backend ==="
echo "LLM判定にはOpenAI互換APIサーバーが必要です。"
echo "推奨: llama-server (llama.cpp)"
echo "  https://github.com/ggerganov/llama.cpp/releases"
echo ""
echo "起動例:"
echo "  llama-server -m model.gguf --port 8080 --n-gpu-layers 99"
echo ""
echo "他の互換サーバー: ollama, LM Studio, vLLM 等も使用可"
echo "LLMなしでも --no-llm でBERTのみモードが使えます。"

echo ""
echo "=== インストール完了 ==="
echo ""
echo "使い方:"
echo "  source .venv/bin/activate"
echo "  python -m ocr_corrector input.txt          # llama-server :8080で校正"
echo "  python -m ocr_corrector --no-llm input.txt # BERTのみモード"
echo "  python -m ocr_corrector --llm-api ollama input.txt  # ollamaを使う場合"
echo "  python -m ocr_corrector --help              # ヘルプ"
