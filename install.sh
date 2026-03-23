#!/bin/bash
set -e

echo "=== OCR Correction Pipeline Installer ==="
echo ""

# Python 3.10+ check
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required (current: $PY_VERSION)"
    exit 1
fi
echo "Python $PY_VERSION detected"

# Create venv
if [ -d ".venv" ]; then
    echo "Existing .venv found. Reusing."
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Detect platform and GPU
OS_NAME="$(uname -s)"
ARCH="$(uname -m)"

pip install --upgrade pip --quiet

if [[ "$OS_NAME" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
    echo ""
    echo "Apple Silicon (MPS) detected"
    pip install torch torchvision
elif command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "NVIDIA GPU detected"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo ""
    echo "No GPU detected. Installing CPU PyTorch."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install package
echo ""
echo "Installing ocr-corrector..."
pip install -e .

# Verify
echo ""
echo "=== Verify Python Packages ==="
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'  MPS: {torch.backends.mps.is_available()}')
import transformers
print(f'  transformers: {transformers.__version__}')
"

# ============================================================
# Download llama-server
# ============================================================
echo ""
echo "=== Setting up llama-server ==="

mkdir -p llm/models

if [ -f "llm/llama-server" ]; then
    echo "llama-server already exists. Skipping download."
else
    echo "Fetching latest llama.cpp release..."
    RELEASE_JSON=$(curl -sL "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest")
    TAG=$(echo "$RELEASE_JSON" | python -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")

    # Determine asset pattern
    if [[ "$OS_NAME" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
        PATTERN="macos-arm64.zip"
    elif [[ "$OS_NAME" == "Darwin" ]]; then
        PATTERN="macos-x64.zip"
    elif command -v nvidia-smi &>/dev/null; then
        PATTERN="linux-x64-cuda.*\.tar\.gz"
    else
        PATTERN="linux-x64\.tar\.gz"
    fi

    ASSET_URL=$(echo "$RELEASE_JSON" | python -c "
import sys, json, re
data = json.load(sys.stdin)
for a in data['assets']:
    if re.search(r'$PATTERN', a['name']):
        print(a['browser_download_url'])
        break
")

    if [ -z "$ASSET_URL" ]; then
        echo "WARNING: Could not find llama.cpp binary for this platform in release $TAG"
        echo "Download manually from: https://github.com/ggerganov/llama.cpp/releases"
    else
        FILENAME=$(basename "$ASSET_URL")
        echo "Downloading $FILENAME ($TAG)..."
        curl -L -o "llm/$FILENAME" "$ASSET_URL"

        echo "Extracting..."
        cd llm
        if [[ "$FILENAME" == *.zip ]]; then
            unzip -o "$FILENAME"
        else
            tar xzf "$FILENAME"
        fi

        # Find and move llama-server
        FOUND=$(find . -name "llama-server" -type f | head -1)
        if [ -n "$FOUND" ] && [ "$FOUND" != "./llama-server" ]; then
            mv "$FOUND" ./llama-server
            chmod +x ./llama-server
        fi

        rm -f "$FILENAME"
        cd ..
        echo "llama-server ready"
    fi
fi

# ============================================================
# Download GGUF model
# ============================================================
echo ""
echo "=== Downloading LLM model ==="

EXISTING_GGUF=$(find llm/models -name "*.gguf" 2>/dev/null | head -1)

if [ -n "$EXISTING_GGUF" ]; then
    echo "GGUF model already exists: $(basename $EXISTING_GGUF). Skipping download."
else
    HF_REPO="Qwen/Qwen2.5-7B-Instruct-GGUF"
    HF_FILE="qwen2.5-7b-instruct-q4_k_m.gguf"

    echo "Downloading $HF_FILE from $HF_REPO ..."
    echo "(This may take a while: ~4.7 GB)"
    python -m huggingface_hub.commands.huggingface_cli download "$HF_REPO" "$HF_FILE" --local-dir llm/models

    FOUND_GGUF=$(find llm/models -name "*.gguf" | head -1)
    if [ -n "$FOUND_GGUF" ]; then
        echo "Model ready: $(basename $FOUND_GGUF)"
    else
        echo "WARNING: Model download may have failed. Place a .gguf file in llm/models/"
    fi
fi

# ============================================================
# Done
# ============================================================
echo ""
echo "=== Installation Complete ==="
echo ""
echo "Everything is set up. Just run:"
echo ""
echo "  source .venv/bin/activate"
echo "  python -m ocr_corrector input.txt"
echo ""
echo "llama-server will start automatically when needed."
echo "For BERT-only mode (no LLM): python -m ocr_corrector --no-llm input.txt"
