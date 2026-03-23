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

# Install NDLOCR-Lite first to avoid pinned deps downgrading ocr-corrector's dependencies
echo ""
echo "=== Setting up NDLOCR-Lite ==="
if [ -d "ndlocr-lite" ]; then
    echo "ndlocr-lite already exists. Skipping clone."
else
    echo "Cloning ndlocr-lite..."
    git clone https://github.com/ndl-lab/ndlocr-lite.git
fi
echo "Installing ndlocr-lite dependencies..."
pip install -r ndlocr-lite/requirements.txt

# Install package (after ndlocr-lite so our deps take precedence)
echo ""
echo "Installing ocr-corrector..."
pip install -e .
pip install "gradio>=4.0"

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
    # Asset naming: llama-{tag}-bin-{platform}.{ext}
    if [[ "$OS_NAME" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
        PATTERN="bin-macos-arm64\.tar\.gz"
    elif [[ "$OS_NAME" == "Darwin" ]]; then
        PATTERN="bin-macos-x64\.tar\.gz"
    elif command -v nvidia-smi &>/dev/null; then
        PATTERN="bin-ubuntu-x64\.tar\.gz"
    else
        PATTERN="bin-ubuntu-x64\.tar\.gz"
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
        rm -f "$FILENAME"

        # Find llama-server in extracted directory and symlink it
        # Must stay in the extracted dir so shared libraries (dylibs/so) are found
        FOUND=$(find . -name "llama-server" -type f | head -1)
        if [ -n "$FOUND" ]; then
            chmod +x "$FOUND"
            # Create symlink at llm/llama-server pointing into the extracted dir
            ln -sf "$FOUND" ./llama-server
            echo "llama-server ready ($(readlink ./llama-server))"
        else
            echo "WARNING: llama-server binary not found in extracted archive"
        fi

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
    HF_REPO="unsloth/Qwen3.5-4B-GGUF"
    HF_FILE="Qwen3.5-4B-Q4_K_M.gguf"

    echo "Downloading $HF_FILE from $HF_REPO ..."
    echo "(~2.7 GB, this will take a while)"
    python -c "
from huggingface_hub import hf_hub_download
import sys
path = hf_hub_download(repo_id='$HF_REPO', filename='$HF_FILE', local_dir='llm/models')
print(path)
"

    FOUND_GGUF=$(find llm/models -name "*.gguf" | head -1)
    if [ -n "$FOUND_GGUF" ]; then
        echo "Model ready: $(basename $FOUND_GGUF)"
    else
        echo "WARNING: Model download failed. Place a .gguf file in llm/models/"
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
