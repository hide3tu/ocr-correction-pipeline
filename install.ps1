#Requires -Version 5.1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== OCR Correction Pipeline Installer ===" -ForegroundColor Cyan
Write-Host ""

# Python 3.10+ check
try {
    $null = Get-Command python -ErrorAction Stop
} catch {
    Write-Host "ERROR: python not found. Install Python 3.10+." -ForegroundColor Red
    exit 1
}

$pyMajor = & python -c "import sys; print(sys.version_info.major)" 2>&1
$pyMinor = & python -c "import sys; print(sys.version_info.minor)" 2>&1
$pyVersionRaw = "$pyMajor.$pyMinor"

if ([int]$pyMajor -lt 3 -or ([int]$pyMajor -eq 3 -and [int]$pyMinor -lt 10)) {
    Write-Host "ERROR: Python 3.10+ required (current: $pyVersionRaw)" -ForegroundColor Red
    exit 1
}
Write-Host "Python $pyVersionRaw detected"

# Create venv
if (Test-Path ".venv") {
    Write-Host "Existing .venv found. Reusing."
} else {
    Write-Host "Creating virtual environment..."
    & python -m venv .venv
}

# Activate venv
$activateScript = Join-Path (Join-Path ".venv" "Scripts") "Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: .venv/Scripts/Activate.ps1 not found." -ForegroundColor Red
    exit 1
}
& $activateScript
Write-Host "Virtual environment activated"

# Detect NVIDIA GPU
$hasNvidia = $false
try {
    $smiOutput = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasNvidia = $true
        Write-Host ""
        Write-Host "NVIDIA GPU detected:" -ForegroundColor Green
        Write-Host "  $smiOutput"
    }
} catch {
    # nvidia-smi not found
}

Write-Host ""
& python -m pip install --upgrade pip --quiet

if ($hasNvidia) {
    Write-Host "Installing PyTorch (CUDA 12.4)..."
    & pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "No GPU detected. Installing CPU PyTorch."
    & pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# Install package
Write-Host ""
Write-Host "Installing ocr-corrector..."
& pip install -e .

# Verify
Write-Host ""
Write-Host "=== Verify Installation ===" -ForegroundColor Cyan
$verifyCode = @'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
import transformers
print(f"  transformers: {transformers.__version__}")
'@
$tempFile = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "ocr_verify.py")
Set-Content -Path $tempFile -Value $verifyCode -Encoding UTF8
& python $tempFile
Remove-Item $tempFile

# LLM backend info
Write-Host ""
Write-Host "=== LLM Backend ===" -ForegroundColor Cyan
Write-Host "LLM judgment requires an OpenAI-compatible API server."
Write-Host "Recommended: llama-server (llama.cpp)" -ForegroundColor Green
Write-Host "  https://github.com/ggerganov/llama.cpp/releases"
Write-Host ""
Write-Host "Start llama-server with a GGUF model:"
Write-Host "  llama-server -m model.gguf --port 8080 --n-gpu-layers 99"
Write-Host ""
Write-Host "Other compatible servers: ollama, LM Studio, vLLM, etc."
Write-Host "Without LLM, use --no-llm for BERT-only mode."

Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Usage:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m ocr_corrector input.txt          # With llama-server on :8080"
Write-Host "  python -m ocr_corrector --no-llm input.txt # BERT-only mode"
Write-Host "  python -m ocr_corrector --llm-api ollama input.txt  # Use ollama"
Write-Host "  python -m ocr_corrector --help              # Help"
