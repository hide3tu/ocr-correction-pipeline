#Requires -Version 5.1
<#
.SYNOPSIS
    OCR Correction Pipeline installer for Windows.
.DESCRIPTION
    Creates a Python venv, installs PyTorch (CUDA or CPU), and sets up the package.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== OCR Correction Pipeline Installer ===" -ForegroundColor Cyan
Write-Host ""

# Python 3.10+ check
try {
    $pyExe = Get-Command python -ErrorAction Stop | Select-Object -ExpandProperty Source
} catch {
    Write-Host "ERROR: python が見つかりません。Python 3.10以上をインストールしてください。" -ForegroundColor Red
    exit 1
}

$pyVersionRaw = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
$pyMajor = & python -c "import sys; print(sys.version_info.major)" 2>&1
$pyMinor = & python -c "import sys; print(sys.version_info.minor)" 2>&1

if ([int]$pyMajor -lt 3 -or ([int]$pyMajor -eq 3 -and [int]$pyMinor -lt 10)) {
    Write-Host "ERROR: Python 3.10以上が必要です（現在: $pyVersionRaw）" -ForegroundColor Red
    exit 1
}
Write-Host "Python $pyVersionRaw 検出"

# Create venv
if (Test-Path ".venv") {
    Write-Host "既存の .venv を検出。再利用します。"
} else {
    Write-Host "仮想環境を作成中..."
    & python -m venv .venv
}

# Activate venv
$activateScript = Join-Path ".venv" "Scripts" "Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: venv の Activate.ps1 が見つかりません。" -ForegroundColor Red
    exit 1
}
& $activateScript
Write-Host "仮想環境を有効化"

# Detect NVIDIA GPU
$hasNvidia = $false
try {
    $smiOutput = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasNvidia = $true
        Write-Host ""
        Write-Host "NVIDIA GPU 検出:" -ForegroundColor Green
        Write-Host "  $smiOutput"
    }
} catch {
    # nvidia-smi not found
}

Write-Host ""
& python -m pip install --upgrade pip --quiet

if ($hasNvidia) {
    Write-Host "PyTorch (CUDA 12.4) をインストール中..."
    & pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "GPU 未検出。CPU版 PyTorch をインストールします。"
    & pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# Install package
Write-Host ""
Write-Host "ocr-corrector をインストール中..."
& pip install -e .

# Verify
Write-Host ""
Write-Host "=== インストール確認 ===" -ForegroundColor Cyan
& python -c @"
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import transformers
print(f'  transformers: {transformers.__version__}')
"@

# Check ollama
Write-Host ""
$hasOllama = Get-Command ollama -ErrorAction SilentlyContinue
if ($hasOllama) {
    $ollamaVer = & ollama --version 2>&1
    Write-Host "ollama 検出: $ollamaVer"
    Write-Host ""
    $yn = Read-Host "デフォルトモデル (qwen3.5:4b) をダウンロードしますか？ [y/N]"
    if ($yn -match "^[Yy]") {
        & ollama pull qwen3.5:4b
    }
} else {
    Write-Host "ollama が見つかりません。" -ForegroundColor Yellow
    Write-Host "Qwen判定を使う場合はインストールしてください:"
    Write-Host "  https://ollama.com/download" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "ollama なしでも --no-qwen オプションでBERTのみモードが使えます。"
}

Write-Host ""
Write-Host "=== インストール完了 ===" -ForegroundColor Green
Write-Host ""
Write-Host "使い方:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m ocr_corrector input.txt           # テキストファイルを校正"
Write-Host "  python -m ocr_corrector --no-qwen input.txt # BERTのみモード"
Write-Host "  python -m ocr_corrector --webui              # WebUI起動"
Write-Host "  python -m ocr_corrector --help               # ヘルプ"
