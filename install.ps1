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
} catch {}

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

# Verify Python packages
Write-Host ""
Write-Host "=== Verify Python Packages ===" -ForegroundColor Cyan
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

# ============================================================
# Download llama-server
# ============================================================
Write-Host ""
Write-Host "=== Setting up llama-server ===" -ForegroundColor Cyan

$llmDir = Join-Path $PWD "llm"
$modelsDir = Join-Path $llmDir "models"
if (-not (Test-Path $llmDir)) { New-Item -ItemType Directory -Path $llmDir | Out-Null }
if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }

$serverExe = Join-Path $llmDir "llama-server.exe"

if (Test-Path $serverExe) {
    Write-Host "llama-server.exe already exists. Skipping download."
} else {
    Write-Host "Fetching latest llama.cpp release..."
    $releaseJson = Invoke-RestMethod -Uri "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    $tag = $releaseJson.tag_name

    # Pick the right asset: llama-b{ver}-bin-win-cuda or cpu
    # Exclude cudart-only packages (they don't contain llama-server)
    if ($hasNvidia) {
        $asset = $releaseJson.assets | Where-Object {
            $_.name -match "^llama-.*bin-win-cuda-12.*x64\.zip$"
        } | Select-Object -First 1
    }
    if (-not $asset) {
        $asset = $releaseJson.assets | Where-Object {
            $_.name -match "^llama-.*bin-win-cpu-x64\.zip$"
        } | Select-Object -First 1
    }

    if (-not $asset) {
        Write-Host "WARNING: Could not find llama.cpp Windows binary in release $tag" -ForegroundColor Yellow
        Write-Host "Download manually from: https://github.com/ggerganov/llama.cpp/releases"
    } else {
        $zipUrl = $asset.browser_download_url
        $zipName = $asset.name
        $zipPath = Join-Path $llmDir $zipName

        Write-Host "Downloading $zipName ($tag)..."
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

        Write-Host "Extracting..."
        Expand-Archive -Path $zipPath -DestinationPath $llmDir -Force

        # Find llama-server.exe in extracted contents
        $found = Get-ChildItem -Path $llmDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
        if ($found -and $found.FullName -ne $serverExe) {
            Move-Item -Path $found.FullName -Destination $serverExe -Force
        }

        Remove-Item $zipPath -Force
        Write-Host "llama-server.exe ready" -ForegroundColor Green
    }
}

# ============================================================
# Download GGUF model
# ============================================================
Write-Host ""
Write-Host "=== Downloading LLM model ===" -ForegroundColor Cyan

$existingGguf = Get-ChildItem -Path $modelsDir -Filter "*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1

if ($existingGguf) {
    Write-Host "GGUF model already exists: $($existingGguf.Name). Skipping download."
} else {
    $hfRepo = "unsloth/Qwen3.5-4B-GGUF"
    $hfFile = "Qwen3.5-4B-Q4_K_M.gguf"

    Write-Host "Downloading $hfFile from $hfRepo ..."
    Write-Host "(~2.7 GB, this will take a while)"

    $dlCode = @'
from huggingface_hub import hf_hub_download
import sys
path = hf_hub_download(
    repo_id=sys.argv[1],
    filename=sys.argv[2],
    local_dir=sys.argv[3],
)
print(path)
'@
    $dlTmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "hf_dl.py")
    Set-Content -Path $dlTmp -Value $dlCode -Encoding UTF8
    $dlResult = & python $dlTmp $hfRepo $hfFile $modelsDir 2>&1
    Remove-Item $dlTmp

    $dlFile = Get-ChildItem -Path $modelsDir -Recurse -Filter "*.gguf" | Select-Object -First 1
    if ($dlFile) {
        if ($dlFile.DirectoryName -ne $modelsDir) {
            Move-Item -Path $dlFile.FullName -Destination (Join-Path $modelsDir $dlFile.Name) -Force
        }
        Write-Host "Model ready: $($dlFile.Name)" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Model download failed. Place a .gguf file in llm/models/" -ForegroundColor Yellow
    }
}

# ============================================================
# Done
# ============================================================
Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Everything is set up. Just run:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m ocr_corrector input.txt"
Write-Host ""
Write-Host "llama-server will start automatically when needed."
Write-Host "For BERT-only mode (no LLM): python -m ocr_corrector --no-llm input.txt"
