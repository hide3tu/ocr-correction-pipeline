#Requires -Version 5.1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

Write-Host "=== OCR Correction Pipeline Installer ===" -ForegroundColor Cyan
Write-Host ""

# Python 3.10+ check / auto-install
$hasPython = $false
try {
    $null = Get-Command python -ErrorAction Stop
    $pyMajor = & python -c "import sys; print(sys.version_info.major)" 2>&1
    $pyMinor = & python -c "import sys; print(sys.version_info.minor)" 2>&1
    if ([int]$pyMajor -ge 3 -and [int]$pyMinor -ge 10) {
        $hasPython = $true
        Write-Host "Python $pyMajor.$pyMinor detected"
    } else {
        Write-Host "Python $pyMajor.$pyMinor found but 3.10+ required" -ForegroundColor Yellow
    }
} catch {}

if (-not $hasPython) {
    Write-Host ""
    Write-Host "Python 3.12 to install..." -ForegroundColor Cyan
    try {
        $null = Get-Command winget -ErrorAction Stop
        Write-Host "winget detected. Installing Python 3.12..."
        & winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        # Verify
        $null = Get-Command python -ErrorAction Stop
        $pyMajor = & python -c "import sys; print(sys.version_info.major)" 2>&1
        $pyMinor = & python -c "import sys; print(sys.version_info.minor)" 2>&1
        Write-Host "Python $pyMajor.$pyMinor installed" -ForegroundColor Green
    } catch {
        Write-Host ""
        Write-Host "Python auto-install failed." -ForegroundColor Red
        Write-Host "Download Python 3.12 from:" -ForegroundColor Yellow
        Write-Host "  https://www.python.org/downloads/" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "IMPORTANT: Check 'Add python.exe to PATH' during installation." -ForegroundColor Yellow
        Write-Host "After installing Python, run this script again."
        exit 1
    }
}

# Git check (optional - used for NDLOCR-Lite clone, falls back to release download)
$hasGit = $false
try {
    $null = Get-Command git -ErrorAction Stop
    $hasGit = $true
} catch {}

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

# Verify PyTorch installed correctly
$torchCheck = & python -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "WARNING: PyTorch installation may have failed." -ForegroundColor Red
    Write-Host "Retrying with pip (no cache)..." -ForegroundColor Yellow
    if ($hasNvidia) {
        & pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
    } else {
        & pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
    }
    $torchCheck2 = & python -c "import torch; print(torch.__version__)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: PyTorch installation failed." -ForegroundColor Red
        Write-Host "After installation completes, run manually:" -ForegroundColor Yellow
        Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
        Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor Cyan
    } else {
        Write-Host "PyTorch $torchCheck2 installed (retry succeeded)" -ForegroundColor Green
    }
} else {
    Write-Host "PyTorch $torchCheck OK" -ForegroundColor Green
}

# Install package
Write-Host ""
Write-Host "Installing ocr-corrector..."
& pip install -e .
& pip install "gradio>=4.0"

# Install NDLOCR-Lite (OCR frontend)
Write-Host ""
Write-Host "=== Setting up NDLOCR-Lite ===" -ForegroundColor Cyan
$ndlocrDir = Join-Path $PWD "ndlocr-lite"
if (Test-Path $ndlocrDir) {
    Write-Host "ndlocr-lite already exists. Skipping."
} elseif ($hasGit) {
    Write-Host "Cloning ndlocr-lite (via git)..."
    & git clone https://github.com/ndl-lab/ndlocr-lite.git
} else {
    Write-Host "Git not found. Downloading ndlocr-lite source archive..."
    # Release assets are GUI app bundles, not Python source.
    # Use GitHub source archive instead (contains requirements.txt + src/).
    try {
        $ndlocrRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/ndl-lab/ndlocr-lite/releases/latest"
        $ndlocrTag = $ndlocrRelease.tag_name
        $ndlocrUrl = "https://github.com/ndl-lab/ndlocr-lite/archive/refs/tags/$ndlocrTag.zip"
        $ndlocrZipPath = Join-Path $PWD "ndlocr-lite-src.zip"

        Write-Host "Downloading ndlocr-lite source (v$ndlocrTag)..."
        Invoke-WebRequest -Uri $ndlocrUrl -OutFile $ndlocrZipPath
        Write-Host "Extracting..."
        Expand-Archive -Path $ndlocrZipPath -DestinationPath $PWD -Force
        Remove-Item $ndlocrZipPath -Force
        # GitHub archives extract to ndlocr-lite-{tag}/
        $extracted = Get-ChildItem -Path $PWD -Directory | Where-Object { $_.Name -like "ndlocr-lite-*" } | Select-Object -First 1
        if ($extracted) {
            Rename-Item -Path $extracted.FullName -NewName "ndlocr-lite"
            Write-Host "ndlocr-lite ready (source v$ndlocrTag)" -ForegroundColor Green
        } else {
            Write-Host "WARNING: ndlocr-lite extraction failed." -ForegroundColor Yellow
            Write-Host "Download manually from: https://github.com/ndl-lab/ndlocr-lite" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "WARNING: Failed to download ndlocr-lite source: $_" -ForegroundColor Yellow
        Write-Host "Install git and re-run, or download manually." -ForegroundColor Yellow
    }
}
Write-Host "Installing ndlocr-lite dependencies..."
& pip install -r (Join-Path $ndlocrDir "requirements.txt")

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
# Download CJK font for searchable PDF
# ============================================================
Write-Host ""
Write-Host "=== Setting up CJK font ===" -ForegroundColor Cyan

$fontsDir = Join-Path $PWD "fonts"
if (-not (Test-Path $fontsDir)) { New-Item -ItemType Directory -Path $fontsDir | Out-Null }

$fontFile = Join-Path $fontsDir "ipaexg.ttf"
if (Test-Path $fontFile) {
    Write-Host "IPAex Gothic font already exists. Skipping download."
} else {
    $fontUrl = "https://moji.or.jp/wp-content/ipafont/IPAexfont/ipaexg00401.zip"
    $fontZip = Join-Path $fontsDir "ipaexg.zip"
    Write-Host "Downloading IPAex Gothic font..."
    try {
        Invoke-WebRequest -Uri $fontUrl -OutFile $fontZip
        $extractCode = @"
import zipfile
with zipfile.ZipFile(r'$fontZip') as z:
    for name in z.namelist():
        if name.endswith('.ttf'):
            with z.open(name) as src, open(r'$fontFile', 'wb') as dst:
                dst.write(src.read())
            break
"@
        $extractTmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "font_extract.py")
        Set-Content -Path $extractTmp -Value $extractCode -Encoding UTF8
        & python $extractTmp
        Remove-Item $extractTmp
        Remove-Item $fontZip -Force
        Write-Host "IPAex Gothic font ready" -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Font download failed. PDF generation will use system fonts as fallback." -ForegroundColor Yellow
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
Write-Host ""
Read-Host "Press Enter to exit"
