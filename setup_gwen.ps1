param(
    [switch]$Cuda,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$gwenPython = Join-Path $PSScriptRoot ".venv-gwen\Scripts\python.exe"
if (-not $CheckOnly) {
    if (-not (Test-Path -LiteralPath $gwenPython)) {
        $gwenBootstrapPython = Join-Path $PSScriptRoot ".venv-rvc\Scripts\python.exe"
        if (Test-Path -LiteralPath $gwenBootstrapPython) {
            & $gwenBootstrapPython -m venv .venv-gwen
        } else {
            py -3.11 -m venv .venv-gwen
        }
        if ($LASTEXITCODE -ne 0) { throw "Could not create the Gwen Python environment" }
    }
    & $gwenPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Could not update Gwen pip" }
    $gwenWheelIndex = if ($Cuda) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
    $gwenTorchTag = if ($Cuda) { "cu128" } else { "cpu" }
    & $gwenPython -m pip install "torch==2.8.0+$gwenTorchTag" "torchaudio==2.8.0+$gwenTorchTag" --extra-index-url $gwenWheelIndex
    if ($LASTEXITCODE -ne 0) { throw "Could not install Gwen PyTorch" }
    & $gwenPython -m pip install -r requirements-gwen.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install Gwen dependencies" }
}
if (-not (Test-Path -LiteralPath $gwenPython)) { throw "Run setup_gwen.ps1 first" }
& $gwenPython -c "from qwen_tts import Qwen3TTSModel; import torch, sea_g2p; print('Gwen runtime OK'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Gwen runtime import failed" }
Write-Host "Ready. Start the API with start_server.ps1 and use /tts?provider=gwen&voice=yen_nhi&text=Xin%20chao"
