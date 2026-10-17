param(
    [switch]$DisableHiddenSuffix,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$ttsPython = Join-Path $PSScriptRoot ".venv-rvc\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ttsPython)) {
    throw "Missing Python environment: $ttsPython"
}

$env:ZEROTTS_ENABLE_HIDDEN_SUFFIX = if ($DisableHiddenSuffix) { "0" } else { "1" }
if (-not $env:ZEROTTS_SUFFIX_DEBUG) { $env:ZEROTTS_SUFFIX_DEBUG = "1" }
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "4" }

if (-not $env:ZEROTTS_SUFFIX_ASR_MODEL) {
    $ttsAsrRoot = Join-Path $PSScriptRoot "models\diagnostics\whisper\models--Systran--faster-whisper-small\snapshots"
    if (Test-Path -LiteralPath $ttsAsrRoot) {
        $ttsAsrSnapshot = Get-ChildItem -LiteralPath $ttsAsrRoot -Directory |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "model.bin") } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($ttsAsrSnapshot) { $env:ZEROTTS_SUFFIX_ASR_MODEL = $ttsAsrSnapshot.FullName }
    }
}
if (-not $env:ZEROTTS_SUFFIX_ALIGN_MODEL) {
    $ttsAlignment = Join-Path $PSScriptRoot "models\diagnostics\alignment\vietnamese-250h"
    if (Test-Path -LiteralPath (Join-Path $ttsAlignment "config.json")) {
        $env:ZEROTTS_SUFFIX_ALIGN_MODEL = $ttsAlignment
    }
}

Write-Host "Python: $ttsPython"
Write-Host "ZEROTTS_ENABLE_HIDDEN_SUFFIX=$env:ZEROTTS_ENABLE_HIDDEN_SUFFIX"
if ($CheckOnly) {
    & $ttsPython -c "import server, faster_whisper, torch, transformers; from zerotts_suffix import HiddenSuffixConfig; c=HiddenSuffixConfig.from_env(); print('API imports OK'); print('Hidden suffix:', c.enabled); print('ASR model:', c.asr_model); print('Alignment model:', c.alignment_model)"
    exit $LASTEXITCODE
} else {
    & $ttsPython -m uvicorn server:app --host 0.0.0.0 --port 8000
}
