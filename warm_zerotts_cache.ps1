param(
    [string]$TextsFile = "",
    [string]$BaseUrl = "https://tranvanhung203-edge-tts-api.hf.space",
    [string]$OutputDir = "",
    [int]$Limit = 0,
    [double]$Timeout = 1200,
    [int]$Attempts = 3,
    [double]$Delay = 0.5,
    [switch]$DryRun,
    [switch]$Rewarm
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$zeroCachePython = Join-Path $PSScriptRoot ".venv-rvc\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $zeroCachePython)) {
    $zeroCachePython = (Get-Command python -ErrorAction Stop).Source
}
$zeroCacheArguments = @(
    (Join-Path $PSScriptRoot "tools\warm_zerotts_cache.py"),
    "--base-url", $BaseUrl, "--limit", "$Limit", "--timeout", "$Timeout",
    "--attempts", "$Attempts", "--delay", "$Delay"
)
if ($TextsFile) { $zeroCacheArguments += @("--texts", $TextsFile) }
if ($OutputDir) { $zeroCacheArguments += @("--output", $OutputDir) }
if ($DryRun) { $zeroCacheArguments += "--dry-run" }
if ($Rewarm) { $zeroCacheArguments += "--rewarm" }
& $zeroCachePython @zeroCacheArguments
exit $LASTEXITCODE
