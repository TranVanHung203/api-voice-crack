param(
    [string]$TextsFile = "",
    [string]$BaseUrl = "https://tranvanhung203-edge-tts-api.hf.space",
    [int]$Limit = 0,
    [switch]$DryRun,
    [switch]$Rewarm
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$cacheWarmPython = Join-Path $PSScriptRoot ".venv-rvc\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $cacheWarmPython)) {
    $cacheWarmPython = (Get-Command python -ErrorAction Stop).Source
}
$cacheWarmArguments = @((Join-Path $PSScriptRoot "tools\warm_gwen_cache.py"), "--base-url", $BaseUrl, "--limit", "$Limit")
if ($TextsFile) { $cacheWarmArguments += @("--texts", $TextsFile) }
if ($DryRun) { $cacheWarmArguments += "--dry-run" }
if ($Rewarm) { $cacheWarmArguments += "--rewarm" }
& $cacheWarmPython @cacheWarmArguments
exit $LASTEXITCODE
