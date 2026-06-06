param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8765,
    [string]$Db = "data/harness.db",
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . ".\venv\Scripts\Activate.ps1"
}

python -m neurokernel_seed.cli serve-core-api --host $HostAddress --port $Port --db $Db --project-root $ProjectRoot
