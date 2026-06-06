param(
  [string]$Workspace = "D:\agi_seed",
  [string]$ProjectDir = "D:\agi_seed\neurokernel-agi-seed",
  [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
Set-Location $ProjectDir

if (-not (Test-Path "venv")) {
  python -m venv venv
}

$Python = Join-Path $ProjectDir "venv\Scripts\python.exe"
& $Python -m pip install -U pip
if ($TorchIndexUrl -and $TorchIndexUrl.Trim().Length -gt 0) {
  & $Python -m pip install torch --index-url $TorchIndexUrl
}
& $Python -m pip install -e ".[model,dev]"
@'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device', torch.cuda.get_device_name(0))
'@ | & $Python -
