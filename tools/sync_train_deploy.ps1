param(
  [string]$Workspace = "D:\agi_seed",
  [string]$ProjectDir = "D:\agi_seed\neurokernel-agi-seed",
  [string]$RemoteHost = "orangepi5",
  [string]$RemoteProject = "~/projects/neurokernel-agi-seed",
  [int]$Episodes = 50,
  [int]$Epochs = 300,
  [string]$Device = "auto",
  [double]$WeightDecay = 0.0001,
  [int]$Patience = 60
)

$ErrorActionPreference = "Stop"
function Run-Native {
  param([string]$Exe, [string[]]$NativeArgs)
  & $Exe @NativeArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed: $Exe $($NativeArgs -join ' ')"
  }
}

function Run-NativeTimed {
  param([string]$Exe, [string[]]$NativeArgs, [int]$TimeoutSeconds)
  $processInfo = [System.Diagnostics.ProcessStartInfo]::new()
  $processInfo.FileName = $Exe
  $processInfo.UseShellExecute = $false
  $processInfo.RedirectStandardOutput = $true
  $processInfo.RedirectStandardError = $true
  foreach ($arg in $NativeArgs) {
    [void]$processInfo.ArgumentList.Add($arg)
  }
  $process = [System.Diagnostics.Process]::Start($processInfo)
  $stdoutTask = $process.StandardOutput.ReadToEndAsync()
  $stderrTask = $process.StandardError.ReadToEndAsync()
  if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
    try { $process.Kill() } catch {}
    throw "Command timed out after ${TimeoutSeconds}s: $Exe $($NativeArgs -join ' ')"
  }
  $process.WaitForExit()
  $stdout = $stdoutTask.Result
  $stderr = $stderrTask.Result
  if ($stdout) { Write-Host $stdout.TrimEnd() }
  if ($stderr) { Write-Host $stderr.TrimEnd() }
  if ($process.ExitCode -ne 0) {
    throw "Command failed: $Exe $($NativeArgs -join ' ')"
  }
}

$DataDir = Join-Path $Workspace "data"
$ArtifactDir = Join-Path $Workspace "artifacts"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

$RemoteCounterfactual = "data/model_ready/counterfactual.jsonl"
$RemoteFeatures = "data/model_ready/features.jsonl"
$SshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2")
$ScpOptions = $SshOptions
$RemoteJobTimeoutSeconds = 1800
$RemoteIoTimeoutSeconds = 120

Write-Host "[1/9] Collecting candidate counterfactual v3.1 dataset on Orange Pi..."
Run-NativeTimed -Exe ssh -NativeArgs @($SshOptions + @($RemoteHost, "cd $RemoteProject && . venv/bin/activate && python -m neurokernel_seed.cli collect-counterfactual-dataset --split all --episodes $Episodes --agents random heuristic gated --counterfactual-out $RemoteCounterfactual --features-out $RemoteFeatures")) -TimeoutSeconds $RemoteJobTimeoutSeconds
Write-Host "[2/9] Copying features to laptop..."
Run-NativeTimed -Exe scp -NativeArgs @($ScpOptions + @("${RemoteHost}:$RemoteProject/$RemoteCounterfactual", (Join-Path $DataDir "counterfactual.jsonl"))) -TimeoutSeconds $RemoteIoTimeoutSeconds
Run-NativeTimed -Exe scp -NativeArgs @($ScpOptions + @("${RemoteHost}:$RemoteProject/$RemoteCounterfactual.meta.json", (Join-Path $DataDir "counterfactual.jsonl.meta.json"))) -TimeoutSeconds $RemoteIoTimeoutSeconds
Run-NativeTimed -Exe scp -NativeArgs @($ScpOptions + @("${RemoteHost}:$RemoteProject/$RemoteFeatures", (Join-Path $DataDir "features.jsonl"))) -TimeoutSeconds $RemoteIoTimeoutSeconds
Run-NativeTimed -Exe scp -NativeArgs @($ScpOptions + @("${RemoteHost}:$RemoteProject/$RemoteFeatures.manifest.json", (Join-Path $DataDir "features.jsonl.manifest.json"))) -TimeoutSeconds $RemoteIoTimeoutSeconds

Set-Location $ProjectDir
$Python = Join-Path $ProjectDir "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing venv Python at $Python. Run tools\setup_laptop_training.ps1 first."
}

Write-Host "[3/9] Validating local features..."
Run-Native -Exe $Python -NativeArgs @("-m", "neurokernel_seed.cli", "validate-features", (Join-Path $DataDir "features.jsonl"))
Write-Host "[4/9] Training world model on $Device..."
Run-Native -Exe $Python -NativeArgs @("-m", "neurokernel_seed.cli", "train-world-model", "--features", (Join-Path $DataDir "features.jsonl"), "--out", (Join-Path $ArtifactDir "world_model.pt"), "--epochs", "$Epochs", "--device", $Device, "--weight-decay", "$WeightDecay", "--patience", "$Patience")
Write-Host "[5/9] Evaluating checkpoint..."
Run-Native -Exe $Python -NativeArgs @("-m", "neurokernel_seed.cli", "eval-world-model", "--checkpoint", (Join-Path $ArtifactDir "world_model.pt"), "--features", (Join-Path $DataDir "features.jsonl"), "--split", "test", "--device", $Device)
Write-Host "[6/9] Evaluating action ranking..."
Run-Native -Exe $Python -NativeArgs @("-m", "neurokernel_seed.cli", "eval-action-ranking", "--checkpoint", (Join-Path $ArtifactDir "world_model.pt"), "--features", (Join-Path $DataDir "features.jsonl"), "--split", "test", "--device", $Device)
Write-Host "[7/9] Exporting ONNX..."
Run-Native -Exe $Python -NativeArgs @("-m", "neurokernel_seed.cli", "export-world-model-onnx", "--checkpoint", (Join-Path $ArtifactDir "world_model.pt"), "--out", (Join-Path $ArtifactDir "world_model.onnx"))

Write-Host "[8/9] Deploying ONNX to Orange Pi..."
Run-NativeTimed -Exe ssh -NativeArgs @($SshOptions + @($RemoteHost, "mkdir -p $RemoteProject/artifacts")) -TimeoutSeconds $RemoteIoTimeoutSeconds
Run-NativeTimed -Exe scp -NativeArgs @($ScpOptions + @((Join-Path $ArtifactDir "world_model.onnx"), "${RemoteHost}:$RemoteProject/artifacts/world_model.onnx")) -TimeoutSeconds $RemoteIoTimeoutSeconds
Run-NativeTimed -Exe scp -NativeArgs @($ScpOptions + @((Join-Path $ArtifactDir "world_model.manifest.json"), "${RemoteHost}:$RemoteProject/artifacts/world_model.manifest.json")) -TimeoutSeconds $RemoteIoTimeoutSeconds

Write-Host "[9/9] Checking Orange Pi ONNX runtime..."
Run-NativeTimed -Exe ssh -NativeArgs @($SshOptions + @($RemoteHost, "cd $RemoteProject && . venv/bin/activate && python - <<'PY'
import importlib.util
print('onnxruntime_available', bool(importlib.util.find_spec('onnxruntime')))
PY")) -TimeoutSeconds $RemoteIoTimeoutSeconds
