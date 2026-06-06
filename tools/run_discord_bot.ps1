param(
    [string]$CoreUrl = "http://127.0.0.1:8765",
    [string]$TokenEnv = "DISCORD_BOT_TOKEN",
    [string]$Prefix = "!nk",
    [Int64]$ChannelId = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . ".\venv\Scripts\Activate.ps1"
}

if (-not [Environment]::GetEnvironmentVariable($TokenEnv)) {
    throw "Missing $TokenEnv. Set it in the current shell before starting the bot."
}

if ($ChannelId -eq 0) {
    $ChannelIdValue = [Environment]::GetEnvironmentVariable("DISCORD_CHANNEL_ID")
    if (-not $ChannelIdValue) {
        throw "Missing DISCORD_CHANNEL_ID. Pass -ChannelId or set DISCORD_CHANNEL_ID."
    }
    $ChannelId = [Int64]$ChannelIdValue
}

python -m neurokernel_seed.cli serve-discord-bot --core-url $CoreUrl --token-env $TokenEnv --prefix $Prefix --channel-id $ChannelId
