@echo off
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
for %%I in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fI"
set "WORKSPACE_DIR=%PROJECT_DIR%\.."

if exist "%PROJECT_DIR%\src\neurokernel_seed\nk_cli.py" goto project_found
echo nk failed: project not found: %PROJECT_DIR%
exit /b 1

:project_found

if not defined NEUROKERNEL_TRAIN_FEATURES set "NEUROKERNEL_TRAIN_FEATURES=%WORKSPACE_DIR%\data\model_ready\features_slot_v2_model_needed_v3.jsonl"
if not defined NEUROKERNEL_TRAIN_RUN_DIR set "NEUROKERNEL_TRAIN_RUN_DIR=%WORKSPACE_DIR%\artifacts\training_runs"
if not defined NEUROKERNEL_EDGE_HOST set "NEUROKERNEL_EDGE_HOST=orangepi5"
if not defined NEUROKERNEL_EDGE_PROJECT set "NEUROKERNEL_EDGE_PROJECT=/home/ubuntu/projects/neurokernel-agi-seed"

set "PYTHONPATH=%PROJECT_DIR%\src;%PYTHONPATH%"
python -m neurokernel_seed.nk_cli %*
exit /b %ERRORLEVEL%
