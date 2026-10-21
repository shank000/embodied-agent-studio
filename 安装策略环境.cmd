@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "POLICY_UV=uv"
where uv >nul 2>nul
if errorlevel 1 set "POLICY_UV=%USERPROFILE%\.hermes\bin\uv.exe"
if not exist ".venv\Scripts\python.exe" (
  echo 请先安装主工作台环境，参见 README.md。
  exit /b 1
)
if not exist ".venv-policy\Scripts\python.exe" (
  "%POLICY_UV%" venv --python ".venv\Scripts\python.exe" .venv-policy
  if errorlevel 1 exit /b 1
)
set "POLICY_TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "POLICY_TORCH_TAG=cpu"
if /I "%~1"=="cuda" (
  set "POLICY_TORCH_INDEX=https://download.pytorch.org/whl/cu128"
  set "POLICY_TORCH_TAG=cu128"
)
"%POLICY_UV%" pip install --python .venv-policy\Scripts\python.exe torch==2.10.0+%POLICY_TORCH_TAG% torchvision==0.25.0+%POLICY_TORCH_TAG% --index-url "%POLICY_TORCH_INDEX%"
if errorlevel 1 exit /b 1
"%POLICY_UV%" pip install --python .venv-policy\Scripts\python.exe -r requirements-policy.txt
if errorlevel 1 exit /b 1
echo 策略环境已安装。工作台会在首次运行 SmolVLA 时下载官方权重并启动本地推理服务。
