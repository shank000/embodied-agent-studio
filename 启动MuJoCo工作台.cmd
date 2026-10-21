@echo off
cd /d "%~dp0"
echo MuJoCo workspace: http://127.0.0.1:8000
echo Keep this window open while using the simulation.
".venv\Scripts\python.exe" -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --no-access-log
pause
