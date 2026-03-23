@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m ocr_corrector --webui
pause
