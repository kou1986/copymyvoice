@echo off
title CopyMyVoice
cd /d "D:\copymyvoice"
set OLLAMA_MODELS=D:\copymyvoice\models\ollama\models
set HF_HOME=D:\copymyvoice\models\huggingface
set ARGOS_STANZA_AVAILABLE=false
call ".venv\Scripts\activate.bat"
echo ============================================================
echo   CopyMyVoice starting...
echo   Browser will open at http://127.0.0.1:7860
echo   Close this window to stop the program.
echo ============================================================
python web_server.py
pause
