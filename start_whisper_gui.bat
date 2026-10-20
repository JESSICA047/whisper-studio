@echo off
title Whisper Studio - Free Local AI Transcription
cd /d "%~dp0"

if not exist "node_modules" (
    echo Installing Whisper ONNX dependencies...
    call npm install
)

echo Starting Whisper Studio...
node server.js
pause
