# Whisper Studio — Free Local AI Video & Audio Transcription GUI

A 100% free, local web GUI for OpenAI's Whisper models with **unlimited file size**, live streaming timestamps, interactive audio/video playback, and one-click subtitle exports (`.SRT`, `.VTT`, `.TXT`, `.JSON`).

## ✨ Features

- **100% Free & Local**: Runs completely on your own computer — no OpenAI API keys, no cloud fees, and **no 25 MB file size limit**.
- **Unlimited Video & Audio File Sizes**: Streams multi-gigabyte video/audio files (`MP4`, `MKV`, `MOV`, `WEBM`, `MP3`, `WAV`, `M4A`, `FLAC`, `OGG`) directly to disk and processes audio in 30-second windows using `FFmpeg`, keeping RAM usage low even on 3-hour recordings.
- **Quantized INT8 Speed (4x Faster)**:
  - **Node.js Engine (`server.js`)**: Uses native C++ ONNX Runtime (`@huggingface/transformers`) with INT8 quantized Whisper models (`tiny`, `base`, `small`, `medium`, `large-v3-turbo`).
  - **Python Engine (`app.py`)**: Uses `faster-whisper` (`CTranslate2` INT8 on CPU / FP16 on NVIDIA CUDA GPUs).
- **Interactive Media Player & Editable Timeline**:
  - Click any timestamp badge (`00:12 → 00:16`) to jump the video/audio player to that exact moment.
  - Click any transcribed line to edit words inline before exporting your subtitles.
- **Translate Any Language → English**: Switch between **Transcribe (Original Language)** and **Translate → English**.
- **One-Click Exports**: Download `.SRT` subtitles, `.VTT` web subtitles, `.TXT` plain text, or `.JSON` segment data, or copy the full transcript to your clipboard.

---

## 🚀 Quick Start (Windows)

Double-click **[`start_whisper_gui.bat`](./start_whisper_gui.bat)** or run:

```powershell
node server.js
```

Your browser will automatically open to **`http://127.0.0.1:8765`**.

> **Note**: The first time you transcribe with a model size (e.g., `Tiny` ~75 MB or `Base` ~145 MB), the model weights are automatically downloaded and cached in the local `models_cache/` folder so future runs work completely offline.
