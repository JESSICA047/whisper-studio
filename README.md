# Whisper Studio — Free AI Video & Audio Transcription GUI

A 100% free, open-source GUI for OpenAI's Whisper models with **unlimited file size**, live streaming timestamps, interactive audio/video playback, and one-click subtitle exports (`.SRT`, `.VTT`, `.TXT`, `.JSON`).

Works **both in the cloud on Vercel** (100% client-side WebAssembly AI engine in your browser) and **locally on your desktop** (powered by `FFmpeg` + quantized ONNX / `faster-whisper`).

## ✨ Features

- **100% Free & No API Keys Needed**: No OpenAI API fees and **no 25 MB file size limit**.
- **Dual Execution Engine**:
  - **🌐 Cloud / Vercel Mode (`static/worker.js`)**: Runs quantized OpenAI Whisper models (`tiny`, `base`, `small`) directly inside the visitor's browser using WebAssembly & WebAudio — zero server limits or serverless timeouts.
  - **⚡ Local Desktop Mode (`server.js` / `app.py`)**: Uses native `FFmpeg` + C++ ONNX Runtime (`@huggingface/transformers`) or Python `faster-whisper` (`CTranslate2`) to stream multi-gigabyte video/audio files in 30-second windows.
- **Interactive Media Player & Editable Timeline**:
  - Click any timestamp badge (`00:12 → 00:16`) to jump the video/audio player to that exact moment.
  - Click any transcribed line to edit words inline before exporting your subtitles.
- **Translate Any Language → English**: Switch between **Transcribe (Original Language)** and **Translate → English**.
- **One-Click Exports**: Download `.SRT` subtitles, `.VTT` web subtitles, `.TXT` plain text, or `.JSON` segment data, or copy the full transcript to your clipboard.

---

## 🚀 Deploy to Vercel

This repository is pre-configured with [`vercel.json`](./vercel.json) and [`.vercelignore`](./.vercelignore) to deploy as a fast static WebAssembly AI app on Vercel with zero serverless function timeouts.

---

## 💻 Run Locally (Windows / macOS / Linux)

Double-click **[`start_whisper_gui.bat`](./start_whisper_gui.bat)** on Windows, or run:

```bash
npm install
node server.js
```

Your browser will automatically open to **`http://127.0.0.1:8765`**.
