import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import ctranslate2
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper Studio - Local AI Transcription")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models_cache"

UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Loaded model cache in memory so repeated runs don't reload from disk
_loaded_models: Dict[str, WhisperModel] = {}
_model_lock = threading.Lock()

# Active jobs and cancellation flags
_jobs: Dict[str, dict] = {}

AVAILABLE_MODELS = [
    {
        "id": "tiny",
        "name": "Tiny",
        "size": "~75 MB",
        "speed": "Fastest (~10x)",
        "description": "Great for quick drafts and clear speech on any CPU.",
    },
    {
        "id": "base",
        "name": "Base (Recommended)",
        "size": "~145 MB",
        "speed": "Very Fast (~7x)",
        "description": "Best everyday balance of speed and accuracy.",
    },
    {
        "id": "small",
        "name": "Small",
        "size": "~480 MB",
        "speed": "Fast (~4x)",
        "description": "High accuracy for accents, technical terms, and multi-language audio.",
    },
    {
        "id": "medium",
        "name": "Medium",
        "size": "~1.5 GB",
        "speed": "Moderate (~2x)",
        "description": "Near-studio quality transcription for complex audio.",
    },
    {
        "id": "large-v3-turbo",
        "name": "Large v3 Turbo",
        "size": "~1.6 GB",
        "speed": "Fast (~3x)",
        "description": "OpenAI's latest pruned Large-v3 model: Large-level accuracy at Medium speed.",
    },
    {
        "id": "large-v3",
        "name": "Large v3",
        "size": "~3.1 GB",
        "speed": "Slowest (1x)",
        "description": "Maximum accuracy across 99+ languages. Best with a GPU or plenty of RAM.",
    },
]

LANGUAGES = [
    {"code": "auto", "name": "Auto-Detect Language"},
    {"code": "en", "name": "English"},
    {"code": "es", "name": "Spanish"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "it", "name": "Italian"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "nl", "name": "Dutch"},
    {"code": "ru", "name": "Russian"},
    {"code": "zh", "name": "Chinese"},
    {"code": "ja", "name": "Japanese"},
    {"code": "ko", "name": "Korean"},
    {"code": "ar", "name": "Arabic"},
    {"code": "hi", "name": "Hindi"},
    {"code": "tr", "name": "Turkish"},
    {"code": "pl", "name": "Polish"},
    {"code": "uk", "name": "Ukrainian"},
    {"code": "vi", "name": "Vietnamese"},
    {"code": "id", "name": "Indonesian"},
    {"code": "th", "name": "Thai"},
    {"code": "sv", "name": "Swedish"},
]


def get_hardware_info() -> dict:
    try:
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception:
        cuda_count = 0

    if cuda_count > 0:
        return {
            "device": "cuda",
            "compute_type": "float16",
            "label": f"NVIDIA CUDA GPU ({cuda_count} detected) • FP16 Accelerated",
        }
    cpu_threads = max(1, (os.cpu_count() or 4) - 1)
    return {
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": cpu_threads,
        "label": f"CPU Mode ({cpu_threads} threads • INT8 Quantized for 4x Speed)",
    }


def get_or_load_model(model_id: str) -> WhisperModel:
    hw = get_hardware_info()
    cache_key = f"{model_id}:{hw['device']}:{hw['compute_type']}"
    with _model_lock:
        if cache_key not in _loaded_models:
            # Keep at most 1 model in RAM at a time to save memory
            _loaded_models.clear()
            kwargs = {
                "device": hw["device"],
                "compute_type": hw["compute_type"],
                "download_root": str(MODELS_DIR),
            }
            if hw["device"] == "cpu":
                kwargs["cpu_threads"] = hw.get("cpu_threads", 4)
            _loaded_models[cache_key] = WhisperModel(model_id, **kwargs)
        return _loaded_models[cache_key]


@app.get("/api/system-info")
def system_info():
    hw = get_hardware_info()
    return {
        "hardware": hw,
        "models": AVAILABLE_MODELS,
        "languages": LANGUAGES,
        "unlimited_file_size": True,
    }


@app.post("/api/upload")
async def upload_media(request: Request):
    """Stream any size audio/video file directly to disk without memory limits."""
    from urllib.parse import unquote

    raw_filename = unquote(request.headers.get("x-filename", "media.mp4"))
    job_id = str(uuid.uuid4())
    ext = Path(raw_filename).suffix or ".bin"
    safe_path = UPLOAD_DIR / f"{job_id}{ext}"

    size_bytes = 0
    with open(safe_path, "wb") as buffer:
        async for chunk in request.stream():
            size_bytes += len(chunk)
            buffer.write(chunk)

    _jobs[job_id] = {
        "id": job_id,
        "filename": raw_filename,
        "filepath": str(safe_path),
        "size_bytes": size_bytes,
        "cancelled": False,
        "created_at": time.time(),
    }

    return {
        "job_id": job_id,
        "filename": raw_filename,
        "size_bytes": size_bytes,
    }


@app.post("/api/cancel/{job_id}")
def cancel_job(job_id: str):
    if job_id in _jobs:
        _jobs[job_id]["cancelled"] = True
        return {"status": "cancelling"}
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/transcribe/{job_id}")
async def transcribe_stream(
    job_id: str,
    model: str = Query("base"),
    language: str = Query("auto"),
    task: str = Query("transcribe"),
    vad_filter: bool = Query(True),
):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Uploaded file session not found.")

    job = _jobs[job_id]
    filepath = job["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Media file no longer exists on disk.")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def push_event(event_type: str, payload: dict):
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))

    def worker():
        start_time = time.time()
        try:
            push_event(
                "status",
                {
                    "stage": "loading_model",
                    "message": f"Loading '{model}' model (first run downloads model weights automatically)...",
                },
            )
            whisper_model = get_or_load_model(model)

            if job.get("cancelled"):
                push_event("cancelled", {"message": "Transcription cancelled by user."})
                return

            push_event(
                "status",
                {
                    "stage": "analyzing",
                    "message": "Decoding media & detecting speech segments...",
                },
            )

            lang_arg: Optional[str] = None if language == "auto" else language
            segments_gen, info = whisper_model.transcribe(
                filepath,
                language=lang_arg,
                task=task,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=500),
                beam_size=5,
            )

            duration = float(info.duration or 0.0)
            push_event(
                "info",
                {
                    "detected_language": info.language,
                    "language_probability": round(float(info.language_probability or 0.0) * 100, 1),
                    "duration": duration,
                    "task": task,
                    "model": model,
                },
            )

            segment_count = 0
            for seg in segments_gen:
                if job.get("cancelled"):
                    push_event("cancelled", {"message": "Transcription stopped by user."})
                    return

                segment_count += 1
                progress = min(99.5, round((seg.end / duration) * 100, 1)) if duration > 0 else 0.0
                elapsed = round(time.time() - start_time, 1)

                push_event(
                    "segment",
                    {
                        "id": segment_count,
                        "start": round(float(seg.start), 3),
                        "end": round(float(seg.end), 3),
                        "text": seg.text.strip(),
                        "progress": progress,
                        "elapsed": elapsed,
                    },
                )

            total_elapsed = round(time.time() - start_time, 2)
            speed_factor = round(duration / total_elapsed, 1) if total_elapsed > 0 and duration > 0 else 0
            push_event(
                "done",
                {
                    "total_segments": segment_count,
                    "elapsed_seconds": total_elapsed,
                    "duration": duration,
                    "speed_factor": speed_factor,
                },
            )
        except Exception as exc:
            push_event("error", {"message": str(exc)})
        finally:
            # Clean up uploaded temporary file to save disk space
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass
            _jobs.pop(job_id, None)
            push_event("__eof__", {})

    threading.Thread(target=worker, daemon=True).start()

    async def event_generator():
        while True:
            event_type, payload = await queue.get()
            if event_type == "__eof__":
                break
            yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    import webbrowser

    port = 8765
    url = f"http://127.0.0.1:{port}"
    print("=" * 62)
    print(f"  Whisper Studio is starting at: {url}")
    print("  100% Free & Local • Unlimited File Size • Powered by faster-whisper")
    print("=" * 62)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
