const http = require("http");
const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");
const { spawn, exec } = require("child_process");

const PORT = 8765;
const BASE_DIR = __dirname;
const STATIC_DIR = path.join(BASE_DIR, "public");
const UPLOAD_DIR = path.join(BASE_DIR, "uploads");
const MODELS_DIR = path.join(BASE_DIR, "models_cache");

fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(MODELS_DIR, { recursive: true });
fs.mkdirSync(STATIC_DIR, { recursive: true });

const FFMPEG_BIN = fs.existsSync("C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe")
  ? "C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe"
  : "ffmpeg";

const AVAILABLE_MODELS = [
  {
    id: "tiny",
    repo: "onnx-community/whisper-tiny",
    name: "Tiny",
    size: "~75 MB",
    speed: "Fastest (~10x)",
    description: "Great for quick drafts and clear speech on any CPU.",
  },
  {
    id: "base",
    repo: "onnx-community/whisper-base",
    name: "Base (Recommended)",
    size: "~145 MB",
    speed: "Very Fast (~7x)",
    description: "Best everyday balance of speed and accuracy.",
  },
  {
    id: "small",
    repo: "onnx-community/whisper-small",
    name: "Small",
    size: "~480 MB",
    speed: "Fast (~4x)",
    description: "High accuracy for accents, technical terms, and multi-language audio.",
  },
  {
    id: "medium",
    repo: "onnx-community/whisper-medium",
    name: "Medium",
    size: "~1.5 GB",
    speed: "Moderate (~2x)",
    description: "Near-studio quality transcription for complex audio.",
  },
  {
    id: "large-v3-turbo",
    repo: "onnx-community/whisper-large-v3-turbo",
    name: "Large v3 Turbo",
    size: "~1.6 GB",
    speed: "Fast (~3x)",
    description: "OpenAI's latest pruned Large-v3 model: Large-level accuracy at Medium speed.",
  },
];

const LANGUAGES = [
  { code: "auto", name: "Auto-Detect / Multilingual" },
  { code: "english", name: "English" },
  { code: "spanish", name: "Spanish" },
  { code: "french", name: "French" },
  { code: "german", name: "German" },
  { code: "italian", name: "Italian" },
  { code: "portuguese", name: "Portuguese" },
  { code: "dutch", name: "Dutch" },
  { code: "russian", name: "Russian" },
  { code: "chinese", name: "Chinese" },
  { code: "japanese", name: "Japanese" },
  { code: "korean", name: "Korean" },
  { code: "arabic", name: "Arabic" },
  { code: "hindi", name: "Hindi" },
  { code: "turkish", name: "Turkish" },
  { code: "polish", name: "Polish" },
  { code: "ukrainian", name: "Ukrainian" },
  { code: "vietnamese", name: "Vietnamese" },
  { code: "indonesian", name: "Indonesian" },
  { code: "thai", name: "Thai" },
  { code: "swedish", name: "Swedish" },
];

const jobs = new Map();
let cachedPipeline = { modelId: null, transcriber: null };

async function getTranscriber(modelId, onProgress) {
  const modelMeta = AVAILABLE_MODELS.find((m) => m.id === modelId) || AVAILABLE_MODELS[1];
  if (cachedPipeline.modelId === modelMeta.id && cachedPipeline.transcriber) {
    return cachedPipeline.transcriber;
  }

  const { pipeline, env } = await import("@huggingface/transformers");
  env.cacheDir = MODELS_DIR;
  env.allowLocalModels = true;

  const transcriber = await pipeline("automatic-speech-recognition", modelMeta.repo, {
    dtype: {
      encoder_model: "fp32",
      decoder_model_merged: "q8",
    },
    progress_callback: (info) => {
      if (onProgress && info.status === "progress" && typeof info.progress === "number") {
        onProgress(`Downloading '${modelMeta.name}' model (${Math.round(info.progress)}%)...`);
      }
    },
  });

  cachedPipeline = { modelId: modelMeta.id, transcriber };
  return transcriber;
}

function getMediaDuration(filepath) {
  return new Promise((resolve) => {
    const proc = spawn(FFMPEG_BIN, ["-hide_banner", "-i", filepath]);
    let stderr = "";
    proc.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    proc.on("close", () => {
      const match = stderr.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/);
      if (match) {
        const hrs = parseFloat(match[1]);
        const mins = parseFloat(match[2]);
        const secs = parseFloat(match[3]);
        resolve(hrs * 3600 + mins * 60 + secs);
      } else {
        resolve(0);
      }
    });
    proc.on("error", () => resolve(0));
  });
}

function extractAudioChunkFloat32(filepath, startSec, durationSec) {
  return new Promise((resolve, reject) => {
    const args = [
      "-hide_banner",
      "-loglevel",
      "error",
      "-ss",
      String(startSec),
      "-t",
      String(durationSec),
      "-i",
      filepath,
      "-f",
      "f32le",
      "-acodec",
      "pcm_f32le",
      "-ac",
      "1",
      "-ar",
      "16000",
      "pipe:1",
    ];

    const proc = spawn(FFMPEG_BIN, args);
    const buffers = [];
    let stderr = "";

    proc.stdout.on("data", (data) => buffers.push(data));
    proc.stderr.on("data", (data) => (stderr += data.toString()));

    proc.on("close", (code) => {
      if (code !== 0 && buffers.length === 0) {
        return reject(new Error(stderr || `FFmpeg exited with code ${code}`));
      }
      const combined = Buffer.concat(buffers);
      const floatCount = Math.floor(combined.byteLength / 4);
      const float32 = new Float32Array(
        combined.buffer,
        combined.byteOffset,
        floatCount
      );
      resolve(float32);
    });

    proc.on("error", (err) => reject(err));
  });
}

function computeRmsEnergy(samples) {
  if (!samples || samples.length === 0) return 0;
  let sum = 0;
  const step = Math.max(1, Math.floor(samples.length / 8000));
  let count = 0;
  for (let i = 0; i < samples.length; i += step) {
    sum += samples[i] * samples[i];
    count++;
  }
  return Math.sqrt(sum / Math.max(1, count));
}

function sendJson(res, status, data) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(data));
}

const server = http.createServer(async (req, res) => {
  const parsedUrl = new URL(req.url, `http://127.0.0.1:${PORT}`);
  const pathname = parsedUrl.pathname;

  // 1. Serve Index & Static Files
  if (req.method === "GET" && (pathname === "/" || pathname === "/index.html")) {
    const htmlPath = path.join(STATIC_DIR, "index.html");
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    return fs.createReadStream(htmlPath).pipe(res);
  }

  if (req.method === "GET" && pathname === "/worker.js") {
    const workerPath = path.join(STATIC_DIR, "worker.js");
    res.writeHead(200, { "Content-Type": "application/javascript; charset=utf-8" });
    return fs.createReadStream(workerPath).pipe(res);
  }

  // 2. System Info
  if (req.method === "GET" && pathname === "/api/system-info") {
    const cpuCount = os.cpus().length;
    return sendJson(res, 200, {
      hardware: {
        device: "cpu",
        compute_type: "int8",
        label: `Local ONNX INT8 Engine (${cpuCount} CPU Threads • FFmpeg 8.1)`,
      },
      models: AVAILABLE_MODELS,
      languages: LANGUAGES,
      unlimited_file_size: true,
    });
  }

  // 3. Unlimited-Size Streaming Upload
  if (req.method === "POST" && pathname === "/api/upload") {
    const rawFilename = decodeURIComponent(req.headers["x-filename"] || "media.mp4");
    const ext = path.extname(rawFilename) || ".bin";
    const jobId = crypto.randomUUID();
    const safePath = path.join(UPLOAD_DIR, `${jobId}${ext}`);

    const writeStream = fs.createWriteStream(safePath);
    let sizeBytes = 0;

    req.on("data", (chunk) => {
      sizeBytes += chunk.length;
    });

    req.pipe(writeStream);

    writeStream.on("finish", () => {
      jobs.set(jobId, {
        id: jobId,
        filename: rawFilename,
        filepath: safePath,
        size_bytes: sizeBytes,
        cancelled: false,
        created_at: Date.now(),
      });
      sendJson(res, 200, {
        job_id: jobId,
        filename: rawFilename,
        size_bytes: sizeBytes,
      });
    });

    writeStream.on("error", (err) => {
      sendJson(res, 500, { detail: err.message });
    });
    return;
  }

  // 4. Cancel Active Job
  if (req.method === "POST" && pathname.startsWith("/api/cancel/")) {
    const jobId = pathname.split("/").pop();
    if (jobs.has(jobId)) {
      jobs.get(jobId).cancelled = true;
      return sendJson(res, 200, { status: "cancelling" });
    }
    return sendJson(res, 404, { detail: "Job not found" });
  }

  // 5. Real-Time SSE Transcription Stream
  if (req.method === "GET" && pathname.startsWith("/api/transcribe/")) {
    const jobId = pathname.split("/").pop();
    const job = jobs.get(jobId);
    if (!job || !fs.existsSync(job.filepath)) {
      return sendJson(res, 404, { detail: "Uploaded media session not found." });
    }

    const modelId = parsedUrl.searchParams.get("model") || "base";
    const language = parsedUrl.searchParams.get("language") || "auto";
    const task = parsedUrl.searchParams.get("task") || "transcribe";
    const vadFilter = parsedUrl.searchParams.get("vad_filter") !== "false";

    res.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    const pushEvent = (event, payload) => {
      if (!res.writableEnded) {
        res.write(`event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`);
      }
    };

    req.on("close", () => {
      job.cancelled = true;
    });

    const startTime = Date.now();
    try {
      pushEvent("status", {
        stage: "loading_model",
        message: `Loading '${modelId}' Whisper model (first run downloads weights automatically)...`,
      });

      const transcriber = await getTranscriber(modelId, (msg) => {
        pushEvent("status", { stage: "downloading_model", message: msg });
      });

      if (job.cancelled) {
        pushEvent("cancelled", { message: "Transcription stopped by user." });
        return res.end();
      }

      pushEvent("status", {
        stage: "analyzing",
        message: "Probing audio/video duration with FFmpeg...",
      });

      const totalDuration = await getMediaDuration(job.filepath);
      pushEvent("info", {
        detected_language: language === "auto" ? "AUTO" : language.toUpperCase(),
        language_probability: 100,
        duration: totalDuration,
        task,
        model: modelId,
      });

      const windowSec = 30;
      let offsetSec = 0;
      let segmentCount = 0;
      const effectiveDuration = totalDuration > 0 ? totalDuration : 36000;

      while (offsetSec < effectiveDuration) {
        if (job.cancelled) {
          pushEvent("cancelled", { message: "Transcription stopped by user." });
          break;
        }

        const audioFloat32 = await extractAudioChunkFloat32(
          job.filepath,
          offsetSec,
          windowSec
        );

        if (!audioFloat32 || audioFloat32.length < 1600) {
          break; // Reached end of stream
        }

        const chunkActualSec = audioFloat32.length / 16000;

        // Smart VAD check: skip completely silent 30s windows
        const rms = computeRmsEnergy(audioFloat32);
        if (!vadFilter || rms > 0.0015) {
          const options = {
            return_timestamps: true,
            chunk_length_s: 30,
            stride_length_s: 5,
          };

          if (transcriber.model?.config?.is_multilingual !== false) {
            options.task = task;
            if (language && language !== "auto") {
              options.language = language;
            }
          }

          const result = await transcriber(audioFloat32, options);
          const chunks = Array.isArray(result.chunks) && result.chunks.length > 0
            ? result.chunks
            : result.text
            ? [{ timestamp: [0, chunkActualSec], text: result.text }]
            : [];

          for (const c of chunks) {
            const text = (c.text || "").trim();
            if (!text) continue;

            const relStart = Array.isArray(c.timestamp) && typeof c.timestamp[0] === "number" ? c.timestamp[0] : 0;
            const relEnd =
              Array.isArray(c.timestamp) && typeof c.timestamp[1] === "number"
                ? c.timestamp[1]
                : Math.min(chunkActualSec, relStart + 4);

            const absStart = Number((offsetSec + relStart).toFixed(3));
            const absEnd = Number((offsetSec + Math.max(relStart + 0.2, relEnd)).toFixed(3));
            const elapsed = Number(((Date.now() - startTime) / 1000).toFixed(1));
            const progress =
              totalDuration > 0
                ? Math.min(99.5, Number(((absEnd / totalDuration) * 100).toFixed(1)))
                : 50;

            segmentCount++;
            pushEvent("segment", {
              id: segmentCount,
              start: absStart,
              end: absEnd,
              text,
              progress,
              elapsed,
            });
          }
        }

        offsetSec += windowSec;
        if (chunkActualSec < windowSec - 0.5) {
          break;
        }
      }

      if (!job.cancelled) {
        const totalElapsed = Number(((Date.now() - startTime) / 1000).toFixed(2));
        const speedFactor =
          totalDuration > 0 && totalElapsed > 0
            ? Number((totalDuration / totalElapsed).toFixed(1))
            : 1.0;

        pushEvent("done", {
          total_segments: segmentCount,
          elapsed_seconds: totalElapsed,
          duration: totalDuration,
          speed_factor: speedFactor,
        });
      }
    } catch (err) {
      pushEvent("error", { message: err.message || String(err) });
    } finally {
      try {
        if (fs.existsSync(job.filepath)) {
          fs.unlinkSync(job.filepath);
        }
      } catch (_) {}
      jobs.delete(jobId);
      if (!res.writableEnded) {
        res.end();
      }
    }
    return;
  }

  sendJson(res, 404, { detail: "Not found" });
});

server.listen(PORT, "127.0.0.1", () => {
  const url = `http://127.0.0.1:${PORT}`;
  console.log("==============================================================");
  console.log(`  Whisper Studio is running at: ${url}`);
  console.log("  100% Free & Local • Unlimited File Size • FFmpeg + ONNX");
  console.log("==============================================================");
  if (process.env.NO_BROWSER !== "1") {
    exec(`start "" "${url}"`);
  }
});
