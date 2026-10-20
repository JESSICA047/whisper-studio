import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0";

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_MAP = {
  tiny: "onnx-community/whisper-tiny",
  base: "onnx-community/whisper-base",
  small: "onnx-community/whisper-small",
  medium: "onnx-community/whisper-small", // fallback to small in browser WASM to prevent OOM
  "large-v3-turbo": "onnx-community/whisper-small",
  "large-v3": "onnx-community/whisper-small",
};

let cachedPipeline = { repo: null, transcriber: null };
let isCancelled = false;

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

async function getTranscriber(modelId) {
  const repo = MODEL_MAP[modelId] || MODEL_MAP.base;
  if (cachedPipeline.repo === repo && cachedPipeline.transcriber) {
    return cachedPipeline.transcriber;
  }

  const fileProgress = {};
  const transcriber = await pipeline("automatic-speech-recognition", repo, {
    dtype: {
      encoder_model: "fp32",
      decoder_model_merged: "q4",
    },
    device: "wasm",
    progress_callback: (info) => {
      if (info.status === "progress" && info.file && typeof info.progress === "number") {
        fileProgress[info.file] = info.progress;
        const values = Object.values(fileProgress);
        const avg = Math.round(values.reduce((a, b) => a + b, 0) / values.length);
        self.postMessage({
          type: "status",
          payload: {
            stage: "downloading_model",
            message: `Downloading '${modelId}' Whisper model to browser cache (${avg}%)...`,
          },
        });
      }
    },
  });

  cachedPipeline = { repo, transcriber };
  return transcriber;
}

self.addEventListener("message", async (event) => {
  const { action, payload } = event.data;

  if (action === "cancel") {
    isCancelled = true;
    return;
  }

  if (action === "transcribe") {
    isCancelled = false;
    const { audioData, model, language, task, vadFilter } = payload;
    const startTime = performance.now();

    try {
      self.postMessage({
        type: "status",
        payload: {
          stage: "loading_model",
          message: `Loading '${model}' model in browser (cached after first run)...`,
        },
      });

      const transcriber = await getTranscriber(model);

      if (isCancelled) {
        self.postMessage({
          type: "cancelled",
          payload: { message: "Transcription stopped by user." },
        });
        return;
      }

      const totalSamples = audioData.length;
      const totalDuration = totalSamples / 16000;

      self.postMessage({
        type: "info",
        payload: {
          detected_language: language === "auto" ? "AUTO" : language.toUpperCase(),
          language_probability: 100,
          duration: totalDuration,
          task,
          model,
        },
      });

      const windowSec = 30;
      const windowSamples = windowSec * 16000;
      let offsetSamples = 0;
      let segmentCount = 0;

      while (offsetSamples < totalSamples) {
        if (isCancelled) {
          self.postMessage({
            type: "cancelled",
            payload: { message: "Transcription stopped by user." },
          });
          return;
        }

        const endSamples = Math.min(totalSamples, offsetSamples + windowSamples);
        const chunkAudio = audioData.subarray(offsetSamples, endSamples);
        const offsetSec = offsetSamples / 16000;
        const chunkActualSec = chunkAudio.length / 16000;

        if (chunkAudio.length < 1600) break;

        const rms = computeRmsEnergy(chunkAudio);
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

          const result = await transcriber(chunkAudio, options);
          const chunks =
            Array.isArray(result.chunks) && result.chunks.length > 0
              ? result.chunks
              : result.text
              ? [{ timestamp: [0, chunkActualSec], text: result.text }]
              : [];

          for (const c of chunks) {
            const text = (c.text || "").trim();
            if (!text) continue;

            const relStart =
              Array.isArray(c.timestamp) && typeof c.timestamp[0] === "number"
                ? c.timestamp[0]
                : 0;
            const relEnd =
              Array.isArray(c.timestamp) && typeof c.timestamp[1] === "number"
                ? c.timestamp[1]
                : Math.min(chunkActualSec, relStart + 4);

            const absStart = Number((offsetSec + relStart).toFixed(3));
            const absEnd = Number((offsetSec + Math.max(relStart + 0.2, relEnd)).toFixed(3));
            const elapsed = Number(((performance.now() - startTime) / 1000).toFixed(1));
            const progress =
              totalDuration > 0
                ? Math.min(99.5, Number(((absEnd / totalDuration) * 100).toFixed(1)))
                : 50;

            segmentCount++;
            self.postMessage({
              type: "segment",
              payload: {
                id: segmentCount,
                start: absStart,
                end: absEnd,
                text,
                progress,
                elapsed,
              },
            });
          }
        } else {
          // Update progress even when skipping silent 30s window
          const absEnd = Number((offsetSec + chunkActualSec).toFixed(3));
          const progress =
            totalDuration > 0
              ? Math.min(99.5, Number(((absEnd / totalDuration) * 100).toFixed(1)))
              : 50;
          self.postMessage({
            type: "progress_only",
            payload: {
              progress,
              elapsed: Number(((performance.now() - startTime) / 1000).toFixed(1)),
            },
          });
        }

        offsetSamples += windowSamples;
      }

      const totalElapsed = Number(((performance.now() - startTime) / 1000).toFixed(2));
      const speedFactor =
        totalDuration > 0 && totalElapsed > 0
          ? Number((totalDuration / totalElapsed).toFixed(1))
          : 1.0;

      self.postMessage({
        type: "done",
        payload: {
          total_segments: segmentCount,
          elapsed_seconds: totalElapsed,
          duration: totalDuration,
          speed_factor: speedFactor,
        },
      });
    } catch (err) {
      self.postMessage({
        type: "error",
        payload: { message: err?.message || String(err) },
      });
    }
  }
});
