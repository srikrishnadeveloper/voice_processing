# voice_processing

Batch transcription of ~4,000 personal call recordings — ~95% native Tamil with
occasional English code-switching — designed to run on **Kaggle (2x NVIDIA T4,
~30 h GPU quota)** using small resumable batches.

All processing is local/offline. No audio or transcripts leave the machine —
important since the recordings contain personal information.

## Model selection

Stock Whisper is **not** good enough for Tamil (~48-73% WER on real speech).
After comparing candidates on public benchmarks and on fit for conversational
telephone speech:

| Model | Tamil WER (FLEURS-Ta) | Notes |
|---|---|---|
| `openai/whisper-large-v3` | ~48.6% | Too weak for Tamil. Ruled out. |
| `ai4bharat/indic-conformer-600m-multilingual` | ~25.5% | Strong, but ONNX custom code + gated repo; heavier to batch. |
| `lemuralabs/tamil-asr-qwen3` | ~25.3% | Best on paper; 1.7B LLM decoder, slow beam search, awkward at scale. |
| **`vasista22/whisper-tamil-large-v2`** | ~7.5% (self-reported) | **Primary.** IIT-Madras Speech Lab fine-tune; trained partly on Babel *telephone* Tamil — closest match to call audio. Whisper format → CTranslate2 speedup. |
| `deepdml/whisper-large-v3-turbo-ta-mix-norm` | ~28.7% (CV17) | Fallback. 4-layer decoder → fastest option; trained on mixed/code-switched data. |

**Primary: `vasista22/whisper-tamil-large-v2` via faster-whisper (CTranslate2).**
Run the built-in eval batch on a sample of your own audio first — self-reported
WER numbers are on read speech, and your calls are conversational. If quality is
disappointing, switch `MODEL_ID` to the turbo-ta checkpoint (faster, still
decent) or the Qwen3-based model (best accuracy, ~5-10x slower).

English words inside Tamil speech are handled: Whisper emits them in Latin
script. Keep `language="ta"` fixed so language detection can't mislabel a clip
as Telugu/Malayalam.

## Pipeline

`src/transcribe.py` is a single-file pipeline:

- **Two GPU workers** — one `WhisperModel` process pinned to each T4
  (`device_index=0/1`). faster-whisper does not do data-parallel across GPUs
  inside one process, so we shard the file list ourselves.
- **Batched decoding** — `BatchedInferencePipeline(batch_size=16)` packs audio
  chunks for ~8x throughput vs sequential.
- **VAD trimming** — Silero VAD skips silence; thresholds are relaxed from
  defaults because the defaults clip the first word of each chunk.
- **Checkpoint/resume** — every finished file is appended to
  `manifest.jsonl`. Re-running skips completed files, so you can work through
  4,000 files in as many Kaggle sessions as needed.
- **Batch windows** — `--max-files` / `--max-minutes` stop cleanly so a session
  can end, upload its outputs, and pick up later.

Per-file output: `out/<name>.json` (segments + metadata) and `out/<name>.txt`
(plain transcript), plus `out/manifest.jsonl` for progress.

## Running on Kaggle

1. Create a Kaggle dataset containing your audio files (upload in chunks, or
   attach multiple datasets — the script scans recursively).
2. Open `notebooks/kaggle_transcribe.ipynb` in a GPU notebook
   (Settings → Accelerator → **GPU T4 x2**).
3. The notebook installs deps, converts the HF checkpoint to CT2 once (cached in
   a Kaggle dataset so later sessions skip conversion), runs the transcriber in
   batches, and writes outputs to `/kaggle/working` → save as a new dataset
   version each session to persist progress.
4. Re-run later sessions with the output dataset attached as input — the
   manifest makes the run resume where it stopped.

Rough throughput estimate: faster-whisper large-v2, `int8_float16`, batch 16,
2x T4 → ~0.02-0.05x real-time. 4,000 calls averaging ~5 min ≈ 20,000 min of
audio ≈ **7-17 h of GPU time**, comfortably inside the 30 h weekly quota with
room for a second pass on failed/low-confidence files.

## Local layout

```
src/transcribe.py              batch transcriber (dual-GPU, resumable)
notebooks/kaggle_transcribe.ipynb  Kaggle driver notebook
requirements.txt
docs/model_notes.md            benchmark sources + tuning notes
```
