# Model notes — Tamil ASR candidates

## Chosen: `vasista22/whisper-tamil-large-v2`

- IIT-Madras Speech Lab fine-tune of `openai/whisper-large-v2`.
- Trained on IISc-MILE Tamil, ULCA, Shrutilipi, Microsoft Speech Corpus (Indian
  languages), FLEURS train+dev, **Babel** — Babel is conversational telephone
  speech, the best public match for call recordings.
- Self-reported: WER 7.5 (FLEURS-Ta test), 6.6 (Common Voice 11). Expect higher
  on real calls; validate with the eval cell before a full run.
- Whisper format → converts to CTranslate2 (`ct2-transformers-converter`) →
  faster-whisper `BatchedInferencePipeline` for batched decode.
- https://huggingface.co/vasista22/whisper-tamil-large-v2

## Alternatives

- `deepdml/whisper-large-v3-turbo-ta-mix-norm` — Whisper turbo (4-layer decoder)
  fine-tuned on CV17 + Kathbath + FLEURS + IISc-MILE + MS Speech Corpus.
  ~28.7 WER CV17 / 4.76 CER. Roughly 2-4x faster decode than large-v2; pick if
  throughput beats accuracy.
  https://huggingface.co/deepdml/whisper-large-v3-turbo-ta-mix-norm
- `ai4bharat/indic-conformer-600m-multilingual` — 25.5 WER FLEURS-Ta (per
  Lemura Labs eval); 31.2 WER on natural IndicVoices speech vs ~78 for vanilla
  Whisper. Gated repo, ONNX + custom code — workable but heavier to batch.
  Note: AI4Bharat points to newer `Indic-Transcribe` as better.
- `lemuralabs/tamil-asr-qwen3` — current FLEURS-Ta #1 (25.27 WER / 7.95 CER),
  but it's a 1.7B LLM decoder via ms-swift with beam=5; much slower per hour of
  audio on T4s and awkward to shard. Revisit only if Whisper quality fails.
- `openai/whisper-large-v3` vanilla — ~48-73% Tamil WER. Do not use.

## Throughput estimate (2x T4, CT2 int8_float16, batch 16)

Published faster-whisper numbers put large-v2 batched decode near ~17s/min of
audio per GPU; VAD skips silence which dominates calls. Expect roughly
0.02-0.05x RTF overall → ~400-1000 min of audio per GPU-hour across both T4s.

## Tuning notes

- Always pass `language="ta"` — Whisper's language detector sometimes flips
  Tamil to Telugu/Malayalam, which corrupts the whole transcript.
- Default Silero VAD parameters clip the first word of chunks (observed in
  published Kaggle T4 work); pipeline uses threshold=0.35,
  min_silence_duration_ms=500, speech_pad_ms=200.
- `int8_float16` on T4: near-fp16 accuracy at ~60% of the VRAM, leaves room for
  batch_size=16.
