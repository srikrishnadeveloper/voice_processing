# Session log

## 2026-10-17 — Repo setup + Tamil ASR model selection + Kaggle pipeline

**Prompt summary:** Configure the folder as a GitHub repo
(`srikrishnadeveloper/voice_processing`). User is a student with ~4,000 call
recordings, ~95% native Tamil with occasional English. Wants the best
Whisper-class model for accurate transcription (audio is personal → must run
locally). Plans to run on Kaggle's 2x T4 GPUs / 30 h quota, in small batches,
with CUDA/CPU optimizations. Asked to find the best model first.

**Response summary:** Researched Tamil ASR candidates; vanilla
whisper-large-v3 ruled out (~48-73% Tamil WER). Chose
`vasista22/whisper-tamil-large-v2` (IIT-M fine-tune incl. Babel telephone
Tamil, Whisper format → CTranslate2 speedup). Built scaffold: README with
model comparison, `src/transcribe.py` (one faster-whisper worker per GPU,
BatchedInferencePipeline, Silero VAD with relaxed params, manifest-based
resume, --max-files/--max-minutes batch windows), `notebooks/kaggle_transcribe.ipynb`
(install → clone → CT2 convert → 4-file eval → full run → persist outputs as
Kaggle dataset), requirements, model notes doc. Initialized git, committed,
pushed main to origin.
