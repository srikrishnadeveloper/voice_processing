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

## 2026-10-17 — Sample smoke test

**Prompt summary:** User pasted ~10 call recordings into `sample/` and asked
for a minimal Kaggle ipynb that transcribes one short (2-5 min) sample to
verify quality. Then offered short recordings from the phone
(realme GT 7 / Call Recordings) to replace the long ones.

**Response summary:** Indexed all 3,091 phone recordings via Shell COM
extended properties (size + duration), shortlisted ~3.4-min calls across
different speakers. MTP copy failed (CopyHere "requested value cannot be
determined" dialog; InvokeVerb copy/paste also no-op'd; phone then
disconnected from This PC). Pivoted: verified the ASR path locally instead —
installed faster-whisper, trimmed 4 min of a 32-min call with ffmpeg,
transcribed on CPU with pre-converted `ukta-app/whisper-tamil-ct2`
(whisper-tamil-small int8) — produced valid Tamil script (RTF ~2 on CPU).
Created `notebooks/sample_test.ipynb` (find audio → ffprobe durations →
pick shortest/trim to 240s → CT2 convert vasista22/whisper-tamil-large-v2 →
transcribe on GPU → print transcript). Open: phone sample swap pending
reconnect; Copy-MTPFiles module downloaded to tools/ for that.
