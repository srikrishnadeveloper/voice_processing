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

## 2026-10-29 — Kaggle API end-to-end run

**Prompt summary:** User gave a Kaggle API token (KGAT_…, saved to
`~/.kaggle/access_token`) and asked me to run the whole thing autonomously
and report the transcript.

**Response summary:** Uploaded a 2-min clip (`sample/sample_2min.wav`, trimmed
from `Hari Roomate SSN-2610170944.mp3`) as private dataset
`srikrishnaos/tamil-call-samples`. Pushed `sample_test.ipynb` as GPU kernel
`srikrishnaos/tamil-sample-test`. Took 4 kernel versions: v1 tokenizer.json
missing in --copy_files; v2 removed flag but converter itself requires
tokenizer.json → v3 snapshot_download + AutoTokenizer.save_pretrained then
convert local dir (worked); v3 then hit PyAV `metadata_errors` TypeError
(PyAV ≥15) → v4 decode via soundfile→ndarray. COMPLETE — transcript verified
valid conversational Tamil. Fetched transcript.txt via signed-URL API call
(bulk output download kept dying on user's flaky network). Learned: don't
leave HF snapshot dirs under /kaggle/working — they become kernel output and
add ~6GB to output downloads.

## 2026-10-29 — Full phone dump + batch upload/run

**Prompt summary:** User asked to pull all call recordings off the phone and
upload/process in parallel small batches. Later unplugged the phone mid-way
("do for what i have") — luckily the copy had already finished.

**Response summary:** Retried MTP with Shell COM CopyHere **without flags**
(previously used 0x14 FOF flags triggered the "requested value cannot be
determined" dialog — the flags were the bug). Single folder-level CopyHere of
`Call Recordings` copied ALL 3,253 files (1,186 MB, 0 zero-byte, ffprobe
verified) to `recordings/Call Recordings` in ~10 min. Actual count 3,253 vs
indexed 3,091 (new recordings since indexing). Avg call ~2 min → ~112 h
audio total ≈ 3-4 GPU-h on dual T4.

Architecture for bulk run (avoids GitHub push dependency + per-run convert):
- `srikrishnaos/voice-processing-code` dataset = src/transcribe.py (private)
- `srikrishnaos/tamil-ct2-model` dataset = pre-converted CT2 int8_float16
  model (1.5 GB, reused from sample-test output — saves ~10 min convert/run)
- `srikrishnaos/tamil-calls-b001..b006` datasets = 6 batches × ~550 files
  (~200 MB each), hardlinked locally (no extra disk)
- Kernel `srikrishnaos/tamil-calls-run` attaches code+model+current batch;
  metadata updated per batch push. transcribe.py unchanged.
