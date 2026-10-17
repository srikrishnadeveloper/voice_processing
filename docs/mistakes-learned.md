# Mistakes learned

(Running log — append whenever something goes wrong or a correction is needed.)

- Stock `whisper-large-v3` is a bad default for low-resource languages; for
  Tamil it loses ~25 WER points to a language-specific fine-tune. Always check
  per-language benchmarks, not the headline model quality.
- faster-whisper has no in-process multi-GPU data parallelism — shard work
  across one process per GPU instead of passing multiple `device_index` values
  and expecting scaling.
- faster-whisper's default VAD settings clip the first word of chunks; relax
  `threshold` / `min_silence_duration_ms` / `speech_pad_ms` for call audio.
