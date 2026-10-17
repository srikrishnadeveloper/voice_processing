# Mistakes learned

(Running log — append whenever something goes wrong or a correction is needed.)

- 2026-10-17: `Shell.Application Folder.CopyHere()` on MTP (phone) items fails
  with "The requested value cannot be determined" and pops a UI dialog that
  blocks automation. Don't retry it blindly — MTP is not a filesystem.
  Reliable options, in order: (1) shell verbs `item.InvokeVerb('copy')` +
  `destFolder.Self.InvokeVerb('paste')`, (2) WPD/`PortableDeviceApiLib` COM
  streaming, (3) Explorer UI via computer-use. Also: enumerate the MTP itemw
  list ONCE into a map — re-scanning a 3,000-item folder per file wastes
  minutes.
- General rule from user: always plan first, pick the correct method once, and
  search the web when information is missing instead of retrying the same
  failing action.

- Stock `whisper-large-v3` is a bad default for low-resource languages; for
  Tamil it loses ~25 WER points to a language-specific fine-tune. Always check
  per-language benchmarks, not the headline model quality.
- faster-whisper has no in-process multi-GPU data parallelism — shard work
  across one process per GPU instead of passing multiple `device_index` values
  and expecting scaling.
- faster-whisper's default VAD settings clip the first word of chunks; relax
  `threshold` / `min_silence_duration_ms` / `speech_pad_ms` for call audio.
