"""Resumable dual-GPU Tamil transcription pipeline.

One worker process per GPU (faster-whisper has no in-process data parallelism).
Each worker appends to its own manifest shard (manifest.gpu<N>.jsonl) so there
are no locks; reruns skip files already present in any manifest shard.

Usage:
    python -m src.transcribe --input-dir /kaggle/input/calls \
        --output-dir /kaggle/working/out --model-dir /kaggle/working/model-ct2
"""

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
import traceback
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma", ".mp4", ".amr", ".3gp"}


def decode(path):
    """Decode to 16 kHz mono float32 ndarray via ffmpeg (avoids the PyAV
    metadata_errors incompatibility in faster-whisper's decoder)."""
    import numpy as np

    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-ar", "16000", "-ac", "1", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(r.stdout, dtype=np.float32)


def find_audio(input_dirs):
    files = []
    for d in input_dirs:
        files.extend(p for p in Path(d).rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    return sorted(files)


def done_set(output_dir):
    done = set()
    for shard in Path(output_dir).glob("manifest*.jsonl"):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def worker(rank, device, files, cfg, out_dir):
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device_index = rank if device == "cuda" else 0
    model = WhisperModel(
        cfg["model"],
        device=device,
        device_index=device_index,
        compute_type=cfg["compute_type"],
    )
    pipe = BatchedInferencePipeline(model=model)
    manifest_path = Path(out_dir) / f"manifest.gpu{rank}.jsonl"
    mf = open(manifest_path, "a", encoding="utf-8")

    for i, path in enumerate(files):
        stem = f"{path.stem}_{abs(hash(str(path))) % 10**8:08d}"
        json_out = Path(out_dir) / "transcripts" / f"{stem}.json"
        txt_out = json_out.with_suffix(".txt")
        if json_out.exists():
            continue
        t0 = time.time()
        record = {"file": str(path), "status": "ok"}
        try:
            audio = decode(path)
            segments, info = pipe.transcribe(
                audio,
                language=cfg["language"],
                batch_size=cfg["batch_size"],
                beam_size=cfg["beam_size"],
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.35,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
            )
            segs = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segments
            ]
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "file": str(path),
                        "duration_sec": round(info.duration, 2),
                        "language": info.language,
                        "model": cfg["model"],
                        "segments": segs,
                    },
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
            txt_out.write_text("\n".join(s["text"] for s in segs), encoding="utf-8")
            record.update(duration_sec=round(info.duration, 2), n_segments=len(segs))
        except Exception as e:
            record.update(status="error", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        record["elapsed_sec"] = round(time.time() - t0, 2)
        mf.write(json.dumps(record, ensure_ascii=False) + "\n")
        mf.flush()
        print(f"[gpu{rank}] {i + 1}/{len(files)} {path.name} -> {record['status']}", flush=True)

    mf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", required=True, help="CT2 model dir or HF model id")
    ap.add_argument("--language", default="ta")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2, help="processes; use 2 for dual T4")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-files", type=int, default=0, help="stop after N files (0 = all)")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop scheduling after N minutes")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = find_audio(args.input_dir)
    done = done_set(out_dir)
    pending = [p for p in all_files if str(p) not in done]
    if args.max_files:
        pending = pending[: args.max_files]
    print(f"found={len(all_files)} done={len(done)} pending={len(pending)}")

    if not pending:
        print("nothing to do")
        return

    cfg = {
        "model": args.model,
        "language": args.language,
        "compute_type": args.compute_type,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
    }

    ctx = mp.get_context("spawn")
    n = max(1, min(args.workers, len(pending)))
    shards = [pending[i::n] for i in range(n)]
    procs = [
        ctx.Process(target=worker, args=(r, args.device, shards[r], cfg, str(out_dir)), daemon=False)
        for r in range(n)
    ]
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    for p in procs:
        p.start()
    for p in procs:
        remaining = max(0, deadline - time.time()) if deadline else None
        p.join(timeout=remaining)
        if p.is_alive():
            print(f"worker hit --max-minutes; terminating (completed files are saved)")
            p.terminate()
            p.join()


if __name__ == "__main__":
    sys.exit(main())
