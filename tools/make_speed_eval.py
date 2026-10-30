import json, os, shutil

ROOT = r'C:\Users\srik2\Desktop\extra\voice_processing'

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "# Speed/config ablation on T4\n",
        "\n",
        "Same 2-min clip, same model (attached CT2 dataset), different\n",
        "beam_size / batch_size / compute_type. Prints wall-time + RTF for each\n",
        "so we pick the fastest config that keeps quality before 6 batch runs."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "!pip -q install faster-whisper ctranslate2 soundfile"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import glob, subprocess, numpy as np, os, time\n",
        "\n",
        "# datasets mount under /kaggle/input/datasets/<owner>/<slug> — find model dir robustly\n",
        "MODEL = None\n",
        "for _ in range(60):\n",
        "    for d in glob.glob('/kaggle/input/datasets/*/tamil-ct2-model') + glob.glob('/kaggle/input/tamil-ct2-model'):\n",
        "        if os.path.isfile(os.path.join(d, 'model.bin')):\n",
        "            MODEL = d\n",
        "    if MODEL:\n",
        "        break\n",
        "    time.sleep(2)\n",
        "assert MODEL, 'model dataset not mounted'\n",
        "print('model dir:', MODEL)\n",
        "\n",
        "clip = sorted(glob.glob('/kaggle/input/**/*.wav', recursive=True) +\n",
        "              glob.glob('/kaggle/input/**/*.mp3', recursive=True))[0]\n",
        "r = subprocess.run(['ffmpeg','-v','error','-i',clip,'-f','f32le','-acodec','pcm_f32le','-ar','16000','-ac','1','-'],\n",
        "                   capture_output=True, check=True)\n",
        "AUDIO = np.frombuffer(r.stdout, dtype=np.float32)\n",
        "print('clip:', clip, len(AUDIO)/16000, 'sec')"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import time, gc\n",
        "from faster_whisper import WhisperModel, BatchedInferencePipeline\n",
        "# MODEL set by previous cell\n",
        "\n",
        "CONFIGS = [\n",
        "    ('int8_f16 beam5 b16  (current)', 'int8_float16', 5, 16),\n",
        "    ('int8_f16 beam1 b16  (greedy)',  'int8_float16', 1, 16),\n",
        "    ('int8_f16 beam5 b32',            'int8_float16', 5, 32),\n",
        "    ('int8_f16 beam1 b32',            'int8_float16', 1, 32),\n",
        "    ('fp16     beam5 b16',            'float16',      5, 16),\n",
        "]\n",
        "\n",
        "rows = []\n",
        "for label, ct, beam, bs in CONFIGS:\n",
        "    m = WhisperModel(MODEL, device='cuda', compute_type=ct)\n",
        "    pipe = BatchedInferencePipeline(model=m)\n",
        "    t0 = time.time()\n",
        "    segs, info = pipe.transcribe(AUDIO, language='ta', beam_size=beam, batch_size=bs,\n",
        "        vad_filter=True, vad_parameters={'threshold':0.35,'min_silence_duration_ms':500,'speech_pad_ms':200})\n",
        "    nseg, nch, sample = 0, 0, []\n",
        "    for s in segs:\n",
        "        nseg += 1; nch += len(s.text)\n",
        "        if len(sample) < 3: sample.append(s.text.strip())\n",
        "    dt = time.time() - t0\n",
        "    rtf = dt / info.duration\n",
        "    rows.append((label, dt, rtf, nseg, nch, sample))\n",
        "    print(f'{label:28s} {dt:6.1f}s RTF={rtf:.3f} segs={nseg} chars={nch}')\n",
        "    del m, pipe; gc.collect()\n",
        "    try:\n",
        "        import torch; torch.cuda.empty_cache()\n",
        "    except Exception: pass\n",
        "\n",
        "print()\n",
        "for label, dt, rtf, nseg, nch, sample in rows:\n",
        "    print('='*70); print(label, f'RTF={rtf:.3f}'); print(' | '.join(sample)[:400])"]},
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 4}

nb_path = os.path.join(ROOT, 'notebooks', 'tamil_speed_eval.ipynb')
json.dump(nb, open(nb_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

run_dir = os.path.join(ROOT, 'kaggle_speed')
os.makedirs(run_dir, exist_ok=True)
json.dump({
    "id": "srikrishnaos/tamil-speed-eval", "title": "tamil speed eval",
    "code_file": "tamil_speed_eval.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["srikrishnaos/tamil-call-samples", "srikrishnaos/tamil-ct2-model"],
    "competition_sources": [], "kernel_sources": []},
    open(os.path.join(run_dir, 'kernel-metadata.json'), 'w'), indent=1)

shutil.copy(nb_path, os.path.join(run_dir, 'tamil_speed_eval.ipynb'))
print('speed eval ready')
