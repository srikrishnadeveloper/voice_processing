import json, os, shutil

ROOT = r'C:\Users\srik2\Desktop\extra\voice_processing'

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "# Tamil call transcription - batch run\n",
        "\n",
        "Dual-T4 batched pipeline over attached `tamil-calls-*` datasets.\n",
        "Uses pre-converted CT2 model (`tamil-ct2-model`) + pipeline code\n",
        "(`voice-processing-code`). Outputs: `out/transcripts/*.json|.txt` +\n",
        "`out/manifest.gpu*.jsonl` (resumable)."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "!nvidia-smi -L\n",
        "!pip -q install faster-whisper ctranslate2 transformers soundfile"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import glob, os, json\n",
        "\n",
        "MODEL_DIR = '/kaggle/input/tamil-ct2-model'\n",
        "CODE = '/kaggle/input/voice-processing-code/transcribe.py'\n",
        "AUDIO_DIRS = sorted(d for d in glob.glob('/kaggle/input/*')\n",
        "                    if os.path.basename(d).startswith('tamil-calls'))\n",
        "\n",
        "assert os.path.isfile(os.path.join(MODEL_DIR, 'model.bin')), 'attach tamil-ct2-model'\n",
        "assert os.path.isfile(CODE), 'attach voice-processing-code'\n",
        "assert AUDIO_DIRS, 'attach at least one tamil-calls-* dataset'\n",
        "\n",
        "n = sum(len([f for f in os.listdir(d) if not f.endswith('.json')]) for d in AUDIO_DIRS)\n",
        "print('model ok | code ok | audio dirs:', AUDIO_DIRS)\n",
        "print('audio files to process:', n)"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import subprocess, sys\n",
        "\n",
        "args = [sys.executable, CODE]\n",
        "for d in AUDIO_DIRS:\n",
        "    args += ['--input-dir', d]\n",
        "args += ['--output-dir', '/kaggle/working/out',\n",
        "         '--model', MODEL_DIR,\n",
        "         '--workers', '2', '--device', 'cuda',\n",
        "         '--batch-size', '16', '--beam-size', '5',\n",
        "         '--compute-type', 'int8_float16', '--language', 'ta']\n",
        "\n",
        "print(' '.join(args))\n",
        "r = subprocess.run(args)\n",
        "print('exit', r.returncode)"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import glob, json, os\n",
        "txts = sorted(glob.glob('/kaggle/working/out/transcripts/*.txt'))\n",
        "print(f'transcripts written: {len(txts)}')\n",
        "errs = []\n",
        "for m in glob.glob('/kaggle/working/out/manifest*.jsonl'):\n",
        "    for line in open(m, encoding='utf-8'):\n",
        "        r = json.loads(line)\n",
        "        if r.get('status') != 'ok': errs.append(r)\n",
        "print('errors:', len(errs))\n",
        "for e in errs[:10]: print(' ', os.path.basename(e['file']), e.get('error'))\n",
        "if txts:\n",
        "    print('='*60)\n",
        "    print(os.path.basename(txts[0]))\n",
        "    print(open(txts[0], encoding='utf-8').read()[:800])"]},
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 4}

nb_path = os.path.join(ROOT, 'notebooks', 'tamil_calls_run.ipynb')
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

run_dir = os.path.join(ROOT, 'kaggle_run')
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, 'kernel-metadata.json'), 'w') as f:
    json.dump({
        "id": "srikrishnaos/tamil-calls-run", "title": "tamil calls run",
        "code_file": "tamil_calls_run.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["srikrishnaos/voice-processing-code", "srikrishnaos/tamil-ct2-model"],
        "competition_sources": [], "kernel_sources": []}, f, indent=1)

shutil.copy(nb_path, os.path.join(run_dir, 'tamil_calls_run.ipynb'))
print('kernel package ready')
