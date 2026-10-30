import json, os, shutil

ROOT = r'C:\Users\srik2\Desktop\extra\voice_processing'

cells = [
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import os, time\n",
        "for i in range(6):\n",
        "    print(f'--- t+{i*10}s ---')\n",
        "    for root, dirs, files in os.walk('/kaggle/input'):\n",
        "        print(root, '->', files[:10])\n",
        "    time.sleep(10)"]},
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 4}

nb_path = os.path.join(ROOT, 'notebooks', 'mount_probe.ipynb')
json.dump(nb, open(nb_path, 'w'), indent=1)

run_dir = os.path.join(ROOT, 'kaggle_probe')
os.makedirs(run_dir, exist_ok=True)
json.dump({
    "id": "srikrishnaos/mount-probe", "title": "mount probe",
    "code_file": "mount_probe.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
    "enable_internet": False,
    "dataset_sources": ["srikrishnaos/tamil-call-samples", "srikrishnaos/tamil-ct2-model",
                        "srikrishnaos/voice-processing-code", "srikrishnaos/tamil-calls-b001"],
    "competition_sources": [], "kernel_sources": []},
    open(os.path.join(run_dir, 'kernel-metadata.json'), 'w'), indent=1)

shutil.copy(nb_path, os.path.join(run_dir, 'mount_probe.ipynb'))
print('probe ready')
