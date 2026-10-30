"""Update kernel-metadata.json with currently-uploaded batch datasets + push.

Usage: python tools/push_kernel.py [b001 b002 ...]   (default: all uploaded)
"""
import json, os, subprocess, sys, time

ROOT = r'C:\Users\srik2\Desktop\extra\voice_processing'
RUN = os.path.join(ROOT, 'kaggle_run')
STATE = os.path.join(ROOT, 'zbatches', 'upload_state.txt')

done = set(l.strip() for l in open(STATE)) if os.path.exists(STATE) else set()
batches = sys.argv[1:] or sorted(done)
if not batches:
    print('no uploaded batches to attach'); sys.exit(1)

sources = (['srikrishnaos/voice-processing-code', 'srikrishnaos/tamil-ct2-model']
           + [f'srikrishnaos/tamil-calls-{b}' for b in batches])

meta_path = os.path.join(RUN, 'kernel-metadata.json')
meta = json.load(open(meta_path))
meta['dataset_sources'] = sources
json.dump(meta, open(meta_path, 'w'), indent=1)
print('attaching:', sources)

for attempt in range(1, 6):
    p = subprocess.run(['kaggle', 'kernels', 'push', '-p', RUN],
                       capture_output=True, text=True, timeout=300)
    out = (p.stdout or '') + (p.stderr or '')
    print(f'push attempt {attempt}: {out.strip()[-300:]}', flush=True)
    if p.returncode == 0 and 'error' not in out.lower():
        print('PUSHED — kernel starting'); sys.exit(0)
    time.sleep(min(60, 10 * attempt))
print('push failed'); sys.exit(1)
