"""One-shot status: upload progress, dataset list, kernel statuses, transcripts."""
import os, subprocess, sys

KER = r'C:\Users\srik2\Desktop\extra\voice_processing'

def sh(args, timeout=90):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or '') + (p.stderr or '')
    except Exception as e:
        return f'<err {e}>'

print('=== upload state ===')
sf = os.path.join(KER, 'zbatches', 'upload_state.txt')
print(open(sf).read().strip() if os.path.exists(sf) else '(none yet)')
log = os.path.join(KER, 'zbatches', 'upload.log')
if os.path.exists(log):
    lines = [l for l in open(log, errors='replace').read().splitlines() if l.strip()]
    print('log tail:', lines[-4:] if lines else '(empty)')

print('\n=== datasets ===')
out = sh(['kaggle', 'datasets', 'list', '-m'])
for l in out.splitlines():
    if 'tamil' in l or 'voice' in l or 'ref' in l[:4]:
        print(l[:110])

print('\n=== kernels ===')
for k in ['tamil-calls-run', 'tamil-model-eval']:
    out = sh(['kaggle', 'kernels', 'status', f'srikrishnaos/{k}'])
    print(out.strip().splitlines()[-1] if out.strip() else f'{k}: ?')

print('\n=== local transcripts ===')
td = os.path.join(KER, 'transcripts')
n = sum(1 for _, _, fs in os.walk(td) for f in fs if f.endswith('.txt')) if os.path.isdir(td) else 0
print(n, 'txt files under', td)
