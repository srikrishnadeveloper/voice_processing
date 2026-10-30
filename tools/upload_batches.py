"""Upload all zbatches/* as private Kaggle datasets, with retries.

If a dataset already exists (partial create), pushes a new version instead.
Each dataset dir contains exactly one <batch>.zip + dataset-metadata.json.
"""
import os, subprocess, sys, time

ZROOT = r'C:\Users\srik2\Desktop\extra\voice_processing\zbatches'
STATE = os.path.join(ZROOT, 'upload_state.txt')

def sh(args, timeout=3600):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return (p.returncode, (p.stdout or '') + (p.stderr or ''))

def dataset_exists(slug):
    rc, out = sh(['kaggle', 'datasets', 'list', '-m', '-s', slug])
    return slug in out

def done_set():
    if not os.path.exists(STATE):
        return set()
    return set(l.strip() for l in open(STATE) if l.strip())

def mark_done(b):
    with open(STATE, 'a') as f:
        f.write(b + '\n')

done = done_set()
batches = sorted(d for d in os.listdir(ZROOT)
                 if os.path.isdir(os.path.join(ZROOT, d)) and d.startswith('b'))
print('batches:', batches, '| already done:', sorted(done), flush=True)

for b in batches:
    if b in done:
        print(f'{b}: skip (done)', flush=True)
        continue
    dpath = os.path.join(ZROOT, b)
    slug = f'srikrishnaos/tamil-calls-{b}'
    ok = False
    for attempt in range(1, 9):
        try:
            exists = dataset_exists(slug.split('/')[1])
            if exists:
                rc, out = sh(['kaggle', 'datasets', 'version', '-p', dpath,
                              '-m', 'zip upload', '--delete-old-versions'])
            else:
                rc, out = sh(['kaggle', 'datasets', 'create', '-p', dpath])
            tail = out.strip().splitlines()[-3:] if out.strip() else []
            print(f'{b} attempt {attempt}: rc={rc} :: {tail}', flush=True)
            if 'Dataset is being created' in out or 'Dataset version is being created' in out or rc == 0:
                ok = True
                break
        except subprocess.TimeoutExpired:
            print(f'{b} attempt {attempt}: timeout', flush=True)
        except Exception as e:
            print(f'{b} attempt {attempt}: {e}', flush=True)
        time.sleep(min(60, 5 * attempt))
    if ok:
        mark_done(b)
        print(f'{b}: DONE', flush=True)
    else:
        print(f'{b}: FAILED after retries — stopping', flush=True)
        sys.exit(1)

print('ALL BATCHES UPLOADED', flush=True)
