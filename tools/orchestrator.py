"""Autonomous batch orchestrator.

Loop until all batches processed or 05:00:
  1. read zbatches/upload_state.txt -> uploaded batch ids
  2. if new uploads not yet attached to a kernel run -> push tamil-calls-run
     with code+model+ALL uploaded-but-unprocessed batches attached
  3. poll kernel: COMPLETE -> pull transcripts+manifests, mark batches processed
                  ERROR    -> log, mark for one retry
  4. repeat every ~90s

State file: zbatches/orch_state.json {attached:[], processed:[], pushed_run:bool}
Log: zbatches/orch.log
"""
import datetime as dt
import json, os, subprocess, sys, time

KER = r'C:\Users\srik2\Desktop\extra\voice_processing'
ZB = os.path.join(KER, 'zbatches')
UP_STATE = os.path.join(ZB, 'upload_state.txt')
STATE_F = os.path.join(ZB, 'orch_state.json')
LOG = os.path.join(ZB, 'orch.log')
RUN_DIR = os.path.join(KER, 'kaggle_run')
KERNEL = 'srikrishnaos/tamil-calls-run'
ALL = [f'b{i:03d}' for i in range(1, 7)]
CUTOFF = dt.time(5, 30)  # stop starting new work after 05:30

def log(msg):
    line = f'[{dt.datetime.now():%H:%M:%S}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def sh(args, timeout=240):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, ((p.stdout or '') + (p.stderr or ''))
    except Exception as e:
        return -1, f'<exc {e}>'

def uploaded():
    if not os.path.exists(UP_STATE):
        return []
    return sorted(set(l.strip() for l in open(UP_STATE) if l.strip()))

def load_state():
    if os.path.exists(STATE_F):
        return json.load(open(STATE_F))
    return {'attached': [], 'processed': [], 'run_active': False, 'retries': {}}

def save_state(s):
    json.dump(s, open(STATE_F, 'w'), indent=1)

def kernel_status():
    for _ in range(3):
        rc, out = sh(['kaggle', 'kernels', 'status', KERNEL])
        for st in ['COMPLETE', 'ERROR', 'RUNNING', 'QUEUED', 'CANCELED']:
            if st in out:
                return st
        time.sleep(10)
    return 'UNKNOWN'

def push(batches):
    meta = json.load(open(os.path.join(RUN_DIR, 'kernel-metadata.json')))
    meta['dataset_sources'] = (['srikrishnaos/voice-processing-code',
                              'srikrishnaos/tamil-ct2-model']
                             + [f'srikrishnaos/tamil-calls-{b}' for b in batches])
    json.dump(meta, open(os.path.join(RUN_DIR, 'kernel-metadata.json'), 'w'), indent=1)
    for i in range(5):
        rc, out = sh(['kaggle', 'kernels', 'push', '-p', RUN_DIR], timeout=300)
        if rc == 0 and 'successfully pushed' in out:
            log(f'pushed kernel with {batches}')
            return True
        log(f'push fail try{i+1}: {out.strip()[-160:]}')
        time.sleep(min(60, 10 * (i + 1)))
    return False

def pull():
    dst = os.path.join(KER, 'transcripts', 'tamil-calls-run')
    os.makedirs(dst, exist_ok=True)
    ok = True
    for pat in [r'transcripts/.*', r'manifest.*\.jsonl']:
        good = False
        for i in range(10):
            rc, out = sh(['python', '-c',
                'import sys; from kaggle import api; '
                f'api.kernels_output("{KERNEL}", path=r"{dst}", file_pattern=r"{pat}", quiet=True, force=True)'],
                timeout=900)
            if rc == 0:
                good = True; break
            log(f'pull {pat} try{i+1} fail: {out.strip()[-140:]}')
            time.sleep(min(45, 5 * (i + 1)))
        ok &= good
    return ok

def main():
    log('orchestrator start')
    while True:
        # runs until all batches processed — no time cutoff
        s = load_state()
        up = uploaded()
        pending_batches = [b for b in up if b not in s['processed']]
        log(f"uploaded={up} processed={s['processed']} active={s['run_active']}")

        if s['run_active']:
            st = kernel_status()
            log(f'kernel status: {st}')
            if st == 'COMPLETE':
                if pull():
                    s['processed'] = sorted(set(s['processed']) | set(s['attached']))
                    s['attached'], s['run_active'] = [], False
                    save_state(s)
                    log(f"processed -> {s['processed']}")
                else:
                    log('pull failed; will retry pull next loop')
            elif st in ('ERROR', 'CANCELED'):
                b = tuple(s['attached'])
                s['retries'][str(b)] = s['retries'].get(str(b), 0) + 1
                s['attached'], s['run_active'] = [], False
                save_state(s)
                log(f'kernel {st}; retry count for that set = {s["retries"][str(b)]}')
            time.sleep(90)
            continue

        if pending_batches:
            if push(pending_batches):
                s['attached'], s['run_active'] = pending_batches, True
                save_state(s)
            time.sleep(60)
            continue

        if len(s['processed']) == len(ALL):
            log('ALL BATCHES PROCESSED — done'); return
        time.sleep(90)

if __name__ == '__main__':
    main()
