"""Persistent puller: keeps retrying kernel-output file_pattern pulls until
everything lands (survives flaky network). Logs to zbatches/pull_daemon.log.

Pulls:
  - tamil-calls-run: transcripts/.*, manifest.*
  - tamil-model-eval: model_comparison.txt
"""
import datetime as dt, os, sys, time

KER = r'C:\Users\srik2\Desktop\extra\voice_processing'
LOG = os.path.join(KER, 'zbatches', 'pull_daemon.log')
JOBS = [
    ('srikrishnaos/tamil-calls-run', os.path.join(KER, 'transcripts', 'tamil-calls-run'), [r'transcripts/.*', r'manifest.*\.jsonl']),
    ('srikrishnaos/tamil-model-eval', os.path.join(KER, 'transcripts', 'model-eval'), [r'model_comparison\.txt']),
]
MAX_MIN = 240  # give up after 4h

def log(m):
    line = f'[{dt.datetime.now():%H:%M:%S}] {m}'
    print(line, flush=True)
    open(LOG, 'a', encoding='utf-8').write(line + '\n')

from kaggle import api

def count_txt(d):
    return sum(1 for _, _, fs in os.walk(d) for f in fs if f.endswith('.txt')) if os.path.isdir(d) else 0

deadline = time.time() + MAX_MIN * 60
remaining = {(s, p): d for s, d, pats in JOBS for p in pats}
log(f'pull daemon start, jobs={len(remaining)}')
i = 0
while remaining and time.time() < deadline:
    for (slug, pat), dst in list(remaining.items()):
        os.makedirs(dst, exist_ok=True)
        try:
            api.kernels_output(slug, path=dst, file_pattern=pat, quiet=True, force=True)
            n = count_txt(dst)
            log(f'OK {slug} {pat} -> {dst} ({n} txt)')
            remaining.pop((slug, pat))
        except Exception as e:
            i += 1
            if i % 5 == 0:
                log(f'retry#{i} {slug} {pat}: {str(e)[:120]}')
            time.sleep(10)
    if remaining:
        time.sleep(20)
log('DONE' if not remaining else f'GAVE UP on {list(remaining)}')
