"""Pull transcript outputs from a finished kernel run via file_pattern API
(one zip per pattern = robust on flaky networks). Merges into transcripts/.

Usage: python tools/pull_outputs.py [kernel-slug]
"""
import os, sys, time

KER = r'C:\Users\srik2\Desktop\extra\voice_processing'
slug = sys.argv[1] if len(sys.argv) > 1 else 'srikrishnaos/tamil-calls-run'
outdir = os.path.join(KER, 'transcripts', slug.split('/')[-1])
os.makedirs(outdir, exist_ok=True)

from kaggle import api

def pull(pattern, tries=10):
    for i in range(tries):
        try:
            api.kernels_output(slug, path=outdir, file_pattern=pattern, quiet=True, force=True)
            print(f'pulled {pattern} -> {outdir}', flush=True)
            return True
        except Exception as e:
            print(f'  {pattern} try{i+1}: {str(e)[:140]}', flush=True)
            time.sleep(min(45, 5 * (i + 1)))
    return False

ok = pull(r'transcripts/.*')
ok &= pull(r'manifest.*\.jsonl')
n = sum(1 for _, _, fs in os.walk(outdir) for f in fs if f.endswith('.txt'))
print(f'transcript txt files now in {outdir}: {n}', flush=True)
