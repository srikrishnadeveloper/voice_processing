"""Decode a Kaggle kernel JSON log into readable lines.
Usage: python tools/show_log.py <logfile> [tail_lines]
"""
import json, re, sys

path = sys.argv[1]
tail = int(sys.argv[2]) if len(sys.argv) > 2 else 80

raw = open(path, encoding='utf-8', errors='replace').read()
# log is a JSON array of {stream_name, time, data} — may have leading commas
raw = raw.strip()
if not raw.startswith('['):
    raw = '[' + raw
if not raw.endswith(']'):
    raw = raw + ']'
try:
    events = json.loads(raw)
except json.JSONDecodeError:
    # line-by-line fallback
    events = []
    for m in re.finditer(r'\{[^{}]*"stream_name"[^{}]*\}', raw):
        try:
            events.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass

for e in events[-tail:]:
    d = e.get('data', '')
    d = d.rstrip('\n')
    if d:
        print(f"[{e.get('time', 0):8.1f}] {d}")
