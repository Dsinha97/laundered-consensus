"""Freeze the clean round-7 rows (7 Nov) as an outside test set.

Keeps a row only if the Codex check says its label is OK AND its mechanism is present. Case type and gold come from
the job. Per the team decision (Deepayan, 7 Nov 15:27), Invent sets are outside test only: never used for training.
Usage: python -I assemble_v7.py <invent_probe_dir>
"""
import csv, hashlib, json, os, re, sys
from datetime import datetime, timezone

D = sys.argv[1]
V = os.path.join(D, 'v7')
CASE = {'silent_record': ('near_miss', 'flag'), 'explained_gap_flag': ('laundered', 'flag'), 'explained_gap_clear': ('supported', 'clear'),
        'described_absent': ('laundered', 'flag'), 'described_present': ('supported', 'clear'),
        'notes_only_flag': ('laundered', 'flag'), 'notes_only_clear': ('benign', 'clear')}
key = [json.loads(l) for l in open(os.path.join(V, 'labelcheck_v7_key.jsonl'), encoding='utf8')]
table = {}
for l in open(os.path.join(V, 'labelcheck_codex_v7.md'), encoding='utf8'):
    m = re.match(r'\|\s*(\d+)\s*\|', l)
    if m:
        table[int(m.group(1))] = [c.strip() for c in l.strip().strip('|').split('|')]


def flat(s):
    try:
        c = json.loads(s)
    except Exception:
        return {}
    if isinstance(c, dict) and 'verdict' in c:
        return c
    if isinstance(c, dict):
        for v in c.values():
            if isinstance(v, dict) and 'verdict' in v:
                return v
    return {}


csvs = {j: list(csv.DictReader(open(os.path.join(V, j + '.csv'), encoding='utf8', newline=''))) for j in CASE}
out_rows = []
for k in key:
    c = table.get(k['row'])
    if c and (c[2] != 'OK' or c[3] != 'PRESENT'):
        continue
    r = csvs[k['job']][k['j'] - 1]
    case, gold = CASE[k['job']]
    ch, rj = flat(r['chosen']), flat(r['rejected'])
    out_rows.append({'id': f"inv7-{len(out_rows) + 1:03d}", 'src': 'v7', 'mechanism': k['job'], 'case_type': case, 'gold': gold,
                     'prompt': r['enhanced_prompt'],
                     'chosen': {'verdict': gold, 'evidence': ch.get('evidence', '')},
                     'rejected': {'verdict': 'clear' if gold == 'flag' else 'flag', 'evidence': rj.get('evidence', '')}})

out = os.path.join(D, 'combined_v7')
os.makedirs(out, exist_ok=True)
p = os.path.join(out, 'invent_v7_test_DO_NOT_TRAIN.jsonl')
with open(p, 'w', encoding='utf8', newline='\n') as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
counts = {}
for r in out_rows:
    counts[r['mechanism']] = counts.get(r['mechanism'], 0) + 1
m = {'built_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'), 'rows': len(out_rows),
     'clear_flag': [sum(r['gold'] == 'clear' for r in out_rows), sum(r['gold'] == 'flag' for r in out_rows)],
     'by_mechanism': counts, 'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest(),
     'use': 'outside test only; never train on it (team decision 7 Nov)'}
json.dump(m, open(os.path.join(out, 'MANIFEST.json'), 'w', encoding='utf8'), indent=2)
print(json.dumps(m, indent=2))
