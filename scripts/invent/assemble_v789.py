"""Freeze one outside-test set from Invent rounds 7, 8 and 9 (7 Nov).

Keeps every row whose Codex check says label OK and mechanism present. The round-7 frozen file
(invent_v7_test_DO_NOT_TRAIN.jsonl, 4bd25dbe...) is included as is and stays frozen separately.
Near-copies (5-word-shingle Jaccard >= 0.9) are dropped. Outside test only (team decision 7 Nov): never train on it.
Usage: python -I assemble_v789.py <invent_probe_dir>
"""
import csv, hashlib, json, os, re, sys
from datetime import datetime, timezone

D = sys.argv[1]
CASE = {'silent_record': ('near_miss', 'flag'), 'explained_gap_flag': ('laundered', 'flag'), 'explained_gap_clear': ('supported', 'clear'),
        'described_absent': ('laundered', 'flag'), 'described_present': ('supported', 'clear'),
        'notes_only_flag': ('laundered', 'flag'), 'notes_only_clear': ('benign', 'clear')}


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


def table(md):
    t = {}
    for l in open(md, encoding='utf8'):
        m = re.match(r'\|\s*(\d+)\s*\|', l)
        if m:
            t[int(m.group(1))] = [c.strip() for c in l.strip().strip('|').split('|')]
    return t


rows = []
# round 7, already frozen
for l in open(os.path.join(D, 'combined_v7', 'invent_v7_test_DO_NOT_TRAIN.jsonl'), encoding='utf8'):
    r = json.loads(l)
    r['round'] = 'v7'
    rows.append(r)


def add(round_, job, r):
    case, gold = CASE[job]
    ch, rj = flat(r['chosen']), flat(r['rejected'])
    rows.append({'id': None, 'src': 'invent', 'round': round_, 'mechanism': job, 'case_type': case, 'gold': gold, 'prompt': r['enhanced_prompt'],
                 'chosen': {'verdict': gold, 'evidence': ch.get('evidence', '')},
                 'rejected': {'verdict': 'clear' if gold == 'flag' else 'flag', 'evidence': rj.get('evidence', '')}})


def from_key(round_, key_file, codex_md, label_col=2, mech_col=3):
    key = [json.loads(l) for l in open(key_file, encoding='utf8')]
    t = table(codex_md)
    cache = {}
    for k in key:
        c = t.get(k['row'])
        if c and (c[label_col] != 'OK' or c[mech_col] not in ('PRESENT', 'Yes')):
            continue
        path = k.get('csv') or os.path.join(D, 'v8', k['job'] + '.csv')
        if path not in cache:
            cache[path] = list(csv.DictReader(open(path, encoding='utf8', newline='')))
        add(round_, k['job'], cache[path][k['j'] - 1])


# round 8 re-probes (explained_gap_flag/clear, described_present) and notes_only_clear re-probe
from_key('v8', os.path.join(D, 'v8', 'labelcheck_v8a_key.jsonl'), os.path.join(D, 'v8', 'labelcheck_codex_v8a.md'))
t = table(os.path.join(D, 'v8', 'labelcheck_codex_v8b.md'))
for j, r in enumerate(csv.DictReader(open(os.path.join(D, 'v8', 'notes_only_clear.csv'), encoding='utf8', newline='')), 1):
    c = t.get(j)
    if c and (c[1] != 'OK' or c[2] not in ('Yes', 'PRESENT')):
        continue
    add('v8', 'notes_only_clear', r)
# round 8 scale-ups and round 9
for job in ('silent_record', 'described_absent', 'notes_only_flag', 'notes_only_clear'):
    from_key('v8scale', os.path.join(D, 'v8', f'check_{job}_key.jsonl'), os.path.join(D, 'v8', f'check_{job}_codex.md'))
from_key('v9', os.path.join(D, 'v9', 'check_v9_key.jsonl'), os.path.join(D, 'v9', 'check_v9_codex.md'))


def sh(s):
    w = re.findall(r'\w+', s.lower())
    return {' '.join(w[i:i + 5]) for i in range(len(w) - 4)}


S = [sh(r['prompt']) for r in rows]
drop = set()
for i in range(len(rows)):
    if i in drop:
        continue
    for j in range(i + 1, len(rows)):
        if j not in drop and S[i] and S[j] and len(S[i] & S[j]) / len(S[i] | S[j]) >= 0.9:
            drop.add(j)
rows = [r for i, r in enumerate(rows) if i not in drop]
for n, r in enumerate(rows, 1):
    r['v789_id'] = f'inv789-{n:04d}'

out = os.path.join(D, 'combined_v789')
os.makedirs(out, exist_ok=True)
p = os.path.join(out, 'invent_v789_test_DO_NOT_TRAIN.jsonl')
with open(p, 'w', encoding='utf8', newline='\n') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
by = {}
for r in rows:
    by[r['mechanism']] = by.get(r['mechanism'], 0) + 1
m = {'built_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'), 'rows': len(rows), 'near_copies_dropped': len(drop),
     'clear_flag': [sum(r['gold'] == 'clear' for r in rows), sum(r['gold'] == 'flag' for r in rows)],
     'by_mechanism': dict(sorted(by.items())), 'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest(),
     'includes': 'round-7 frozen set (156 rows, 4bd25dbe) plus usable rows from round 8 re-probes, round 8 scale-ups and round 9',
     'use': 'outside test only; never train on it (team decision 7 Nov)'}
json.dump(m, open(os.path.join(out, 'MANIFEST.json'), 'w', encoding='utf8'), indent=2)
print(json.dumps(m, indent=2))
