"""Assemble the combined Invent set (7 Nov) from label-checked rounds v3-v6.

Rules (see ITERATION_LOG.md):
- Keep only rows whose Codex label check is OK.
- Case type and gold come from the job, never from Invent's tags.
- Test slice: about one third of each mechanism's FIRST clean batch.
  * Base case types (laundered, supported, near_miss, benign_clear, benign_flag): first batches are v3 + v4.
  * Traps (format, reverse, boundary, decoy): first batch is v5; only rows with the trap present can go to test.
  * v6 near-miss rows came after a prompt fix, so they go to training only.
- Each row keeps its source, mechanism, flaw list and chosen/rejected answers.
- The earlier Invent held-out set (172 rows) is separate and never mixed in.
Run: python -I assemble_combined.py <invent_probe_dir>
"""
import csv, hashlib, json, os, random, re, sys
from datetime import datetime, timezone

D = sys.argv[1]
SEED = 20261107
rows = []


def flat(s):
    """Return the chosen/rejected answer as a flat dict, unwrapping Invent's nesting."""
    try:
        c = json.loads(s)
    except Exception:
        c = None
    if isinstance(c, dict):
        if 'verdict' in c:
            return c
        for v in c.values():
            if isinstance(v, dict) and 'verdict' in v:
                return v
            if isinstance(v, str):
                try:
                    x = json.loads(v)
                    if isinstance(x, dict) and 'verdict' in x:
                        return x
                except Exception:
                    pass
        if 'chosen_verdict' in c:
            return {'verdict': c['chosen_verdict'], 'evidence': c.get('chosen_evidence', '')}
    return None


def bad_rows(md, col_label=2):
    """Row numbers whose label column is not OK in a Codex check table."""
    out = set()
    for l in open(md, encoding='utf8'):
        m = re.match(r'\|\s*(\d+)\s*\|', l)
        if m:
            cells = [c.strip() for c in l.strip().strip('|').split('|')]
            if cells[col_label - 1] in ('MISLABEL', 'AMBIGUOUS'):
                out.add(int(m.group(1)))
    return out


def table(md):
    """Row number -> list of cells from a Codex check table."""
    t = {}
    for l in open(md, encoding='utf8'):
        m = re.match(r'\|\s*(\d+)\s*\|', l)
        if m:
            t[int(m.group(1))] = [c.strip() for c in l.strip().strip('|').split('|')]
    return t


def answers(r):
    ch, rj = flat(r['chosen']), flat(r['rejected'])
    return ch, rj


def add(src, job, mech, case, gold, prompt, ch, rj, flaws='', trap=None, first_batch=False):
    if ch is not None and ch.get('verdict') not in (None, gold):
        return  # generator's own answer contradicts the job; the label check may still pass, but the chosen text would be wrong
    rows.append({
        'src': src, 'job': job, 'mechanism': mech, 'case_type': case, 'gold': gold,
        'trap_present': trap, 'all_flaws': flaws, 'first_batch': first_batch,
        'prompt': prompt,
        'chosen': {'verdict': gold, 'evidence': (ch or {}).get('evidence', '')},
        'rejected': {'verdict': 'clear' if gold == 'flag' else 'flag', 'evidence': (rj or {}).get('evidence', '')},
    })


# v3 probe (one mixed job)
bad3 = bad_rows(os.path.join(D, 'labelcheck_codex_v3.md'))
v3csv = list(csv.DictReader(open(os.path.join(D, 'invent_probe_v3_50.csv'), encoding='utf8', newline='')))
for i, l in enumerate(open(os.path.join(D, 'invent_probe_v3_50_norm.jsonl'), encoding='utf8'), 1):
    if i in bad3:
        continue
    n = json.loads(l)
    ch, rj = answers(v3csv[i - 1])
    case = n['case_type']
    mech = 'benign_' + n['gold'] if case == 'benign' else case
    add('v3', 'v3_probe', mech, case, n['gold'], n['prompt'], ch, rj, first_batch=True)

# v4 (one job per case type); clean ids and retro flaw lists
clean4 = {json.loads(l)['id'] for l in open(os.path.join(D, 'v4', 'v4_clean_352.jsonl'), encoding='utf8')}
flaws = {json.loads(l)['id']: json.loads(l)['all_flaws'] for l in open(os.path.join(D, 'allflaws_retro_joined.jsonl'), encoding='utf8')}
exp = [('laundered', 'laundered', 'flag'), ('supported', 'supported', 'clear'), ('near_miss', 'near_miss', 'flag'),
       ('benign_clear', 'benign', 'clear'), ('benign_flag', 'benign', 'flag')]
i = 0
for job, case, gold in exp:
    for r in csv.DictReader(open(os.path.join(D, 'v4', f'v4_{job}.csv'), encoding='utf8', newline='')):
        i += 1
        rid = f'v4-{i:03d}'
        if rid not in clean4:
            continue
        ch, rj = answers(r)
        add('v4', 'v4_' + job, job, case, gold, r['enhanced_prompt'], ch, rj, flaws.get(rid, ''), first_batch=True)

# v5 traps (all supported, gold clear)
v5 = os.path.join(D, 'v5')
checks = [('reverse', 'labelcheck_codex_v5a.md', 0), ('boundary', 'labelcheck_codex_v5a.md', 30),
          ('format', 'labelcheck_codex_v5b.md', 0), ('decoy', 'labelcheck_codex_v5c.md', 0)]
for trap, md, off in checks:
    t = table(os.path.join(v5, md))
    for j, r in enumerate(csv.DictReader(open(os.path.join(v5, f'{trap}.csv'), encoding='utf8', newline='')), 1):
        cells = t.get(j + off)
        label = cells[2] if (cells and md.endswith('v5a.md')) else (cells[1] if cells else 'OK')
        if label != 'OK':
            continue
        present = 'ABSENT' not in (cells or [])
        ch, rj = answers(r)
        add('v5', 'v5_' + trap, trap if present else 'supported', 'supported', 'clear', r['enhanced_prompt'], ch, rj,
            trap=present, first_batch=present)

# v6 near-miss by flaw (training only: made after a prompt fix)
v6 = os.path.join(D, 'v6')
retro_naa = {k: v for k, v in flaws.items() if k.startswith('v6-naa-')}
for flaw, md, off in [('not_an_approval', 'labelcheck_codex_v6_naa.md', 0), ('wrong_approver', 'labelcheck_codex_v6_wa.md', 0),
                      ('wrong_limit', 'labelcheck_codex_v6_ls.md', 0), ('wrong_scope', 'labelcheck_codex_v6_ls.md', 30)]:
    t = table(os.path.join(v6, md))
    for j, r in enumerate(csv.DictReader(open(os.path.join(v6, f'{flaw}.csv'), encoding='utf8', newline='')), 1):
        cells = t.get(j + off)
        if cells and cells[1] != 'OK':
            continue
        if flaw == 'not_an_approval':
            fl = retro_naa.get(f'v6-naa-{j:02d}', '')
        else:
            fl = cells[3] if cells and len(cells) > 3 else flaw
        ch, rj = answers(r)
        add('v6', 'v6_' + flaw, 'near_miss', 'near_miss', 'flag', r['enhanced_prompt'], ch, rj, fl, first_batch=False)

# Split: one third of each mechanism's first clean batch goes to test.
rng = random.Random(SEED)
for k, r in enumerate(rows, 1):
    r['id'] = f'inv-{k:04d}'
    r['split'] = 'train'
by = {}
for r in rows:
    if r['first_batch']:
        by.setdefault(r['mechanism'], []).append(r)
for mech, rs in sorted(by.items()):
    rs = sorted(rs, key=lambda x: x['id'])
    rng.shuffle(rs)
    for r in rs[:round(len(rs) / 3)]:
        r['split'] = 'test'

# Near-duplicates: 5-word-shingle Jaccard. >= 0.9 counts as a copy (keep the first only);
# > 0.5 counts as related, and a related pair split across train/test moves the test row to train.
def sh(s):
    w = re.findall(r'\w+', s.lower())
    return {' '.join(w[i:i + 5]) for i in range(len(w) - 4)}


S = [sh(r['prompt']) for r in rows]
dropped, moved = [], []
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        a, b = S[i], S[j]
        if not (a and b) or rows[j].get('drop'):
            continue
        x = len(a & b) / len(a | b)
        if x >= 0.9:
            rows[j]['drop'] = True
            dropped.append((rows[j]['id'], rows[i]['id'], round(x, 2)))
        elif x > 0.5 and rows[i]['split'] != rows[j]['split']:
            t = rows[i] if rows[i]['split'] == 'test' else rows[j]
            t['split'] = 'train'
            moved.append((t['id'], round(x, 2)))
rows = [r for r in rows if not r.get('drop')]
for r in rows:
    r['has_answers'] = bool(r['chosen']['evidence'] and r['rejected']['evidence'])
    r['rejected_loaded'] = any(w in r['rejected']['evidence'].lower() for w in ('careless', 'suspicious', 'collusion', 'obviously'))

out = os.path.join(D, 'combined')
os.makedirs(out, exist_ok=True)
paths = {}
for split in ('train', 'test'):
    p = os.path.join(out, f'invent_combined_{split}.jsonl')
    with open(p, 'w', encoding='utf8', newline='\n') as f:
        for r in rows:
            if r['split'] == split:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    paths[split] = p

summary = {}
for r in rows:
    key = f"{r['split']}|{r['mechanism']}|{r['gold']}"
    summary[key] = summary.get(key, 0) + 1
manifest = {
    'built_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
    'seed': SEED,
    'counts': {s: sum(1 for r in rows if r['split'] == s) for s in ('train', 'test')},
    'near_copies_dropped': dropped,
    'related_rows_moved_test_to_train': moved,
    'rows_missing_chosen_or_rejected_text': sum(1 for r in rows if not r['has_answers']),
    'rows_with_loaded_rejected_wording': sum(1 for r in rows if r['rejected_loaded']),
    'clear_flag': {s: [sum(1 for r in rows if r['split'] == s and r['gold'] == g) for g in ('clear', 'flag')] for s in ('train', 'test')},
    'by_split_mechanism_gold': dict(sorted(summary.items())),
    'sha256': {s: hashlib.sha256(open(p, 'rb').read()).hexdigest() for s, p in paths.items()},
}
json.dump(manifest, open(os.path.join(out, 'MANIFEST.json'), 'w', encoding='utf8'), indent=2)
print(json.dumps(manifest, indent=2))
