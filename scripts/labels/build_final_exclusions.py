"""Build, hash and timestamp the FINAL held-out exclusion list (prepared 7 Nov, waiting on Prashanth's blind marks).

Inputs (swarm-sprint worktree, sprint/heldout_audit/):
  heldout_v2_exclusions.csv  the first-read list of 20 supported-row ids; blind packet numbers 1-20 follow its row order
  rulecheck_codex.md         Codex rule-text reread: packet,STATUS,... with STATUS MISLABEL-STATED / AMBIGUOUS / CLEAR
  marks file                 Prashanth's blind marks, any text with lines like "3: FLAG" or "3,AMBIGUOUS"

Agreed rule (00033-1 style, two readers):
  exclude as "mislabel"  if BOTH readers say the gold "clear" is wrong (Codex MISLABEL-STATED and Prashanth FLAG)
  exclude as "ambiguous" if EITHER reader says AMBIGUOUS, or the two readers disagree
  keep                   if both say CLEAR
Fallback (Deepayan, 7 Nov): if no marks by Sun 06:00 UTC, the main row uses the first 20-id list (258 rows remain).

Usage:
  python -I build_final_exclusions.py <heldout_audit_dir> <out_dir> --marks <marks_file>
  python -I build_final_exclusions.py <heldout_audit_dir> <out_dir> --fallback
Writes FINAL_EXCLUSIONS.csv (id,status,reason) and FINAL_EXCLUSIONS_FREEZE.txt with sha256 and UTC time.
It refuses to overwrite an existing final list.
"""
import argparse, csv, hashlib, os, re, sys
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument('audit_dir'); ap.add_argument('out_dir')
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--marks'); g.add_argument('--fallback', action='store_true')
a = ap.parse_args()

ids = [r['id'] for r in csv.DictReader(open(os.path.join(a.audit_dir, 'heldout_v2_exclusions.csv'), encoding='utf8', newline=''))]
assert len(ids) == 20, len(ids)
codex = {}
for line in open(os.path.join(a.audit_dir, 'rulecheck_codex.md'), encoding='utf8'):
    m = re.match(r'\s*(\d+)\s*,\s*(MISLABEL-STATED|AMBIGUOUS|CLEAR)\s*,', line)
    if m:
        codex[int(m.group(1))] = m.group(2)
assert sorted(codex) == list(range(1, 21)), sorted(codex)

rows = []
if a.fallback:
    mode = 'fallback: first 20-id list (no blind marks by the cutoff)'
    for n, i in enumerate(ids, 1):
        rows.append((i, 'mislabel' if codex[n] == 'MISLABEL-STATED' else 'ambiguous' if codex[n] == 'AMBIGUOUS' else 'excluded_first_list',
                     f'first-read flag; Codex reread {codex[n]}; no second read'))
else:
    mode = 'two readers: Codex rule-text reread + Prashanth blind marks'
    marks = {}
    for line in open(a.marks, encoding='utf8'):
        m = re.search(r'\b(\d{1,2})\b\D{0,12}\b(CLEAR|FLAG|AMBIGUOUS)\b', line, re.I)
        if m and 1 <= int(m.group(1)) <= 20:
            marks[int(m.group(1))] = m.group(2).upper()
    missing = [n for n in range(1, 21) if n not in marks]
    if missing:
        sys.exit(f'Marks missing for packets {missing}; not building a final list.')
    for n, i in enumerate(ids, 1):
        c, p = codex[n], marks[n]
        if c == 'MISLABEL-STATED' and p == 'FLAG':
            rows.append((i, 'mislabel', 'both readers: record fails a stated requirement'))
        elif c == 'AMBIGUOUS' or p == 'AMBIGUOUS':
            rows.append((i, 'ambiguous', f'Codex {c}; Prashanth {p}'))
        elif (c == 'CLEAR') != (p == 'CLEAR'):
            rows.append((i, 'ambiguous', f'readers disagree: Codex {c}; Prashanth {p}'))
        # both CLEAR: kept, not excluded

os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, 'FINAL_EXCLUSIONS.csv')
if os.path.exists(out):
    sys.exit('FINAL_EXCLUSIONS.csv already exists; refusing to overwrite.')
with open(out, 'w', encoding='utf8', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['id', 'status', 'reason'])
    w.writerows(rows)
h = hashlib.sha256(open(out, 'rb').read()).hexdigest()
t = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
note = (f'FINAL held-out exclusion list\nmode: {mode}\nexcluded: {len(rows)} of 278; main row keeps {278 - len(rows)}\n'
        f'file: FINAL_EXCLUSIONS.csv (id,status,reason; holds ids)\nsha256: {h}\nhashed: {t}\n')
open(os.path.join(a.out_dir, 'FINAL_EXCLUSIONS_FREEZE.txt'), 'w', encoding='utf8').write(note)
print(note)
