"""Batch checks for generated packets (mirrors Deepayan's review checks, 6 Nov).

python -B check_batch.py out/train_test30_v2.jsonl
"""
import collections, json, re, statistics, sys


def section(text, name, nxt):
    m = re.search(name + r':\s*(.*?)(?:\n\n' + nxt + r':|\Z)', text, re.S)
    return m.group(1).strip() if m else ''


def main(path):
    R = [json.loads(l) for l in open(path, encoding='utf-8')]
    n = len(R)
    print(f'{path}: {n} rows, {len({r["set_id"] for r in R})} sets')
    print('CASE-META lines:', sum('CASE-META' in r['packet_text'] for r in R))

    ct = collections.Counter(r['case_type'] for r in R)
    print('1. mix:', ', '.join(f'{k} {v} ({100*v/n:.0f}%)' for k, v in ct.most_common()))
    sets = collections.defaultdict(list)
    for r in R:
        sets[r['set_id']].append(r)
    kinds = collections.Counter('benign' if v[0]['case_type'] in ('benign', 'plain_violation') else 'matched' for v in sets.values())
    print('   sets:', dict(kinds))

    print('2. near-miss flaws:', dict(collections.Counter(r['detail'] for r in R if r['case_type'] == 'near_miss')))
    print('3. record types (matched rows):', dict(collections.Counter(r['record_type'] for r in R if r['record_type'] != 'none')))
    print('   families:', dict(collections.Counter(r['family'] for r in R)))

    rules = {}
    for v in sets.values():
        rules[v[0]['set_id']] = section(v[0]['packet_text'], 'RULE', 'PROPOSED CHANGE')
    openings = collections.Counter(' '.join(t.split()[:2]).lower() for t in rules.values())
    print(f'4. rule openings: {len(openings)} distinct two-word openings across {len(rules)} sets; top:', openings.most_common(4))

    print('5. median words by case type:', {k: statistics.median(len(r['packet_text'].split()) for r in R if r['case_type'] == k) for k in ct})
    lw = [r for r in R if r['case_type'] == 'laundered']
    for flag in (False, True):
        xs = [len(r['packet_text'].split()) for r in lw if r.get('laundered_with_unrelated') is flag]
        if xs:
            print(f'   laundered {"with unrelated record" if flag else "no record"}: median {statistics.median(xs)} words (n={len(xs)})')

    bad = 0
    for v in sets.values():
        m = [r for r in v if r['case_type'] in ('supported', 'laundered', 'near_miss')]
        if len(m) > 1:
            heads = {r['packet_text'].split('\n\nREC' 'ORDS:')[0] for r in m}
            bad += len(heads) != 1
    print('matched sets with differing rule/change/handoffs:', bad)
    rec = [r for r in R if 'No records attached.' not in r['packet_text']]
    print('rows with a record:', len(rec), 'clear', sum('"clear"' in r['chosen'] for r in rec),
          'flag', sum('"flag"' in r['chosen'] for r in rec))


if __name__ == '__main__':
    main(sys.argv[1])
