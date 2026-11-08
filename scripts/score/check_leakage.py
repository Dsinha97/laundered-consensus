"""Offline split audit: IDs, normalized text, n-grams and held-out metadata."""
import argparse
import re
import sys
import unicodedata
from run_watchdogs import read_rows


def normalized(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def grams(words, n):
    n = min(n, len(words))
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)} if words else set()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--heldout', required=True)
    p.add_argument('--conartist')
    p.add_argument('--heldout-families', default='')
    p.add_argument('--heldout-record-types', default='')
    p.add_argument('--ngram-threshold', type=float, default=0.8)
    p.add_argument('--ngram-size', type=int, default=3)
    a = p.parse_args(argv)
    if not 0 < a.ngram_threshold <= 1 or a.ngram_size < 1:
        p.error('threshold must be in (0,1], ngram-size positive')
    train = read_rows(a.train)
    refs = read_rows(a.heldout) + (read_rows(a.conartist) if a.conartist else [])
    if not train or not refs:
        raise ValueError('Nonempty training and reference sets required')
    for r in train + refs:
        if not isinstance(r.get('packet_text'), str) or not normalized(r['packet_text']):
            raise ValueError('Nonempty packet_text required for leakage checking')
    families = {x.strip() for x in a.heldout_families.split(',') if x.strip()}
    types = {x.strip() for x in a.heldout_record_types.split(',') if x.strip()}
    failures = 0
    prepared = [(r, normalized(r['packet_text']), grams(normalized(r['packet_text']), a.ngram_size)) for r in refs]
    for index, row in enumerate(train):
        reasons = []
        if row.get('family') in families:
            reasons.append('heldout_family')
        if row.get('record_type') in types:
            reasons.append('heldout_record_type')
        words = normalized(row['packet_text'])
        ng = grams(words, a.ngram_size)
        for ref, rw, rg in prepared:
            if row['id'] == ref['id']:
                reasons.append('case_id')
            if words == rw:
                reasons.append('normalized_exact')
            # Overlap coefficient detects a held-out passage embedded in longer text.
            if ng and rg and len(ng & rg) / min(len(ng), len(rg)) >= a.ngram_threshold:
                reasons.append('ngram_overlap')
        if reasons:
            failures += 1
            print('Training row ' + str(index + 1) + ': ' + ', '.join(sorted(set(reasons))))
    print(f'Checked {len(train)} training rows against {len(refs)} references; {failures} failing rows.')
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('Leakage check refused (' + type(exc).__name__ + ').', file=sys.stderr)
        sys.exit(2)
