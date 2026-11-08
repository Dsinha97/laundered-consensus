"""Leakage check, run on the laptop only. Sprint work, 6 Nov 2026 or later.

Compares every training row against held-out v2 and against the pilot and crafted-seed packets, by (1) an exact check on
normalised packet text and (2) word 8-gram overlap on the whole packet and on the rule plus proposed change block alone.
Fails closed: a missing file, an unreadable row, or a held-out hash that differs from the expected one exits non-zero and
writes no passing report. The report holds counts, ids and scores, never held-out text.

  python train-modal/leak_check.py --train PATH --heldout PATH --heldout-sha HEX \\
      [--reference PATH ...] --out train-modal/data/leak_report.json [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

N = 8


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def ngrams(text: str, n: int = N) -> set:
    w = norm(text).split(" ")
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def rule_change(packet_text: str) -> str:
    head = packet_text.split("\n\nHAND" "OFFS:", 1)[0]
    return head


def sha_text(t: str) -> str:
    return hashlib.sha256(norm(t).encode("utf-8")).hexdigest()


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def packet_of(r: dict) -> str:
    return r.get("packet_text") or r.get("packet") or ""


def compare(train: list[dict], refs: dict[str, list[dict]], threshold: float) -> dict:
    """refs: label -> rows. Returns colliding training ids with scores (no text)."""
    exact, near = [], []
    for label, rows in refs.items():
        ref_hash = defaultdict(list)
        idx_full, idx_rc = defaultdict(set), defaultdict(set)
        grams_full, grams_rc = {}, {}
        for k, r in enumerate(rows):
            p = packet_of(r)
            ref_hash[sha_text(p)].append(k)
            grams_full[k], grams_rc[k] = ngrams(p), ngrams(rule_change(p))
            for g in grams_full[k]:
                idx_full[g].add(k)
            for g in grams_rc[k]:
                idx_rc[g].add(k)
        for r in train:
            p = packet_of(r)
            if sha_text(p) in ref_hash:
                exact.append({"train_id": r.get("id"), "against": label})
                continue
            for part, idx, grams_ref, g_tr in (("full", idx_full, grams_full, ngrams(p)),
                                               ("rule_change", idx_rc, grams_rc, ngrams(rule_change(p)))):
                if not g_tr:
                    continue
                counts = defaultdict(int)
                for g in g_tr:
                    for k in idx.get(g, ()):
                        counts[k] += 1
                for k, c in counts.items():
                    union = len(g_tr) + len(grams_ref[k]) - c
                    score = c / union if union else 0.0
                    if score >= threshold:
                        near.append({"train_id": r.get("id"), "against": label, "part": part, "jaccard": round(score, 3)})
                        break
    return {"exact": exact, "near": near}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--heldout-sha", required=True)
    ap.add_argument("--reference", action="append", default=[], help="extra packet files to check against (pilot, seeds)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args(argv)
    try:
        held_bytes = Path(a.heldout).read_bytes()
        if hashlib.sha256(held_bytes).hexdigest() != a.heldout_sha.lower():
            print("FAILED: the held-out file's sha256 differs from the expected one", file=sys.stderr)
            return 2
        train = load_jsonl(a.train)
        refs = {"heldout_v2": load_jsonl(a.heldout)}
        for p in a.reference:
            refs[Path(p).name] = load_jsonl(p)
    except (OSError, ValueError) as e:
        print(f"FAILED (fail closed): {type(e).__name__}", file=sys.stderr)
        return 2
    res = compare(train, refs, a.threshold)
    report = {"status": "pass" if not res["exact"] and not res["near"] else "fail",
              "exact_matches": len(res["exact"]), "near_matches": len(res["near"]),
              "exact": res["exact"], "near": res["near"], "threshold": a.threshold, "ngram": N,
              "train_rows": len(train), "train_sha256": hashlib.sha256(Path(a.train).read_bytes()).hexdigest(),
              "heldout_sha256": a.heldout_sha.lower(),
              "references": {k: len(v) for k, v in refs.items()}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"leak check {report['status']}: exact {report['exact_matches']}, near {report['near_matches']}")
    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    sys.exit(main())
