"""Build the evaluation input from held-out v2. Sprint work, 6 Nov 2026 or later. Runs on the laptop only.

Reads the held-out file only after its sha256 matches the expected one. Writes one row per held-out packet with the fields
the evaluator and the scorer need and NOTHING else from the file: id, set_id, case_type, family, gold (the verdict inside
`chosen`), packet_text, plus `family_seen` (the family also appears in the training split). `chosen` and `rejected` are not
copied. Exclusion lists are NOT in this file: the scorer takes them as separate files at scoring time (the main list is
hashed by Liezl after the two-reader rule; the first 20-id list is a sensitivity row).
The output is for evaluation only and must never be passed to a training function.

  python train-modal/make_eval_input.py --heldout PATH --heldout-sha HEX --train PATH --out PATH [--exclusions CSV]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(held: list[dict], train_families: set) -> list[dict]:
    out = []
    for r in held:
        v = json.loads(r["chosen"])["verdict"]
        if v not in ("clear", "flag"):
            raise SystemExit(f"REFUSED: {r['id']} has no valid verdict in chosen")
        out.append({"id": r["id"], "set_id": r["set_id"], "case_type": r["case_type"], "family": r["family"], "gold": v,
                    "family_seen": r["family"] in train_families,
                    "packet_text": r["packet_text"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--heldout-sha", required=True)
    ap.add_argument("--exclusions", help="optional: only checked (ids must exist in held-out), not copied into the output")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if sha_file(a.heldout) != a.heldout_sha.lower():
        print("FAILED: the held-out file's sha256 differs from the expected one", file=sys.stderr)
        return 2
    held = [json.loads(line) for line in open(a.heldout, encoding="utf-8") if line.strip()]
    if a.exclusions:
        excl = {r["id"] for r in csv.DictReader(open(a.exclusions, encoding="utf-8-sig"))}
        unknown = excl - {r["id"] for r in held}
        if unknown:
            print("FAILED: exclusion ids not in held-out:", sorted(unknown)[:5], file=sys.stderr)
            return 2
    fams = {json.loads(line)["family"] for line in open(a.train, encoding="utf-8") if line.strip()}
    rows = build(held, fams)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "family_seen": sum(r["family_seen"] for r in rows),
                      "out_sha256": sha_file(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
