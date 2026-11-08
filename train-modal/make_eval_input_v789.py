"""Build the evaluation input from Liezl's Invent v789 outside-test set. Sprint work, 7 Nov 2026. Runs on the laptop only.

Reads the v789 file only after its sha256 matches the expected one. Writes one row per packet with the fields the evaluator
and the scorer need and NOTHING else from the file: id (the unique v789_id), orig_id, mechanism, case_type, round, src, gold,
packet_text (the file's `prompt`). `chosen` and `rejected` are not copied. Mechanism labels name the attack shape: the output
is for evaluation only, stays inside the team, and must never be passed to a training function.

A packet whose prompt text contains answer or label metadata (a "verdict", "case_type", "family" or "mechanism" JSON key) is
REFUSED unless its id is named with --exclude-id, or a fixed copy of it is supplied with --fixed-row (addendum 2, 7 Nov).
Excluded ids are listed in the summary, never silently dropped.

A fixed row (from Liezl's errata) replaces the original only if its sha256 matches (after CRLF to LF), every field except
`prompt` equals the original's, its prompt is the original prompt cut at the leftover text (a prefix), and no label or verdict
key remains in it.

  python train-modal/make_eval_input_v789.py --test PATH --sha HEX --out PATH [--exclude-id v789_id ...]
                                             [--fixed-row FILE --fixed-sha HEX]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ARTIFACT = re.compile(r'"(verdict|case_type|family|mechanism|evidence)"\s*:')


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def apply_fixes(rows: list[dict], fixes: dict) -> tuple[list[dict], list[str]]:
    """Replace rows by their fixed copies after checking that only the leftover text was removed."""
    by = {r["v789_id"]: r for r in rows}
    for vid, fx in fixes.items():
        o = by.get(vid)
        if o is None:
            raise SystemExit(f"REFUSED: fixed row {vid} is not in the test file")
        if any(o.get(k) != fx.get(k) for k in o if k != "prompt"):
            raise SystemExit(f"REFUSED: fixed row {vid} differs from the original in a field other than prompt")
        if not o["prompt"].startswith(fx["prompt"].rstrip()) or fx["prompt"].rstrip() == o["prompt"]:
            raise SystemExit(f"REFUSED: fixed row {vid} is not the original prompt cut at the end")
        if ARTIFACT.search(fx["prompt"]):
            raise SystemExit(f"REFUSED: fixed row {vid} still has label or verdict metadata")
    return [dict(fixes[r["v789_id"]], prompt=fixes[r["v789_id"]]["prompt"].rstrip()) if r["v789_id"] in fixes else r
            for r in rows], sorted(fixes)


def build(rows: list[dict], exclude: set, fixes: dict | None = None) -> tuple[list[dict], list[str]]:
    out, dropped, seen = [], [], set()
    fixes = fixes or {}
    rows, _ = apply_fixes(rows, fixes) if fixes else (rows, [])
    for r in rows:
        vid = r["v789_id"]
        if vid in seen:
            raise SystemExit(f"REFUSED: duplicate v789_id {vid}")
        seen.add(vid)
        gold = r["gold"]
        if gold not in ("clear", "flag") or r["chosen"].get("verdict") != gold:
            raise SystemExit(f"REFUSED: {vid} gold and chosen verdict disagree or are invalid")
        if ARTIFACT.search(r["prompt"]):
            if vid not in exclude:
                raise SystemExit(f"REFUSED: {vid} has answer or label metadata inside its prompt (name it with --exclude-id)")
            dropped.append(vid)
            continue
        out.append({"id": vid, "orig_id": r["id"], "mechanism": r["mechanism"], "case_type": r["case_type"],
                    "round": r["round"], "src": r["src"], "gold": gold, "packet_text": r["prompt"]})
    unused = exclude - set(dropped)
    if unused:
        raise SystemExit(f"REFUSED: --exclude-id names rows that were not flagged: {sorted(unused)}")
    return out, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-id", action="append", default=[])
    ap.add_argument("--fixed-row", action="append", default=[])
    ap.add_argument("--fixed-sha", action="append", default=[])
    a = ap.parse_args(argv)
    if len(a.fixed_row) != len(a.fixed_sha):
        print("FAILED: each --fixed-row needs a --fixed-sha", file=sys.stderr)
        return 2
    fixes = {}
    for fp, want in zip(a.fixed_row, a.fixed_sha):
        raw = Path(fp).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != want.lower():
            print(f"FAILED: fixed row file {fp} sha256 (LF-normalised) differs from the expected one", file=sys.stderr)
            return 2
        fx = json.loads(raw.decode("utf-8").splitlines()[0])
        fixes[fx["v789_id"]] = {k: fx[k] for k in ("v789_id", "id", "gold", "chosen", "rejected", "prompt", "mechanism",
                                                    "case_type", "round", "src")}
    if sha_file(a.test) != a.sha.lower():
        print("FAILED: the v789 file's sha256 differs from the expected one", file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in open(a.test, encoding="utf-8") if line.strip()]
    out, dropped = build(rows, set(a.exclude_id), fixes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"rows_in": len(rows), "rows_out": len(out), "excluded_for_metadata_in_prompt": dropped,
                      "replaced_by_fixed_copy": sorted(fixes),
                      "out_sha256": sha_file(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
