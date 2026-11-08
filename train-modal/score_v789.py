"""Scoring for the v789 outside test: wrong verdicts per mechanism, per model. Sprint work, 7 Nov 2026.

Input: eval JSONL files from modal_train.py `evaluate` on the v789 eval input (one record per row with id, mechanism, round,
gold, parse_ok, verdict). Rule as in score_eval.py: cleared = parse_ok and verdict == clear; a parse failure never counts as a
clear. Wrong = a missed violation (gold flag, cleared) or a false alarm (gold clear, parse_ok and verdict == flag).
Parse failures are counted apart and are in neither numerator. Rates carry Wilson 95% intervals; paired comparisons to the
first model use the exact McNemar test (score_eval.py, imported, not changed). Round-7 rows (round == "v7") are also
reported on their own, since the group already planned to report them apart.

  python train-modal/score_v789.py --out OUT.json MODEL=path.jsonl [MODEL=path.jsonl ...]   (first model is the reference)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_eval as S  # noqa: E402


def wrong_of(r: dict):
    """True / False / None (None = parse failure, not counted either way)."""
    if not r.get("parse_ok"):
        return None
    if r["gold"] == "flag":
        return r["verdict"] == "clear"
    return r["verdict"] == "flag"


def cell(rows: list[dict]) -> dict:
    w = [wrong_of(r) for r in rows]
    k, n = sum(1 for x in w if x is True), sum(1 for x in w if x is not None)
    d = S.wilson(k, n)
    d["parse_failures"] = sum(1 for x in w if x is None)
    d["rows"] = len(rows)
    return d


def score(by_model: dict, sig_ids: set | None = None) -> dict:
    out = {"models": {}, "paired_vs_first": {}}
    names = list(by_model)
    for m, rows in by_model.items():
        res = {"all": cell(rows), "round7_only": cell([r for r in rows if r["round"] == "v7"]),
               "without_round7": cell([r for r in rows if r["round"] != "v7"]), "by_mechanism": {}}
        if sig_ids is not None:
            res["signature_rule_rows"] = cell([r for r in rows if r["id"] in sig_ids])
            res["other_rows"] = cell([r for r in rows if r["id"] not in sig_ids])
            for g in ("flag", "clear"):
                res[f"signature_rule_gold_{g}"] = cell([r for r in rows if r["id"] in sig_ids and r["gold"] == g])
                res[f"other_gold_{g}"] = cell([r for r in rows if r["id"] not in sig_ids and r["gold"] == g])
        mech = defaultdict(list)
        for r in rows:
            mech[(r["mechanism"], r["gold"])].append(r)
        for (mc, g), rs in sorted(mech.items()):
            res["by_mechanism"][f"{mc} (gold {g})"] = cell(rs)
        out["models"][m] = res
    if len(names) > 1:
        ref = {r["id"]: wrong_of(r) for r in by_model[names[0]]}
        for m in names[1:]:
            cur = {r["id"]: wrong_of(r) for r in by_model[m]}
            ids = [i for i in ref if i in cur and ref[i] is not None and cur[i] is not None]
            b = sum(1 for i in ids if ref[i] and not cur[i])   # reference wrong, this model right
            c = sum(1 for i in ids if not ref[i] and cur[i])   # reference right, this model wrong
            out["paired_vs_first"][m] = {"reference": names[0], "paired_rows": len(ids), "ref_wrong_only": b,
                                         "model_wrong_only": c, "mcnemar_p": S.mcnemar_exact(b, c)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--signature-ids", help="id list from signature_split.py on the v789 eval input (one id per line)")
    ap.add_argument("--signature-ids-sha", help="expected SHA-256 of the id list as signature_split.py computes it")
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args(argv)
    by = {}
    for p in a.pairs:
        name, path = p.split("=", 1)
        by[name] = S.load(path)
    sig = None
    if a.signature_ids:
        import hashlib
        ids = sorted(x for x in Path(a.signature_ids).read_text(encoding="utf-8").split("\n") if x)
        if not a.signature_ids_sha or hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest() != a.signature_ids_sha.lower():
            raise SystemExit("REFUSED: the signature id list's SHA-256 differs from the expected one (or none was given)")
        sig = set(ids)
    res = score(by, sig)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({m: v["all"] for m, v in res["models"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
