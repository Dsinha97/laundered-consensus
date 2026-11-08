"""Label-sensitivity scoring for the v789 outside test (addendum 1 to the evalV789 freeze, 7 Nov 2026).

Runs score_v789.score on the eval records with the ids on an exclusion list removed, and prints the full-set and the
reduced-set cells side by side. The exclusion list must match the SHA-256 given (sha256 of the sorted ids joined by newlines).
The list was chosen after the base watchdogs' outcomes were known (blind readers read only packets that fooled 2+ base
watchdogs), so excluding it can only lower miss rates; read the two columns as a range, never replace the full-set row.
score_v789.py is imported unchanged.

  python train-modal/score_v789_sensitivity.py --exclude-ids FILE --exclude-ids-sha HEX --out OUT.json MODEL=path.jsonl ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_eval as S  # noqa: E402
import score_v789 as G  # noqa: E402


def sensitivity(by_model: dict, excl: set) -> dict:
    full = G.score(by_model)
    kept = {m: [r for r in rows if r["id"] not in excl] for m, rows in by_model.items()}
    reduced = G.score(kept)
    out = {"excluded_ids": len(excl), "full": {}, "reduced": {}}
    for m in by_model:
        for tag, res in (("full", full), ("reduced", reduced)):
            mm = res["models"][m]
            out[tag][m] = {"all": mm["all"],
                           "silent_record": mm["by_mechanism"].get("silent_record (gold flag)"),
                           "explained_gap_flag": mm["by_mechanism"].get("explained_gap_flag (gold flag)")}
    out["paired_vs_first_reduced"] = reduced["paired_vs_first"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude-ids", required=True)
    ap.add_argument("--exclude-ids-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args(argv)
    ids = sorted(x for x in Path(a.exclude_ids).read_text(encoding="utf-8").split("\n") if x)
    if hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest() != a.exclude_ids_sha.lower():
        raise SystemExit("REFUSED: the exclusion list's SHA-256 differs from the expected one")
    by = {}
    for p in a.pairs:
        name, path = p.split("=", 1)
        by[name] = S.load(path)
    res = sensitivity(by, set(ids))
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"excluded": res["excluded_ids"], "full": {m: v["silent_record"] for m, v in res["full"].items()},
                      "reduced": {m: v["silent_record"] for m, v in res["reduced"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
