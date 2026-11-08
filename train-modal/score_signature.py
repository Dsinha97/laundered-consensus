"""Pre-registered signature-rule split for the dpoC analysis. Sprint work, 7 Nov 2026.

Separate from score_eval.py on purpose: score_eval.py is part of the `evalH` freeze, so it is not edited. This script reuses its
functions (wilson, metrics, paired) and adds one thing: results per model on two groups, rows whose rule section matches the
signature pattern in signature_split.py (the id list is fixed and hashed before any held-out output is opened) and all other
rows, within the main subset (the final list) and over all rows. It also pairs dpoC with dpoB (McNemar) on the signature group.

  python train-modal/score_signature.py --eval base=PATH --eval sftA=PATH --eval dpoB=PATH --eval dpoC=PATH \\
      --signature-ids train-modal/eval_input/heldout_v2_signature_rule_ids.txt --signature-ids-sha HEX \\
      --final-list CSV --final-list-sha HEX --out FILE

The group is small (27 of 278 rows), so intervals are wide: it can flag a problem but not rule one out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score_eval as S  # noqa: E402


def read_ids(path) -> list[str]:
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").split("\n") if x.strip()]


def id_list_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def group_report(by_model: dict, sig_ids: set, excl_final: set) -> dict:
    names = list(by_model)
    keys = {n: {r["id"]: r for r in rs} for n, rs in by_model.items()}
    all_ids = set(keys[names[0]])
    main_ids = all_ids - excl_final
    out = {"models": names, "signature_rows_total": len(sig_ids & all_ids), "groups": {}, "paired_signature": {}}
    for scope, scope_ids in (("main_final_list", main_ids), ("all_rows", all_ids)):
        for gname, gids in (("signature_rule", scope_ids & sig_ids), ("other", scope_ids - sig_ids)):
            out["groups"][f"{scope}:{gname}"] = {
                "rows": len(gids), **{n: S.metrics([r for i, r in keys[n].items() if i in gids]) for n in names}}
        gids = scope_ids & sig_ids
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out["paired_signature"][f"{scope}:{a}__vs__{b}"] = S.paired(
                    [r for r in by_model[a] if r["id"] in gids], [r for r in by_model[b] if r["id"] in gids])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True, help="NAME=PATH of an eval JSONL (repeat per model)")
    ap.add_argument("--signature-ids", required=True)
    ap.add_argument("--signature-ids-sha", required=True, help="sha256 of the sorted, newline-joined id list (as logged)")
    ap.add_argument("--final-list", required=True)
    ap.add_argument("--final-list-sha", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sig = read_ids(a.signature_ids)
    if id_list_sha(sig) != a.signature_ids_sha.lower():
        print("REFUSED: the signature id list differs from the one fixed before scoring", file=sys.stderr)
        return 2
    if hashlib.sha256(open(a.final_list, "rb").read()).hexdigest() != a.final_list_sha.lower():
        print("REFUSED: the final list's sha256 differs from the one Liezl logged", file=sys.stderr)
        return 2
    by_model = {}
    for item in a.eval:
        name, path = item.split("=", 1)
        by_model[name] = S.load(path)
    res = group_report(by_model, set(sig), S.read_exclusions(a.final_list))
    res["signature_ids_sha256"] = a.signature_ids_sha.lower()
    res["final_list_sha256"] = a.final_list_sha.lower()
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
