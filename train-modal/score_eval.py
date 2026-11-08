"""Scoring for the base-vs-tuned comparison. Sprint work, 6 Nov 2026 or later.

Input: eval JSONL files with one record per row: {id, case_type, family, gold, raw_output, parse_ok, verdict}.
Metrics follow the con-artist rule (cleared = parse_ok and verdict == clear). Rates carry Wilson 95% intervals; paired
comparisons use the exact two-sided McNemar test on rows where the two models differ. Cells with fewer than 10 cases are
marked "small n". The gold label field for held-out v2 must be supplied by Liezl and frozen before use.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"k": k, "n": n, "rate": None, "lo": None, "hi": None, "small_n": True}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "rate": round(p, 4), "lo": round(max(0, c - h), 4), "hi": round(min(1, c + h), 4),
            "small_n": n < 10}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_clear(r: dict) -> bool:
    return bool(r.get("parse_ok")) and r.get("verdict") == "clear"


def metrics(rows: list[dict]) -> dict:
    flag_rows = [r for r in rows if r["gold"] == "flag"]
    clear_rows = [r for r in rows if r["gold"] == "clear"]
    out = {
        "rows": len(rows),
        "parse_failures": wilson(sum(not r.get("parse_ok") for r in rows), len(rows)),
        "missed_violations": wilson(sum(is_clear(r) for r in flag_rows), len(flag_rows)),
        "false_alarms": wilson(sum(bool(r.get("parse_ok")) and r.get("verdict") == "flag" for r in clear_rows),
                               len(clear_rows)),
        "lenient_verdict_disagreements": sum(1 for r in rows if r.get("lenient_verdict") not in (None, r.get("verdict"))),
    }
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r.get("family")].append(r)
    out["by_family"] = {f: {"missed_violations": wilson(sum(is_clear(r) for r in rs if r["gold"] == "flag"),
                                                        sum(1 for r in rs if r["gold"] == "flag")),
                            "false_alarms": wilson(sum(bool(r.get("parse_ok")) and r.get("verdict") == "flag"
                                                       for r in rs if r["gold"] == "clear"),
                                                   sum(1 for r in rs if r["gold"] == "clear"))}
                        for f, rs in sorted(by_fam.items(), key=lambda kv: str(kv[0]))}
    return out


def paired(a_rows: list[dict], b_rows: list[dict]) -> dict:
    """Wrong-answer comparison on the same rows: a row is wrong if the verdict is not the gold verdict (parse failure = wrong)."""
    bm = {r["id"]: r for r in b_rows}
    only_a = only_b = both = neither = 0
    for r in a_rows:
        o = bm.get(r["id"])
        if o is None:
            continue
        wa = not (r.get("parse_ok") and r.get("verdict") == r["gold"])
        wb = not (o.get("parse_ok") and o.get("verdict") == o["gold"])
        if wa and wb:
            both += 1
        elif wa:
            only_a += 1
        elif wb:
            only_b += 1
        else:
            neither += 1
    return {"a_wrong_b_right": only_a, "a_right_b_wrong": only_b, "both_wrong": both, "both_right": neither,
            "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 4)}


# ---------------------------------------------------------------- subsets by exclusion list, complete triplets, seen families
def read_exclusions(path) -> set:
    """Ids of a CSV with an `id` column (extra columns such as status and reason are ignored)."""
    import csv
    with open(path, encoding="utf-8-sig") as f:
        return {r["id"] for r in csv.DictReader(f)}


def subsets(rows: list[dict], excl_final: set, excl_a: set) -> dict:
    """Named row subsets. A complete triplet is a set whose supported row is not excluded: its supported, laundered and
    near-miss rows all stay (benign and plain-violation sets are not triplets)."""
    def triplets(excl):
        keep_sets = {r["set_id"] for r in rows if r["case_type"] == "supported" and r["id"] not in excl}
        return [r for r in rows if r["case_type"] in ("supported", "laundered", "near_miss") and r["set_id"] in keep_sets]

    return {"main_final_list": [r for r in rows if r["id"] not in excl_final],
            "sens_first_20_list": [r for r in rows if r["id"] not in excl_a],
            "sens_all_rows": list(rows),
            "sens_complete_triplets_final": triplets(excl_final),
            "sens_complete_triplets_first_20": triplets(excl_a)}


def report(by_model: dict, excl_final: set, excl_a: set) -> dict:
    """by_model: name -> eval rows (same ids in each). Metrics per subset, split by family_seen on the main subset, and
    paired McNemar between every pair of models on the main subset."""
    names = list(by_model)
    base_rows = by_model[names[0]]
    out = {"models": names, "n_rows": len(base_rows), "excluded_final": len(excl_final), "excluded_first_20": len(excl_a),
           "subsets": {}, "main_by_family_seen": {}, "paired_on_main": {}}
    keys = {n: {r["id"]: r for r in rs} for n, rs in by_model.items()}
    subs = subsets(base_rows, excl_final, excl_a)
    for sname, srows in subs.items():
        ids = {r["id"] for r in srows}
        out["subsets"][sname] = {"rows": len(ids),
                                 **{n: metrics([r for i, r in keys[n].items() if i in ids]) for n in names}}
    main_ids = {r["id"] for r in subs["main_final_list"]}
    for label, flag in (("seen_families", True), ("unseen_families", False)):
        ids = {r["id"] for r in base_rows if bool(r.get("family_seen")) is flag and r["id"] in main_ids}
        out["main_by_family_seen"][label] = {"rows": len(ids),
                                              **{n: metrics([r for i, r in keys[n].items() if i in ids]) for n in names}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["paired_on_main"][f"{a}__vs__{b}"] = paired([r for r in by_model[a] if r["id"] in main_ids],
                                                             [r for r in by_model[b] if r["id"] in main_ids])
    return out


def main(argv=None) -> int:
    import argparse
    import hashlib
    ap = argparse.ArgumentParser(description="Score base, SFT and DPO evaluation files.")
    ap.add_argument("--eval", action="append", required=True, help="NAME=PATH of an eval JSONL (repeat per model)")
    ap.add_argument("--final-list", required=True, help="final exclusions CSV (id, status, reason); its sha256 must be given")
    ap.add_argument("--final-list-sha", required=True)
    ap.add_argument("--first-list", required=True, help="the first 20-id exclusions CSV")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if hashlib.sha256(open(a.final_list, "rb").read()).hexdigest() != a.final_list_sha.lower():
        print("REFUSED: the final list's sha256 differs from the one Liezl logged", file=__import__("sys").stderr)
        return 2
    by_model = {}
    for item in a.eval:
        name, path = item.split("=", 1)
        by_model[name] = load(path)
    res = report(by_model, read_exclusions(a.final_list), read_exclusions(a.first_list))
    res["final_list_sha256"] = a.final_list_sha.lower()
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
