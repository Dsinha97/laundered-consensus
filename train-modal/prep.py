"""Training-data prep for the Modal fallback fine-tune. Sprint work, 6 Nov 2026 or later.

Reads Liezl's training split (rows with packet_text, chosen, rejected), checks it, builds prompt/completion (SFT) and
prompt/chosen/rejected (DPO) records with the PLAIN system prompt, and splits train and dev by set_id.

Fail-closed rules:
- a path that contains "heldout" or "do_not_train" is refused (held-out v2 never reaches training code);
- a leak report must exist and show zero collisions (written by leak_check.py on the laptop);
- every chosen and rejected must pass the strict verdict parser, and chosen and rejected must disagree on the verdict;
- ids must be unique.

  python train-modal/prep.py --train PATH --leak-report train-modal/data/leak_report.json --out train-modal/data
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAIN_PROMPT = ROOT / "plans/prompts/prompts/prompt_plain.txt"
PARSE_PATH = ROOT / "con-artist/parse.py"
FORBIDDEN = ("heldout", "held_out", "do_not_train")


def load_parse():
    spec = importlib.util.spec_from_file_location("ca_parse", PARSE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def refuse_forbidden(path) -> None:
    low = str(path).lower().replace("\\", "/")
    if any(w in low for w in FORBIDDEN):
        raise SystemExit(f"REFUSED: {path} looks like held-out material and must not reach training code")


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rows(path) -> list[dict]:
    refuse_forbidden(path)
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(rows: list[dict], parse=None) -> list[str]:
    """Problems found in the rows. An empty list means the rows are usable."""
    parse = parse or load_parse()
    problems, seen = [], set()
    for r in rows:
        rid = r.get("id")
        if not rid or rid in seen:
            problems.append(f"duplicate or missing id: {rid}")
        seen.add(rid)
        for k in ("packet_text", "chosen", "rejected", "set_id"):
            if not r.get(k):
                problems.append(f"{rid}: missing {k}")
        if not (r.get("chosen") and r.get("rejected")):
            continue
        c, j = parse.parse_strict(r["chosen"]), parse.parse_strict(r["rejected"])
        if c is None:
            problems.append(f"{rid}: chosen does not pass the strict parser")
        if j is None:
            problems.append(f"{rid}: rejected does not pass the strict parser")
        if c and j and c["verdict"] == j["verdict"]:
            problems.append(f"{rid}: chosen and rejected have the same verdict")
    return problems


def split_by_set(rows: list[dict], seed: int = 0, dev_fraction: float = 0.05):
    """All rows of a set_id go to the same side. Dev rows are training data, never held-out data."""
    def is_dev(set_id: str) -> bool:
        h = int(hashlib.sha256(f"{seed}|{set_id}".encode()).hexdigest(), 16)
        return (h % 10000) < dev_fraction * 10000
    train = [r for r in rows if not is_dev(r["set_id"])]
    dev = [r for r in rows if is_dev(r["set_id"])]
    return train, dev


def messages(system: str, packet_text: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": packet_text}]


def sft_record(r: dict, system: str) -> dict:
    return {"id": r["id"], "prompt": messages(system, r["packet_text"]),
            "completion": [{"role": "assistant", "content": r["chosen"]}]}


def dpo_record(r: dict, system: str) -> dict:
    return {"id": r["id"], "prompt": messages(system, r["packet_text"]),
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}]}


def check_leak_report(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"REFUSED: no leak report at {path}; run leak_check.py first")
    rep = json.loads(p.read_text(encoding="utf-8"))
    if rep.get("exact_matches") != 0 or rep.get("near_matches") != 0 or rep.get("status") != "pass":
        raise SystemExit("REFUSED: the leak report does not show zero exact and zero near-duplicate matches")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--leak-report", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-fraction", type=float, default=0.05)
    ap.add_argument("--validate-only", action="store_true",
                    help="check the rows and show the split sizes; write nothing, need no leak report")
    a = ap.parse_args(argv)
    rows = load_rows(a.train)
    if a.validate_only:
        problems = validate(rows)
        train, dev = split_by_set(rows, a.seed, a.dev_fraction)
        print(json.dumps({"rows": len(rows), "problems": len(problems), "train": len(train), "dev": len(dev),
                          "train_sets": len({r["set_id"] for r in train}), "dev_sets": len({r["set_id"] for r in dev}),
                          "train_sha256": sha_file(a.train)}))
        print(*problems[:10], sep="\n", file=sys.stderr)
        return 0 if not problems else 2
    if not a.leak_report or not a.out:
        raise SystemExit("REFUSED: --leak-report and --out are required unless --validate-only is given")
    rep = check_leak_report(a.leak_report)
    if rep.get("train_sha256") != sha_file(a.train):
        raise SystemExit("REFUSED: the leak report was made for a different training file")
    problems = validate(rows)
    if problems:
        print(f"{len(problems)} problem(s); first 10:", *problems[:10], sep="\n  ", file=sys.stderr)
        return 2
    system = PLAIN_PROMPT.read_text(encoding="utf-8-sig")
    train, dev = split_by_set(rows, a.seed, a.dev_fraction)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rs in (("train", train), ("dev", dev)):
        for kind, fn in (("sft", sft_record), ("dpo", dpo_record)):
            with open(out / f"{name}_{kind}.jsonl", "w", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(fn(r, system), ensure_ascii=False) + "\n")
    summary = {"rows": len(rows), "train": len(train), "dev": len(dev), "train_sha256": sha_file(a.train),
               "plain_prompt_sha256": sha_file(PLAIN_PROMPT), "parse_py_sha256": sha_file(PARSE_PATH),
               "seed": a.seed, "dev_fraction": a.dev_fraction}
    (out / "prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("prep OK:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
