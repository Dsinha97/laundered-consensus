"""Write and verify freeze files for training and evaluation runs. Sprint work, 6 Nov 2026 or later.

A freeze holds the sha256 of every code, config, prompt and data file of a run, plus pinned versions and the verdict rule.
The held-out file is recorded as its expected sha256 string only; this module never opens it. Run it before a run starts;
the Modal functions call verify first and stop on any difference.

  python train-modal/freeze_train.py write  RUN CONFIG_NAME [--data-dir train-modal/data] [--heldout-sha HEX] [--extra JSON_FILE]
  python train-modal/freeze_train.py verify RUN
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("REPO_ROOT", HERE.parent))
CODE = ["prep.py", "leak_check.py", "make_eval_input.py", "freeze_train.py", "score_eval.py", "modal_train.py"]
VERDICT_RULE = "cleared = parse_ok and verdict == clear (strict parser of con-artist/parse.py); a parse failure never counts as a clear"


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def collect(config_name: str, data_dir: Path) -> dict:
    files = {"config": HERE / "config" / f"{config_name}.json",
             "plain_prompt": ROOT / "plans/prompts/prompts/prompt_plain.txt",
             "parse_py": ROOT / "con-artist/parse.py"}
    for n in CODE:
        if (HERE / n).exists():
            files["code:" + n] = HERE / n
    for n in ("train_sft.jsonl", "dev_sft.jsonl", "train_dpo.jsonl", "dev_dpo.jsonl", "leak_report.json", "prep_summary.json"):
        if (data_dir / n).exists():
            files["data:" + n] = data_dir / n
    def rel(v: Path) -> str:
        try:
            return str(v.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(v)

    return {k: {"path": rel(v), "sha256": sha_file(v)} for k, v in files.items()}


def write(run: str, config_name: str, data_dir: Path, heldout_sha: str | None, extra: dict | None = None) -> Path:
    cfg = json.loads((HERE / "config" / f"{config_name}.json").read_text(encoding="utf-8"))
    d = {"run": run, "config_name": config_name, "model_id": cfg["model_id"], "revision": cfg.get("revision"),
         "data_dir": str(data_dir), "files": collect(config_name, data_dir), "heldout_v2_sha256_expected": heldout_sha,
         "verdict_rule": VERDICT_RULE, "seed": cfg["seed"], "extra": extra or {}}
    try:
        d["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                         timeout=10).stdout.strip() or None
    except Exception:
        d["git_commit"] = None
    out = HERE / "freezes" / f"{run}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return out


def verify(path: Path) -> list[str]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    now = collect(d["config_name"], Path(d["data_dir"]))
    bad = []
    for k, v in d["files"].items():
        if k not in now:
            bad.append(f"{k}: missing now")
        elif now[k]["sha256"] != v["sha256"]:
            bad.append(f"{k}: hash differs ({v['path']})")
    return bad


def main(argv):
    if len(argv) < 2 or argv[0] not in ("write", "verify"):
        print(__doc__)
        return 2
    if argv[0] == "write":
        if len(argv) < 3:
            print(__doc__)
            return 2
        dd = Path(argv[argv.index("--data-dir") + 1]) if "--data-dir" in argv else HERE / "data"
        hs = argv[argv.index("--heldout-sha") + 1] if "--heldout-sha" in argv else None
        ex = json.loads(Path(argv[argv.index("--extra") + 1]).read_text(encoding="utf-8")) if "--extra" in argv else None
        print("wrote", write(argv[1], argv[2], dd, hs, ex))
        return 0
    bad = verify(HERE / "freezes" / f"{argv[1]}.json")
    print("freeze OK" if not bad else "freeze FAILED:\n  " + "\n  ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
