"""Write and verify freeze files. A freeze file holds the sha256 of every input and code file of a cell, the model IDs,
routes and token limits, the verdict rule and the price table. The runner refuses to start when any hash differs.

  python con-artist/freeze.py write  <cell> [--mode control|lever] [--source-cell CELL] [--cell-config PATH]
  python con-artist/freeze.py verify <cell>
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = ["providers.py", "packets.py", "parse.py", "runlog.py", "watchdog.py", "attacker.py", "checker.py",
        "score.py", "freeze.py", "run.py", "configs/settings.json"]
VERDICT_RULE = "cleared = parse_ok and verdict == clear; a parse failure or empty reply is never a clear"


def root() -> Path:
    return Path(os.environ.get("REPO_ROOT", HERE.parent))


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_cfg(cell_config: str | None = None) -> dict:
    """Base settings, with the attacker block of a cell config merged over it (one lever per cell)."""
    cfg = json.loads((HERE / "configs/settings.json").read_text(encoding="utf-8"))
    if cell_config:
        over = json.loads((root() / cell_config).read_text(encoding="utf-8"))
        cfg["attacker"].update(over.get("attacker", {}))
    return cfg


def collect(cfg: dict, cell_config: str | None = None) -> dict:
    r = root()
    files = {"packets": cfg["data"]["packets"], "attacker_prompt": cfg["attacker"]["prompt"],
             "reviewer_prompt": cfg["reviewer"]["prompt"]}
    files.update({f"prompt_{k}": v for k, v in cfg["data"]["prompts"].items()})
    if cell_config:
        files["cell_config"] = cell_config
    files["reviewer_calibration"] = "con-artist/calibration/reviewer_calibration_v1.jsonl"
    out = {k: {"path": v, "sha256": sha_file(r / v)} for k, v in files.items()}
    for c in CODE:
        out["code:" + c] = {"path": "con-artist/" + c, "sha256": sha_file(HERE / c)}
    return out


def git_info() -> dict:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root(), capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "con-artist"], cwd=root(), capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return {"commit": h or None, "con_artist_dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "con_artist_dirty": None}


def write(cell: str, mode: str, source_cell: str | None = None, cell_config: str | None = None) -> Path:
    cfg = load_cfg(cell_config)
    d = {"cell": cell, "mode": mode, "cell_config": cell_config, "files": collect(cfg, cell_config),
         "models": {k: {"route": v["route"], "model": v["model"], "effort": v["effort"],
                        "max_tokens": v["max_tokens_lever" if mode == "lever" else "max_tokens_control"]}
                    for k, v in cfg["watchdogs"].items()},
         "attacker": cfg["attacker"], "reviewer": cfg["reviewer"], "rounds": cfg["rounds"],
         "confirm": [cfg["confirm_needed"], cfg["confirm_reps"]], "null_reps": cfg["null_reps"],
         "verdict_rule": VERDICT_RULE, "prices_out_per_million_usd": cfg["prices_out_per_million_usd"],
         "prices_in_per_million_usd": cfg.get("prices_in_per_million_usd", {}),
         "git": git_info()}
    if source_cell:
        src = HERE / "runs" / source_cell / "log.jsonl"
        d["source_log"] = {"cell": source_cell, "sha256": sha_file(src)}
    out = HERE / "freezes" / f"{cell}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return out


def verify(path: Path, mode: str | None = None) -> list[str]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cc = d.get("cell_config")
    cfg = load_cfg(cc)
    bad = []
    now = collect(cfg, cc)
    if cfg["attacker"] != d["attacker"]:
        bad.append("attacker settings differ from the frozen ones")
    for k, v in d["files"].items():
        if k not in now:
            bad.append(f"{k}: missing now")
        elif now[k]["sha256"] != v["sha256"]:
            bad.append(f"{k}: hash differs ({v['path']})")
    if mode and mode != d["mode"]:
        bad.append(f"mode differs: frozen {d['mode']}, asked {mode}")
    return bad


def main(argv):
    if len(argv) < 2 or argv[0] not in ("write", "verify"):
        print(__doc__)
        return 2
    cell = argv[1]
    if argv[0] == "write":
        mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "control"
        sc = argv[argv.index("--source-cell") + 1] if "--source-cell" in argv else None
        cc = argv[argv.index("--cell-config") + 1] if "--cell-config" in argv else None
        print("wrote", write(cell, mode, sc, cc))
        return 0
    bad = verify(HERE / "freezes" / f"{cell}.json")
    print("freeze OK" if not bad else "freeze FAILED:\n  " + "\n  ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
