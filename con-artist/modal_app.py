"""Modal entry points for the con-artist cells (sprint work, 6 Nov 2026).

Keys stay in Modal Secrets referenced by name (adaption, openrouter, anthropic-secret, gemini-secret); nothing is read from the laptop. Run logs go to the
Volume "con-artist-runs"; copy them back with
  PYTHONUTF8=1 .venv/Scripts/modal volume get con-artist-runs <cell> con-artist/runs/

  PYTHONUTF8=1 .venv/Scripts/modal run con-artist/modal_app.py::smoke
  PYTHONUTF8=1 .venv/Scripts/modal run con-artist/modal_app.py --stage control --cell ctl1 --extra "--freeze con-artist/freezes/ctl1.json"

Only the packet file, the three prompt files and the pilot outputs are shipped from plans/prompts/. The held-out
generation prompt and every held-out file stay on the laptop.
"""
from __future__ import annotations

import shlex
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REMOTE = "/root/repo"

app = modal.App("con-artist")
vol = modal.Volume.from_name("con-artist-runs", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("httpx==0.28.1")
         .add_local_dir(str(HERE), f"{REMOTE}/con-artist", ignore=["runs/**", "tests/**", "__pycache__/**", "*.pyc"])
         .add_local_file(str(ROOT / "plans/prompts/pilot_packets_87_with_labels.jsonl"),
                         f"{REMOTE}/plans/prompts/pilot_packets_87_with_labels.jsonl")
         .add_local_file(str(ROOT / "plans/prompts/outputs/pilot_outputs_all.csv"),
                         f"{REMOTE}/plans/prompts/outputs/pilot_outputs_all.csv")
         .add_local_dir(str(ROOT / "plans/prompts/prompts"), f"{REMOTE}/plans/prompts/prompts"))


@app.function(image=image, secrets=[modal.Secret.from_name("adaption"), modal.Secret.from_name("openrouter"),
                       modal.Secret.from_name("anthropic-secret"), modal.Secret.from_name("gemini-secret")],
              volumes={"/runs": vol}, timeout=60 * 60 * 4)
def run(argv: list[str]) -> int:
    import os
    import sys

    os.environ["REPO_ROOT"] = REMOTE
    sys.path.insert(0, f"{REMOTE}/con-artist")
    import run as runner

    cell = argv[argv.index("--cell") + 1]
    rc = runner.main(argv + ["--run-dir", f"/runs/{cell}"])
    vol.commit()
    return rc


@app.local_entrypoint()
def smoke():
    rc = run.remote(["smoke", "--cell", "smoke1", "--spend-cap", "0.5"])
    print("smoke exit code:", rc)


@app.local_entrypoint()
def main(stage: str, cell: str, extra: str = ""):
    rc = run.remote([stage, "--cell", cell, *shlex.split(extra)])
    print("exit code:", rc)


@app.function(image=image, secrets=[modal.Secret.from_name("adaption"), modal.Secret.from_name("openrouter"),
                                    modal.Secret.from_name("anthropic-secret"), modal.Secret.from_name("gemini-secret")],
              timeout=300)
def ping_impl(route: str, model: str, effort: str = "", max_tokens: int = 50) -> dict:
    """One tiny call (50 output tokens unless told) to check a route, a model name and the reply shape. Prints no key."""
    import asyncio
    import sys

    sys.path.insert(0, f"{REMOTE}/con-artist")
    from providers import Providers

    r = asyncio.run(Providers().call(route, model, "Answer in one short sentence.", "Say that the route works.", max_tokens,
                                     effort=effort or None))
    return {k: v for k, v in r.to_dict().items()}


@app.local_entrypoint()
def ping(route: str, model: str, effort: str = "", max_tokens: int = 50):
    print(ping_impl.remote(route, model, effort, max_tokens))


@app.function(image=image, secrets=[modal.Secret.from_name("adaption"), modal.Secret.from_name("openrouter"),
                                    modal.Secret.from_name("anthropic-secret"), modal.Secret.from_name("gemini-secret")],
              volumes={"/runs": vol}, timeout=60 * 60 * 3)
def run_chain(argv: list[str]) -> int:
    """Live-chain cells: con-artist/chain.py run CELL ... (see that file)."""
    import os
    import sys

    os.environ["REPO_ROOT"] = REMOTE
    sys.path.insert(0, f"{REMOTE}/con-artist")
    import chain as chain_runner

    cell = argv[argv.index("run") + 1]
    rc = chain_runner.main(argv + ["--run-dir", f"/runs/{cell}"])
    vol.commit()
    return rc


@app.function(image=image, secrets=[modal.Secret.from_name("adaption"), modal.Secret.from_name("openrouter"),
                                    modal.Secret.from_name("anthropic-secret"), modal.Secret.from_name("gemini-secret")],
              volumes={"/runs": vol}, timeout=60 * 60 * 3)
def run_rejudge(argv: list[str]) -> int:
    """Blind-watchdog re-judge (cell ch4): con-artist/rejudge.py run CELL ... (reads the ch3 log on the volume, read only)."""
    import os
    import sys

    os.environ["REPO_ROOT"] = REMOTE
    sys.path.insert(0, f"{REMOTE}/con-artist")
    import rejudge as rejudge_runner

    cell = argv[argv.index("run") + 1]
    rc = rejudge_runner.main(argv + ["--run-dir", f"/runs/{cell}"])
    vol.commit()
    return rc


@app.local_entrypoint()
def rejudge(cell: str, extra: str = ""):
    rc = run_rejudge.remote(["run", cell, *shlex.split(extra)])
    print("exit code:", rc)


@app.local_entrypoint()
def chain(cell: str, stage: str = "chain", extra: str = ""):
    rc = run_chain.remote(["run", cell, "--stage", stage, *shlex.split(extra)])
    print("exit code:", rc)
