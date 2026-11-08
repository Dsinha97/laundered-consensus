"""Run one watchdog on one packet. The system message is the prompt file as sent, the user message is the packet text,
exactly as in the pilot (bundle README, 6 Nov)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from parse import parse_verdict


def read_text(path) -> str:
    return Path(path).read_text(encoding="utf-8-sig")


def load_prompts(root: Path, paths: dict) -> dict:
    """name -> {text, sha256}. The text is passed through unchanged (no strip), as her runner does."""
    out = {}
    for name, rel in paths.items():
        t = read_text(root / rel)
        out[name] = {"text": t, "sha256": hashlib.sha256(t.encode("utf-8")).hexdigest()}
    return out


def max_tokens_for(wd: dict, mode: str) -> int:
    return wd["max_tokens_lever" if mode == "lever" else "max_tokens_control"]


async def judge(P, wd: dict, system: str, packet_text: str, mode: str) -> dict:
    r = await P.call(wd["route"], wd["model"], system, packet_text, max_tokens_for(wd, mode), effort=wd.get("effort"))
    d = r.to_dict()
    if not r.error:
        d.update(parse_verdict(r.text))
    return d
