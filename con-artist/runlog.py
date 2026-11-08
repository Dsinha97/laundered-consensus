"""Append-only JSONL log with resume. A record with no error is "done" and is reused on a restart; an error record is
written for the audit trail and tried again. Every record carries not_for_training = true (5 Nov rule: no con-artist or
held-out case goes into any training split until Liezl agrees to change it)."""
from __future__ import annotations

import asyncio
import json
import os


class RunLog:
    def __init__(self, path: str):
        self.path = path
        self.done: dict[str, dict] = {}
        self.lock = asyncio.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if not r.get("error"):
                            self.done[r["key"]] = r

    async def get_or_run(self, key: str, fn, meta: dict | None = None) -> dict:
        if key in self.done:
            return self.done[key]
        rec = {"key": key, **(meta or {}), **(await fn()), "not_for_training": True}
        async with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        if not rec.get("error"):
            self.done[key] = rec
        return rec

    def records(self) -> list[dict]:
        out = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                out = [json.loads(line) for line in f if line.strip()]
        return out
